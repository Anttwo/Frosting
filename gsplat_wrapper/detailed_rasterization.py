import math
import torch
import torch.nn as nn
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_indices_in_range,
    spherical_harmonics,
)
from nerfacc import accumulate_along_rays, render_weight_from_alpha
from .utils import fov2focal, SH2RGB


def get_intrinsics_for_gsplat(fx, fy, width, height):
    return torch.tensor(
        [
            [fx, 0, width / 2],
            [0, fy, height / 2],
            [0, 0, 1],
        ]
    )
    
    
def combine_colors_from_projected_gaussians(
    means2d,  # [C, N, 2]
    colors,  # [C, N, 3]
    conics,  # [C, N, 3]
    opacities,  # [C, N]
    pixel_ids,  # [M]
    gs_ids,  # [M]
    camera_ids,  # [M]
    width:int,
    height:int,
    return_accumulated_alphas:bool = False,
):
    C = means2d.shape[0]
    channels = colors.shape[-1]

    # Computing alpha values for each pixel-gaussian intersection, 
    # using (a) the pixel coordinates and (b) the projected 2D Gaussian.
    pixel_ids_x = pixel_ids % width
    pixel_ids_y = pixel_ids // width
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = pixel_coords - means2d[camera_ids, gs_ids]  # [M, 2]
    c = conics[camera_ids, gs_ids]  # [M, 3]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
        + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    alphas = torch.clamp_max(
        opacities[camera_ids, gs_ids] * torch.exp(-sigmas), 0.999
    )

    # Compute render weights and transmittance
    indices = camera_ids * height * width + pixel_ids
    total_pixels = C * height * width
    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels
    )
    
    # Integrate color along rays
    renders = accumulate_along_rays(
        weights,
        colors[camera_ids, gs_ids],
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(C, height, width, channels)
    if return_accumulated_alphas:
        # Integrate alpha along rays
        alphas = accumulate_along_rays(
            weights, None, ray_indices=indices, n_rays=total_pixels
        ).reshape(C, height, width, 1)
        return renders, alphas
    else:
        return renders, None
    
    
class GaussianRasterizationSettings:
    def __init__(
        self,
        image_height:int,
        image_width:int,
        tanfovx:float,
        tanfovy:float,
        bg:torch.Tensor,
        scale_modifier:float,
        viewmatrix:torch.Tensor,
        projmatrix:torch.Tensor,
        sh_degree:int,
        campos:torch.Tensor,
        prefiltered:bool,
        debug:bool,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.tanfovx = tanfovx
        self.tanfovy = tanfovy
        self.bg = bg
        self.scale_modifier = scale_modifier
        self.viewmatrix = viewmatrix
        self.projmatrix = projmatrix
        self.sh_degree = sh_degree
        self.campos = campos
        self.prefiltered = prefiltered
        self.debug = debug


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super(GaussianRasterizer, self).__init__()
        self.raster_settings = raster_settings
        
        self.image_height = raster_settings.image_height
        self.image_width = raster_settings.image_width
        self.tanfovx = raster_settings.tanfovx
        self.tanfovy = raster_settings.tanfovy
        self.bg = raster_settings.bg
        self.scale_modifier = raster_settings.scale_modifier
        self.viewmatrix = raster_settings.viewmatrix
        self.projmatrix = raster_settings.projmatrix
        self.sh_degree = raster_settings.sh_degree
        self.campos = raster_settings.campos
        self.prefiltered = raster_settings.prefiltered
        self.debug = raster_settings.debug
        
        self.fx = fov2focal(2. * math.atan(self.tanfovx), self.image_width)
        self.fy = fov2focal(2. * math.atan(self.tanfovy), self.image_height)
        
        self.K = get_intrinsics_for_gsplat(
            self.fx, self.fy, 
            self.image_width, self.image_height
        )[None].to(self.viewmatrix.device)
        
        if len(self.bg.shape) == 1:
            self.bg = self.bg[None].repeat(self.K.shape[0], 1)
        
    def forward(
        self, means3D, means2D,
        shs, colors_precomp, opacities,
        scales, rotations, cov3D_precomp,
        verbose=False,
        ):
    
        # Inputs
        means = means3D
        quats = rotations
        scales = scales
        viewmats = self.viewmatrix.transpose(-1, -2)[None]
        Ks = self.K
        height = self.image_height
        width = self.image_width
        eps2d = 0.3
        near_plane = 0.01
        far_plane = 1e10
        packed = True
        sparse_grad = False
        rasterize_mode = "classic"
        
        radius_clip = 0.
        tile_size = 16
        
        C = viewmats.shape[0]
        N = means.shape[0]
        
        # -----Step 1: Projecting Gaussians to image plane-----
        proj_results = fully_fused_projection(
            means,
            None,  # covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            packed=packed,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            sparse_grad=sparse_grad,
            calc_compensations=(rasterize_mode == "antialiased"),
        )
        (camera_ids,  # Gives, for each projected Gaussian, the corresponding camera index.
        gaussian_ids,  # Gives, for each projected Gaussian, the corresponding Gaussian index.
        radii,  # Gives, for each projected Gaussian, the max radius of the projection.
        means2d,  # Gives, for each projected Gaussian, the 2D coordinates of the projection.
        depths,  # Gives, for each projected Gaussian, the depth of the projection.
        conics,  # Gives, for each projected Gaussian, the inverse of the projected covariance with only upper triangle values.
        compensations  # Gives, for each projected Gaussian, the view-dependent opacity compensation factor for antialiased rendering.
        ) = proj_results
        _opacities = opacities[..., 0][gaussian_ids]  # Gives, for each projected Gaussian, the corresponding opacity.
        if compensations is not None:
            _opacities = _opacities * compensations
        if verbose:
            print("Finished step 1.")

        # -----Step 2: Identify intersecting tiles-----
        tile_width = math.ceil(width / float(tile_size))
        tile_height = math.ceil(height / float(tile_size))
        (tiles_per_gauss, # Gives, for each projected Gaussian, the number of intersected tiles. Shape ()
        isect_ids,  # Gives, for each tile intersection, the indices of both the corresponding camera and tile, and the corresponding depth  
        flatten_ids,  # Gives, for each tile intersection, the index of the corresponding projected Gaussian
        ) = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=packed,
            n_cameras=C,
            camera_ids=camera_ids,
            gaussian_ids=gaussian_ids,
        )
        isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
        if verbose:
            print("Finished step 2.")
        
        # -----Step 3: Get pixel-wise indices-----
        n_isects = len(flatten_ids)
        block_size = tile_size * tile_size
        isect_offsets_fl = torch.cat(
            [isect_offsets.flatten(), torch.tensor([n_isects], device=means.device)]
        )
        max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
        num_batches = (max_range + block_size - 1) // block_size
        transmittances = torch.ones((C, height, width), device=means.device) * 1.
        (gs_ids,  # Gives, for each pixel-gaussian intersection, the index of the corresponding projected Gaussian
        pixel_ids,  # Gives, for each pixel-gaussian intersection, the index of the corresponding pixel
        cam_ids,  # Gives, for each pixel-gaussian intersection, the index of the corresponding camera
        ) = rasterize_to_indices_in_range(
            range_start=0,
            range_end=num_batches,
            transmittances=transmittances,  # [C, H, W]
            means2d=means2d[None],  # [C, N, 2]
            conics=conics[None],  # [C, N, 3]
            opacities=_opacities[None],  # [C, N]
            image_width=width,
            image_height=height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,  # [C, tile_height, tile_width]
            flatten_ids=flatten_ids,  # [n_isects]
        )
        if verbose:
            print("Finished step 3.")
        
        # -----Step 4: Compute colors-----
        if colors_precomp is not None:
            if colors_precomp.dim() == 2:
                # Turn [N, D] into [nnz, D]
                colors = colors_precomp[gaussian_ids]
            else:
                # Turn [C, N, D] into [nnz, D]
                colors = colors_precomp[camera_ids, gaussian_ids]
        else:
            # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
            camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
            dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
            masks = radii > 0  # [nnz]
            if shs.dim() == 3:
                # Turn [N, K, 3] into [nnz, 3]
                _shs = shs[gaussian_ids, :, :]  # [nnz, K, 3]
            else:
                # Turn [C, N, K, 3] into [nnz, 3]
                _shs = shs[camera_ids, gaussian_ids, :, :]  # [nnz, K, 3]
            colors = spherical_harmonics(self.sh_degree, dirs, _shs, masks=masks)  # [nnz, 3]
            colors = torch.clamp_min(colors + 0.5, 0.0)
        if verbose:
            print("Finished step 4.")
            
        # -----Step 5: Rendering-----
        render_colors, _ = combine_colors_from_projected_gaussians(
            means2d=means2d[None],  # [C, N, 2]
            colors=colors[None],  # [C, N, 3]
            conics=conics[None],  # [C, N, 3]
            opacities=_opacities[None],  # [C, N]
            pixel_ids=pixel_ids,  # [M]
            gs_ids=gs_ids,  # [M]
            camera_ids=cam_ids,  # [M]
            width=width,
            height=height,
            return_accumulated_alphas=False,
        )
        if verbose:
            print("Finished step 5.")
        rendered_image = render_colors[0].permute(2, 0, 1)  # [3, 1080, 1920]
        means2D = means2d
        
        return rendered_image, radii
