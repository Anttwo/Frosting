from typing import Union
import torch.nn as nn
from torch.nn.functional import normalize as torch_normalize
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_apply, quaternion_invert, matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.utils import ico_sphere as create_ico_sphere
from pytorch3d.ops import knn_points
from simple_knn._C import distCUDA2
from frosting_utils.spherical_harmonics import (
    eval_sh, RGB2SH, SH2RGB,
)
from frosting_utils.graphics_utils import *
from frosting_utils.general_utils import inverse_sigmoid
from frosting_scene.gs_model import GaussianSplattingWrapper, GaussianModel
from frosting_scene.cameras import CamerasWrapper
from frosting_scene.sugar_model import SuGaR
from frosting_utils.spherical_harmonics import get_samples_on_sphere
from frosting_utils.mesh_rasterization import RasterizationSettings as OCRasterizationSettings
from frosting_utils.mesh_rasterization import MeshRasterizer as OCMeshRasterizer
import open3d as o3d

use_gsplat_rasterizer = False
if use_gsplat_rasterizer:
    print("Using gsplat rasterizer from Nerfstudio.")
    from gsplat_wrapper.rasterization import GaussianRasterizationSettings, GaussianRasterizer
else:
    print("Using original 3DGS rasterizer from Inria.")
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


scale_activation = torch.exp
scale_inverse_activation = torch.log


def contract(x:torch.Tensor, c:torch.Tensor, L:float):
    """Function to contract a point x towards a point c if the distance between x and c is larger than L.

    Args:
        x (torch.Tensor): Tensor of shape (n_points, 3) representing the points to contract.
        c (torch.Tensor): Tensor of shape (1, 3) or (n_points, 3) representing the points to contract towards.
        L (float): Distance threshold.
    """
    res = x + 0.
    dists = (x - c).norm(dim=-1, keepdim=True)
    mask = dists[..., 0] > L
    res[mask] = c + L * (2 - L / dists[mask]) * ((x - c)[mask] / dists[mask])
    return res


def get_bary_coords_in_triangle(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)
    
    inv_denom = 1 / (d00 * d11 - d01 * d01)
    v = (d11 * d20 - d01 * d21) * inv_denom
    w = (d00 * d21 - d01 * d20) * inv_denom
    u = 1 - v - w
    
    return torch.stack([u, v, w], dim=-1)


def get_points_depth_in_depthmaps(
    points_in_world_space:torch.Tensor, 
    depths:Union[torch.Tensor, list],
    p3d_cameras,
    already_in_camera_space:bool=False,
    return_whole_package=True,
    ):
    """Get the depth of points in the depth map of a camera.

    Args:
        points_in_world_space (torch.Tensor): Has shape (n_points, 3)
        depths (torch.Tensor): Has shape (n_cameras, W, H) or (W, H). WARNING: Works currently only with n_cameras=1.
        p3d_cameras (P3DCameras): _description_
        already_in_camera_space (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # TODO: Not efficient right now, as the current implementation only works for n_cameras = 1.
    # TODO: I should implement a parallelized version that processes k depth maps at once.

    if already_in_camera_space:
        points_in_camera_space = points_in_world_space
    else:
        points_in_camera_space = p3d_cameras.get_world_to_view_transform().transform_points(points_in_world_space)

    image_width = depths.shape[-1]
    image_height = depths.shape[-2]
    
    if len(depths.shape) == 2:  # if depths has shape (W, H)
        depth_view = depths.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)  # Shape (1, 1, W, H)
    elif len(depths.shape) == 3:  # if depths has shape (n_cameras, W, H)
        depth_view = depths.unsqueeze(-1).permute(0, 3, 1, 2)  # Shape (n_cameras, 1, W, H)
    
    n_cameras = depth_view.shape[0]
    pts_projections = p3d_cameras.get_projection_transform().transform_points(points_in_camera_space)

    factor = -1 * min(image_height, image_width)
    # todo: Parallelize these two lines with a tensor [image_width, image_height]
    pts_projections[..., 0] = factor / image_width * pts_projections[..., 0]
    pts_projections[..., 1] = factor / image_height * pts_projections[..., 1]
    pts_projections = pts_projections[..., :2].view(1, -1, 1, 2)
    
    oof_mask = (pts_projections[0, ..., 0, :].min(dim=-1)[0] < -1) + (pts_projections[0, ..., 0, :].max(dim=-1)[0] > 1)
    inside_mask = ~oof_mask

    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=pts_projections,
        mode='bilinear',
        padding_mode='zeros'  # 'reflection', 'zeros'
    )[0, 0, :, 0]
        
    if return_whole_package:
        output_pkg = {
            'map_z': map_z,
            'inside_mask': inside_mask,
            'pts_projections': pts_projections,
            'real_z': points_in_camera_space[..., 2],
        }
        return output_pkg
    else:
        return map_z
    
    
def rasterization_with_pix_to_face(
    mesh, cameras, cam_idx,
    image_height, image_width,
    ):
    p3d_camera = cameras.p3d_cameras[cam_idx]

    # Create a mesh renderer
    occlusion_raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=0.0, 
        faces_per_pixel=1,
        # max_faces_per_bin=max_faces_per_bin
    )
    occlusion_rasterizer = MeshRasterizer(
            cameras=p3d_camera, 
            raster_settings=occlusion_raster_settings,
    )
    
    with torch.no_grad():
        fragments = occlusion_rasterizer(mesh, cameras=p3d_camera)     
        
    return fragments


class Frosting(nn.Module):
    """Main class for Gaussian Frosting models.
    Because Frosting optimization starts with first optimizing a vanilla Gaussian Splatting model for 7k iterations,
    we built this class as a wrapper of a vanilla Gaussian Splatting model.
    Consequently, a corresponding Gaussian Splatting model trained for 7k iterations must be provided.
    However, this wrapper implementation may not be the most optimal one for memory usage, so we might change it in the future.
    """
    def __init__(
        self, 
        nerfmodel: GaussianSplattingWrapper,
        coarse_sugar: SuGaR,
        sh_levels:int=4,
        shell_base_to_bind=None,  # Open3D mesh
        learn_shell:bool=False,
        min_frosting_factor:float=-0.5,  # Deprecated
        max_frosting_factor:float=1.5,  # Deprecated
        n_gaussians_in_frosting:int=2_000_000,
        # Frosting initialization
        n_closest_gaussians_to_use_for_initializing_frosting:int=16,
        n_points_per_pass_for_initializing_frosting:int=2_000_000,
        n_samples_per_vertex_for_initializing_frosting:int=21,  # 21 works. 51?
        frosting_level:float=0.01,  # Should be 0.01
        smooth_initial_frosting:bool=True,
        n_neighbors_for_smoothing_initial_frosting:int=4,
        n_min_gaussian_per_cell:int='auto',
        # Edition
        editable:bool=False,
        # Misc
        use_softmax_for_bary_coords:bool=True,
        min_frosting_range=0.,  # Deprecated
        min_frosting_size:float=0.001,  # 0.001
        # New method
        initial_proposal_std_range:float=3.,  # Should be 3.
        final_proposal_range:float=3.,  # Should be 3.; 1. works too, but 3. is better
        final_clamping_range:float=0.1,  # 0.1
        # Misc
        initialize_model:bool=True,
        use_background_sphere:bool=False,
        use_background_gaussians:bool=True,  # Should be True for real or unbounded scenes, False for synthetic and bounded scenes
        contract_points:bool=True,  # Was False before, but should be True
        avoid_self_intersections:bool=True,  # Was False before, but should be True
        n_iterations_for_avoiding_self_intersections:int=20,
        # Ablation
        use_constant_frosting_size:bool=False,  # Should be False. Used for ablation only.
        constant_frosting_factor:float=1.,  # Used for ablation only.
        avoid_frosting_size_refinement:bool=False,  # Should be False. Used for ablation only.
        project_gaussians_on_base_mesh:bool=False,  # Should be False. Used for ablation only.
        device=None,
        use_simple_adapt=False,
        *args, **kwargs) -> None:
        """
        Args:
            nerfmodel (GaussianSplattingWrapper): A vanilla Gaussian Splatting model trained for 7k iterations.
            sh_levels (int, optional): Number of spherical harmonics levels to use for the color features. Defaults to 4.
            shell_base_to_bind (open3d.geometry.TriangleMesh, optional): The shell base mesh to bind to the frosting. Defaults to None.
            learn_shell (bool, optional): Whether to optimize the shell base. Defaults to True.
            min_frosting_factor (float, optional): Minimum frosting factor. Defaults to -1..
            max_frosting_factor (float, optional): Maximum frosting factor. Defaults to 1..
            n_gaussians_in_frosting (int, optional): Number of gaussians in the frosting. Defaults to 2_000_000.
            n_closest_gaussians_to_use_for_initializing_frosting (int, optional): Number of closest gaussians to use for initializing the frosting. Defaults to 16.
            n_points_per_pass_for_initializing_frosting (int, optional): Number of points per pass for initializing the frosting. Defaults to 2_000_000.
            n_samples_per_vertex_for_initializing_frosting (int, optional): Number of samples per vertex for initializing the frosting. Defaults to 21.
            frosting_level (float, optional): Frosting level. Defaults to 0.01.
            smooth_initial_frosting (bool, optional): Whether to smooth the initial frosting. Defaults to True.
            n_neighbors_for_smoothing_initial_frosting (int, optional): Number of neighbors for smoothing the initial frosting. Defaults to 4.
            editable (bool, optional): Whether the frosting is editable. Defaults to False.
            use_softmax_for_bary_coords (bool, optional): Whether to use softmax for barycentric coordinates. Defaults to True.
            min_frosting_size (float, optional): Minimum frosting size. Defaults to 0.001.
            initial_proposal_std_range (float, optional): Initial proposal standard deviation range. Defaults to 3..
            final_proposal_range (float, optional): Final proposal range. Defaults to 3..
            final_clamping_range (float, optional): Final clamping range. Defaults to 0.1.
        """
        
        super(Frosting, self).__init__()
        
        if device is None:
            if nerfmodel is None:
                raise ValueError("You must provide a device if no nerfmodel is provided.")
            device = nerfmodel.device
            
        if nerfmodel is None:
            scene_spatial_extent = 5.
        else:
            scene_spatial_extent = nerfmodel.training_cameras.get_spatial_extent()
        
        self.occlusion_culling_rasterizer = None
        
        self.nerfmodel = nerfmodel
        
        self.learn_shell = learn_shell
        
        self.frosting_level = frosting_level
        
        self.n_gaussians_in_frosting = n_gaussians_in_frosting
        self.editable = editable
        
        self.use_softmax_for_bary_coords = use_softmax_for_bary_coords
        
        self.project_gaussians_on_base_mesh = project_gaussians_on_base_mesh
        
        # When positions are absolute, skip the barycentric coordinate computation and saves the current positions in memory.
        self.positions_are_absolute = False
        self.absolute_positions = None
        
        if project_gaussians_on_base_mesh:
            print("[WARNING] Projecting gaussians on base mesh is enabled. This is an ablation option.")
            use_constant_frosting_size = True
            constant_frosting_factor = 0.000000001
        
        if (coarse_sugar is None) or (nerfmodel is None):
            print("No coarse sugar model provided. The frosting will not be initialized.")
            initialize_model = False
        
        # ===== Shell base parameters =====
        shell_base_verts = torch.tensor(np.array(shell_base_to_bind.vertices)).float().to(device)
        self._shell_base_verts = torch.nn.Parameter(
            shell_base_verts,
            requires_grad=False).to(device)
        
        self._shell_base_faces = torch.nn.Parameter(
            torch.tensor(np.array(shell_base_to_bind.triangles)).to(device),
            requires_grad=False).to(device)
        
        shell_base = Meshes(
            verts=[shell_base_verts],
            faces=[self._shell_base_faces],
        )
        verts_normals = shell_base.verts_normals_list()[0]
        
        # ===== Initialize shell parameters =====

        # Compute default range from shell base
        if initialize_model:
            print("\nInitializing model...")
            use_last_intersection_as_inner_level_point=True  #TODO: Check if this is the best option

            # We compute the standard deviation of the gaussian at each point
            shell_base_verts = shell_base.verts_list()[0]
            shell_base_verts_normals = 0. + verts_normals

            with torch.no_grad():
                closest_gaussians_idx = knn_points(
                    shell_base_verts[None], 
                    coarse_sugar.points[None], 
                    K=n_neighbors_for_smoothing_initial_frosting
                ).idx[0]
                gaussian_standard_deviations = (coarse_sugar.scaling[closest_gaussians_idx] 
                                                * quaternion_apply(quaternion_invert(coarse_sugar.quaternions[closest_gaussians_idx]), 
                                                                shell_base_verts_normals[:, None])
                                                ).norm(dim=-1)
                points_stds = gaussian_standard_deviations.mean(dim=-1)
                
                # Building a proposal interval for the frosting using a surface model (SuGaR)
                print("\nBuilding a proposal interval for the frosting using the constrained surface representation (SuGaR)...")
                outputs_surface = compute_level_points_along_normals(
                    gaussian_model=coarse_sugar,
                    mesh_verts=shell_base_verts,
                    mesh_verts_normals=shell_base_verts_normals,
                    inner_range=initial_proposal_std_range*points_stds,
                    outer_range=-1.*initial_proposal_std_range*points_stds,
                    n_samples_per_vertex=n_samples_per_vertex_for_initializing_frosting,
                    n_points_per_pass=n_points_per_pass_for_initializing_frosting,
                    n_closest_gaussians_to_use=n_closest_gaussians_to_use_for_initializing_frosting,
                    level=frosting_level,
                    smooth_points=smooth_initial_frosting,
                    n_neighbors_for_smoothing=n_neighbors_for_smoothing_initial_frosting,
                    use_last_intersection_as_inner_level_point=use_last_intersection_as_inner_level_point,
                    min_clamping_inner_dist=None,
                    max_clamping_outer_dist=None,
                    min_layer_size=min_frosting_size,
                )
                
                center_dist = (outputs_surface["outer_dist"] + outputs_surface["inner_dist"]) / 2
                min_outer_dist = center_dist + final_proposal_range * (outputs_surface["outer_dist"] - center_dist)
                max_inner_dist = center_dist + final_proposal_range * (outputs_surface["inner_dist"] - center_dist)
                clamping_outer_dist = center_dist + final_clamping_range * (outputs_surface["outer_dist"] - center_dist)
                clamping_inner_dist = center_dist + final_clamping_range * (outputs_surface["inner_dist"] - center_dist)
                
                # Refining the proposal interval for the frosting using a volumetric model (3DGS)
                if avoid_frosting_size_refinement:
                    outer_dist = outputs_surface["outer_dist"]
                    inner_dist = outputs_surface["inner_dist"]
                    outer_verts = outputs_surface["outer_verts"]
                    inner_verts = outputs_surface["inner_verts"]
                    
                else:
                    print("\nRefining the proposal interval for the frosting using the unconstrained volumetric representation (3DGS)...")
                    outputs_volumetric = compute_level_points_along_normals(
                        gaussian_model=nerfmodel,
                        mesh_verts=shell_base_verts,
                        mesh_verts_normals=shell_base_verts_normals,
                        inner_range=max_inner_dist,
                        outer_range=min_outer_dist,
                        n_samples_per_vertex=n_samples_per_vertex_for_initializing_frosting,
                        n_points_per_pass=n_points_per_pass_for_initializing_frosting,
                        n_closest_gaussians_to_use=n_closest_gaussians_to_use_for_initializing_frosting,
                        level=frosting_level,
                        smooth_points=smooth_initial_frosting,
                        n_neighbors_for_smoothing=n_neighbors_for_smoothing_initial_frosting,
                        use_last_intersection_as_inner_level_point=use_last_intersection_as_inner_level_point,
                        min_clamping_inner_dist=clamping_inner_dist,
                        max_clamping_outer_dist=clamping_outer_dist,
                        min_layer_size=min_frosting_size,
                    )
                    
                    outer_dist = outputs_volumetric["outer_dist"]
                    inner_dist = outputs_volumetric["inner_dist"]
                    outer_verts = outputs_volumetric["outer_verts"]
                    inner_verts = outputs_volumetric["inner_verts"]
                    print("Number of final inner dist larger than proposal:", (inner_dist > outputs_surface["inner_dist"]).sum().item())
                    print("Number of final outer dist smaller than proposal:", (outer_dist < outputs_surface["outer_dist"]).sum().item())
                    
                print("\nNumber of nan in final outer dist:", torch.isnan(outer_dist).sum().item())
                print("Number of nan in final inner dist:", torch.isnan(inner_dist).sum().item())
        else:
            outer_dist = -0.5 * min_frosting_size * scene_spatial_extent * torch.ones(len(shell_base_verts), device=device)
            inner_dist = 0.5 * min_frosting_size * scene_spatial_extent * torch.ones(len(shell_base_verts), device=device)
            inner_verts = shell_base_verts.clone()
            outer_verts = shell_base_verts.clone()
        
        if use_constant_frosting_size:
            print("\n============================================================")
            print("WARNING: Using constant frosting size for ablation purposes.")
            print(f"Using a constant frosting factor of {constant_frosting_factor}.")
            print("============================================================")
            
            median_outer_dist = outer_dist.median().item()
            median_inner_dist = inner_dist.median().item()
            outer_dist = constant_frosting_factor * median_outer_dist * torch.ones(len(shell_base_verts), device=device)
            inner_dist = constant_frosting_factor * median_inner_dist * torch.ones(len(shell_base_verts), device=device)
            outer_verts = shell_base_verts + outer_dist[:, None] * verts_normals
            inner_verts = shell_base_verts + inner_dist[:, None] * verts_normals
            self._constant_frosting_factor = nn.Parameter(
                torch.tensor(constant_frosting_factor, device=device),
                requires_grad=False).to(device)
        
        self._outer_dist = nn.Parameter(outer_dist, requires_grad=learn_shell).to(device)
        self._inner_dist = nn.Parameter(inner_dist, requires_grad=learn_shell).to(device)
        
        if avoid_self_intersections:
            print("\nProcessing the bounds to avoid self-intersections...")
            old_inner_dists, old_outer_dists = self._inner_dist.clone(), self._outer_dist.clone()
            with torch.no_grad():
                new_inner_dist = torch.zeros_like(self._inner_dist)
                new_outer_dist = torch.zeros_like(self._outer_dist)
                self._inner_dist[...] = new_inner_dist
                self._outer_dist[...] = new_outer_dist

                inner_update_mask = torch.ones_like(self._inner_dist, dtype=torch.bool)
                outer_update_mask = torch.ones_like(self._outer_dist, dtype=torch.bool)

                for i_iter in range(n_iterations_for_avoiding_self_intersections):
                    print(f"\n===Iteration {i_iter}===")
                    inner_dist_i = self._inner_dist.clone()
                    outer_dist_i = self._outer_dist.clone()
                    
                    new_inner_dist = inner_dist_i[inner_update_mask] + 1 / n_iterations_for_avoiding_self_intersections * old_inner_dists[inner_update_mask]
                    new_outer_dist = outer_dist_i[outer_update_mask] + 1 / n_iterations_for_avoiding_self_intersections * old_outer_dists[outer_update_mask]
                    print("new_inner_dist shape:", new_inner_dist.shape)
                    print("new_outer_dist shape:", new_outer_dist.shape)

                    self._inner_dist[inner_update_mask] = new_inner_dist.clone()
                    self._outer_dist[outer_update_mask] = new_outer_dist.clone()
                    
                    inner_verts_inside = self.is_inside_frosting(self.inner_verts[inner_update_mask].clone(), k_neighbors_to_use=8, proj_th=1e-6)
                    outer_verts_inside = self.is_inside_frosting(self.outer_verts[outer_update_mask].clone(), k_neighbors_to_use=8, proj_th=1e-6)
                    print("inner_verts_inside shape:", inner_verts_inside.shape)
                    print("outer_verts_inside shape:", outer_verts_inside.shape)
                    
                    new_inner_dist[inner_verts_inside] = inner_dist_i[inner_update_mask][inner_verts_inside]
                    new_outer_dist[outer_verts_inside] = outer_dist_i[outer_update_mask][outer_verts_inside]
                    
                    self._inner_dist[inner_update_mask] = new_inner_dist.clone()
                    self._outer_dist[outer_update_mask] = new_outer_dist.clone()
                    
                    inner_update_mask[inner_update_mask.clone()] = ~inner_verts_inside
                    outer_update_mask[outer_update_mask.clone()] = ~outer_verts_inside
            
        # ===== Sample Gaussians for frosting =====
        
        # We first approximate the volume of each cell by a regular triangular prism.
        # The base is the triangle of the shell base, and the height is the average distance between the inner and outer vertices. 
        shell_base_faces_verts = shell_base_verts[self._shell_base_faces]  # n_faces, 3, 3
        
        if contract_points:
            print("Contracting prismatic cell volumes...")
            camera_center = nerfmodel.training_cameras.p3d_cameras.get_camera_center().mean(dim=0, keepdim=True)
            camera_bbox_half_diag = scene_spatial_extent 
            list_shell_base_faces_verts = shell_base_faces_verts.view(-1, 3)
            list_shell_base_faces_verts = contract(
                x=list_shell_base_faces_verts, 
                c=camera_center, 
                L=camera_bbox_half_diag)
            shell_base_faces_verts = list_shell_base_faces_verts.view(-1, 3, 3)
        
        # Area of base triangles with Heron's formula
        a = (shell_base_faces_verts[:, 0] - shell_base_faces_verts[:, 1]).norm(dim=-1)
        b = (shell_base_faces_verts[:, 1] - shell_base_faces_verts[:, 2]).norm(dim=-1)
        c = (shell_base_faces_verts[:, 2] - shell_base_faces_verts[:, 0]).norm(dim=-1)
        s = (a + b + c) / 2
        base_areas = (s * (s - a) * (s - b) * (s - c)).clamp_min(0.).sqrt()
        
        # Volume of the triangular prisms
        base_heights = (inner_dist - outer_dist).abs()  # n_verts
        base_heights.clamp_min_(min_frosting_size * scene_spatial_extent)
        if contract_points:
            dists = (shell_base_verts - camera_center).norm(dim=-1)  # n_verts
            mask = dists > camera_bbox_half_diag  # n_verts
            base_heights[mask] = base_heights[mask] * (camera_bbox_half_diag / dists[mask])**2  # n_verts
            base_heights = base_heights[self._shell_base_faces].mean(dim=-1)  # n_faces
        else:
            base_heights = base_heights[self._shell_base_faces].mean(dim=-1)  # n_faces
                        
        cell_volumes = base_areas * base_heights

        # We first sample at least n_min_gaussian_per_cell points per cell. 
        # We update the number of gaussians if needed.
        if n_min_gaussian_per_cell == 'auto':
            n_min_gaussian_per_cell = int(np.ceil(n_gaussians_in_frosting / (2*len(cell_volumes))))
            print(f"Using n_min_gaussian_per_cell={n_min_gaussian_per_cell} (computed automatically).")

        if (len(cell_volumes) * n_min_gaussian_per_cell) > n_gaussians_in_frosting:
            print(f"Warning: The number of cells ({len(cell_volumes)}) x n_min_gaussian_per_cell ({n_min_gaussian_per_cell}) is larger than the number of gaussians in the frosting ({n_gaussians_in_frosting}).")
            print(f"Updating the number of gaussians in the frosting to {len(cell_volumes) * n_min_gaussian_per_cell}.")
            n_gaussians_in_frosting = len(cell_volumes) * n_min_gaussian_per_cell
            self.n_gaussians_in_frosting = n_gaussians_in_frosting
        point_cell_indices = torch.arange(len(cell_volumes), device=device).repeat(n_min_gaussian_per_cell)
        
        # And we sample more points in cells with larger volumes
        if n_gaussians_in_frosting > (len(cell_volumes) * n_min_gaussian_per_cell):
            random_indices = torch.multinomial(
                cell_volumes / cell_volumes.sum(), 
                num_samples=max(0, n_gaussians_in_frosting - (len(cell_volumes) * n_min_gaussian_per_cell)),
                replacement=True
                ).to(device)
            point_cell_indices = torch.cat([point_cell_indices, random_indices], dim=0)
        self._point_cell_indices = torch.nn.Parameter(point_cell_indices, requires_grad=False).to(device)


        # ===== Initialize Gaussians for frosting =====
        shell_verts = torch.cat([inner_verts[:, None], outer_verts[:, None]], dim=1)
        shell_cells_verts = shell_verts[self._shell_base_faces].transpose(-2, -3)  # n_faces, 2, 3, 3

        # We sample random barycentric coordinates in each cell, that sum to 1
        bary_coords = torch.rand(n_gaussians_in_frosting, 6, device=device)  # (n_gaussians_in_final_scene, 6)
        bary_coords[..., -1] = 1.
        bary_coords = bary_coords.sort(dim=-1)[0]
        bary_coords[..., 1:] = bary_coords[..., 1:] - bary_coords[..., :-1]  # (n_gaussians_in_final_scene, 6)
        bary_coords = bary_coords  # (n_gaussians_in_final_scene, 6)
        if self.use_softmax_for_bary_coords:
            self._bary_coords = torch.nn.Parameter(torch.log(bary_coords.clamp_min(1e-8)), requires_grad=True).to(device)
        else:
            self._bary_coords = torch.nn.Parameter(bary_coords, requires_grad=True).to(device)

        # Compute points from barycentric coordinates
        points = (bary_coords[..., None] * shell_cells_verts[point_cell_indices].reshape(-1, 6, 3)).sum(dim=-2)  # (n_gaussians_in_final_scene, 3)
        self.n_points = len(points)
        
        with torch.no_grad():
            if initialize_model:
                # Compute closest unconstrained gaussians to each point
                closest_unconstrained_idx = knn_points(points[None], nerfmodel.gaussians.get_xyz[None], K=1).idx[0, ..., 0]
        
        # Initialize scales. Scales should not be larger than the smallest side of the cell
        self.scale_activation = scale_activation
        self.scale_inverse_activation = scale_inverse_activation
        shell_side_lengths = (shell_cells_verts[:, 1] - shell_cells_verts[:, 0]).norm(dim=-1) 
        shell_base_lengths = (shell_cells_verts - shell_cells_verts[:, :, [1, 2, 0]]).norm(dim=-1)
        shell_cell_lengths = torch.cat([shell_side_lengths, shell_base_lengths.view(-1, 6)], dim=1)
        max_scales = shell_cell_lengths.max(dim=-1)[0][point_cell_indices]  # TODO: Should we add * 0.5?
        if initialize_model:
            dist2 = torch.clamp_min(distCUDA2(points).clamp_max(max_scales**2), 0.0000001)
            scales = dist2.sqrt()
        else:
            scales = 0.5 * max_scales
        scales = scale_inverse_activation(scales)[...,None].repeat(1, 3)
        self._scales = torch.nn.Parameter(scales, requires_grad=True).to(device)  # (n_gaussians_in_final_scene, 3)

        # Initialize rotations
        quaternions = torch.zeros((points.shape[0], 4), device="cuda")
        quaternions[:, 0] = 1
        self._quaternions = torch.nn.Parameter(quaternions, requires_grad=True).to(device)  # (n_gaussians_in_final_scene, 4)

        # Initialize opacities and spherical harmonics from unconstrained Gaussians
        with torch.no_grad():
            opacities = inverse_sigmoid(0.1 * torch.ones((points.shape[0], 1), dtype=torch.float, device=device))
            if initialize_model:
                sh_coordinates_dc = nerfmodel.gaussians._features_dc[closest_unconstrained_idx].clone()
                sh_coordinates_rest = nerfmodel.gaussians._features_rest[closest_unconstrained_idx].clone()
            else:
                sh_coordinates_dc = RGB2SH(torch.zeros(points.shape[0], 1, 3, dtype=torch.float, device=device))
                sh_coordinates_rest = torch.zeros(points.shape[0], sh_levels**2-1, 3, dtype=torch.float, device=device)
            self.sh_levels = sh_levels
        self._opacities = torch.nn.Parameter(opacities, requires_grad=True).to(device)  # (n_gaussians_in_final_scene, 1)
        self._sh_coordinates_dc = torch.nn.Parameter(sh_coordinates_dc, requires_grad=True).to(device)  # (n_gaussians_in_final_scene, 1, 3)
        self._sh_coordinates_rest = torch.nn.Parameter(sh_coordinates_rest, requires_grad=True).to(device)  # (n_gaussians_in_final_scene, sh_levels**2-1, 3)
        
        # ===== Initialize background sphere =====
        self.use_background_sphere = use_background_sphere
        if self.use_background_sphere:
            raise NotImplementedError(
                "Background sphere feature has been removed, as background gaussians are more efficient. \
                Please use use_background_gaussians=True instead."
            )
        
        self.use_background_gaussians = use_background_gaussians
        if self.use_background_gaussians:
            with torch.no_grad():
                # Compute foreground bbox
                fg_bbox_min_tensor = shell_base_verts.min(dim=0, keepdim=True)[0]
                fg_bbox_max_tensor = shell_base_verts.max(dim=0, keepdim=True)[0]
                fg_mask = (nerfmodel.gaussians.get_xyz > fg_bbox_min_tensor).all(dim=-1) * (nerfmodel.gaussians.get_xyz < fg_bbox_max_tensor).all(dim=-1)
                
                # Compute background Gaussians as the Gaussians outside the foreground bbox
                bg_mask = ~fg_mask
                _bg_points = nerfmodel.gaussians._xyz[bg_mask].detach()
                _bg_opacities = nerfmodel.gaussians._opacity[bg_mask].detach()
                _bg_sh_coordinates_dc = nerfmodel.gaussians._features_dc[bg_mask].detach()
                _bg_sh_coordinates_rest = nerfmodel.gaussians._features_rest[bg_mask].detach()
                _bg_scales = nerfmodel.gaussians._scaling[bg_mask].detach()
                _bg_quaternions = nerfmodel.gaussians._rotation[bg_mask].detach()
            if len(_bg_points) == 0:
                print("No background gaussians found. Disabling background gaussians.")
                self.use_background_gaussians = False
            else:
                print(f"Using {len(_bg_points)} background gaussians.")
                self.n_points = len(_bg_points) + self.n_points
                self._bg_points = torch.nn.Parameter(_bg_points, requires_grad=True).to(device)
                self._bg_opacities = torch.nn.Parameter(_bg_opacities, requires_grad=True).to(device)
                self._bg_sh_coordinates_dc = torch.nn.Parameter(_bg_sh_coordinates_dc, requires_grad=True).to(device)
                self._bg_sh_coordinates_rest = torch.nn.Parameter(_bg_sh_coordinates_rest, requires_grad=True).to(device)
                self._bg_scales = torch.nn.Parameter(_bg_scales, requires_grad=True).to(device)
                self._bg_quaternions = torch.nn.Parameter(_bg_quaternions, requires_grad=True).to(device)
           
        # ===== Render parameters =====
        if nerfmodel is not None:
            self.image_height = int(nerfmodel.training_cameras.height[0].item())
            self.image_width = int(nerfmodel.training_cameras.width[0].item())
            self.focal_factor = max(nerfmodel.training_cameras.p3d_cameras.K[0, 0, 0].item(),
                                    nerfmodel.training_cameras.p3d_cameras.K[0, 1, 1].item())
            
            self.fx = nerfmodel.training_cameras.fx[0].item()
            self.fy = nerfmodel.training_cameras.fy[0].item()
            self.fov_x = focal2fov(self.fx, self.image_width)
            self.fov_y = focal2fov(self.fy, self.image_height)
            self.tanfovx = math.tan(self.fov_x * 0.5)
            self.tanfovy = math.tan(self.fov_y * 0.5)
        
        # Reference scaling factor
        self.use_simple_adapt = use_simple_adapt
        if self.editable:
            self.make_editable(use_simple_adapt=use_simple_adapt)
            
    @property
    def device(self):
        return self._shell_base_verts.device
        
    @property
    def shell_base(self):
        return Meshes(
            verts=[self._shell_base_verts],
            faces=[self._shell_base_faces],
        )
        
    @property
    def shell_outer(self):
        return Meshes(
            verts=[self.outer_verts],
            faces=[self._shell_base_faces],
        )
        
    @property
    def shell_inner(self):
        return Meshes(
            verts=[self.inner_verts],
            faces=[self._shell_base_faces],
        )
        
    @property
    def shell_base_normals(self):
        return self.shell_base.verts_normals_list()[0]
    
    @property
    def shell_outer_normals(self):
        return self.shell_outer.verts_normals_list()[0]
    
    @property
    def shell_inner_normals(self):
        return self.shell_inner.verts_normals_list()[0]
        
    @property
    def outer_verts(self):
        if self.editable:
            current_faces_verts = self._shell_base_verts[self._shell_base_faces]  # n_faces, 3, 3
            
            if self._thickness_rescaling_method == "median":
                current_distance = (current_faces_verts - current_faces_verts.mean(dim=-2, keepdim=True))[self._edition_mask].norm(dim=-1).median().item()
                distance_factor = torch.ones(len(self._shell_base_verts), 1, device=self.device)
                distance_factor[self._verts_edition_mask] = current_distance / self._reference_distance
                
            elif self._thickness_rescaling_method == "triangle":
                current_distance = (current_faces_verts - current_faces_verts.mean(dim=-2, keepdim=True))[self._edition_mask].norm(dim=-1).median(dim=-1).values
                face_distance_factor = torch.ones_like(self._edition_mask, dtype=torch.float32)
                face_distance_factor[self._edition_mask] = (current_distance / self._reference_distance).nan_to_num()
                face_distance_factor = face_distance_factor.unsqueeze(-1)  # n_faces, 1
                face_distance_factor = face_distance_factor.repeat(1, 3).flatten()  # n_faces * 3

                distance_factor = torch.zeros(len(self._shell_base_verts), device=self.device)  # n_verts
                distance_factor.index_add_(dim=0, index=self._shell_base_faces.flatten(), source=face_distance_factor)
                
                counts = torch.zeros(len(self._shell_base_verts), device=self.device)  # n_verts
                counts.index_add_(dim=0, index=self._shell_base_faces.flatten(), source=torch.ones_like(face_distance_factor))
                
                distance_factor = (distance_factor / counts.clamp(min=1)).unsqueeze(-1)  # n_verts, 1
                
            return self._shell_base_verts + distance_factor * self._outer_dist[:, None] * self.shell_base_normals  # (n_verts, 3)
        else:
            return self._shell_base_verts + self._outer_dist[:, None] * self.shell_base_normals  # (n_verts, 3)

    @property
    def inner_verts(self):
        if self.editable:
            current_faces_verts = self._shell_base_verts[self._shell_base_faces]  # n_faces, 3, 3
            
            if self._thickness_rescaling_method == "median":
                current_distance = (current_faces_verts - current_faces_verts.mean(dim=-2, keepdim=True))[self._edition_mask].norm(dim=-1).median().item()
                distance_factor = torch.ones(len(self._shell_base_verts), 1, device=self.device)
                distance_factor[self._verts_edition_mask] = current_distance / self._reference_distance
                
            elif self._thickness_rescaling_method == "triangle":
                current_distance = (current_faces_verts - current_faces_verts.mean(dim=-2, keepdim=True))[self._edition_mask].norm(dim=-1).median(dim=-1).values
                face_distance_factor = torch.ones_like(self._edition_mask, dtype=torch.float32)
                face_distance_factor[self._edition_mask] = (current_distance / self._reference_distance).nan_to_num()
                face_distance_factor = face_distance_factor.unsqueeze(-1)  # n_faces, 1
                face_distance_factor = face_distance_factor.repeat(1, 3).flatten()  # n_faces * 3

                distance_factor = torch.zeros(len(self._shell_base_verts), device=self.device)  # n_verts
                distance_factor.index_add_(dim=0, index=self._shell_base_faces.flatten(), source=face_distance_factor)
                
                counts = torch.zeros(len(self._shell_base_verts), device=self.device)  # n_verts
                counts.index_add_(dim=0, index=self._shell_base_faces.flatten(), source=torch.ones_like(face_distance_factor))
                
                distance_factor = (distance_factor / counts.clamp(min=1)).unsqueeze(-1)  # n_verts, 1
                
            return self._shell_base_verts + distance_factor * self._inner_dist[:, None] * self.shell_base_normals  # (n_verts, 3)
        else:
            return self._shell_base_verts + self._inner_dist[:, None] * self.shell_base_normals  # (n_verts, 3)
    
    @property
    def shell_cells_verts(self):    
        shell_verts = torch.cat([self.inner_verts[:, None], self.outer_verts[:, None]], dim=1)  # n_verts, 2, 3
        return shell_verts[self._shell_base_faces].transpose(-2, -3)  # n_faces, 2, 3, 3
    
    @property
    def bary_coords(self):
        if self.use_softmax_for_bary_coords:
            bary_coords = torch.softmax(self._bary_coords, dim=-1)
        else:
            bary_coords = torch.relu(self._bary_coords)
            bary_coords = bary_coords / bary_coords.sum(dim=-1, keepdim=True)
        return bary_coords  # (n_gaussians_in_final_scene, 6)
    
    @property
    def points(self):
        if self.positions_are_absolute:
            return self.absolute_positions
        else:
            return (self.bary_coords[..., None] * self.shell_cells_verts[self._point_cell_indices].reshape(-1, 6, 3)).sum(dim=-2)  # (n_gaussians_in_final_scene, 3)
    
    @property
    def strengths(self):
        return torch.sigmoid(self._opacities.view(-1, 1))
    
    @property
    def sh_coordinates(self):
        return torch.cat([self._sh_coordinates_dc, self._sh_coordinates_rest], dim=1)
    
    @property
    def scaling(self):
        if self.editable:
            if self.use_simple_adapt:
                scales = self.scale_activation(self._scales)
                
                # Edited Gaussians
                current_faces_verts = self._shell_base_verts[self._shell_base_faces]  # n_edited_faces, 3, 3
                
                if self._thickness_rescaling_method == "median":
                    current_distance = (current_faces_verts - current_faces_verts.mean(dim=-2, keepdim=True))[self._edition_mask].norm(dim=-1).median().item()
                    distance_factor = current_distance / self._reference_distance
                
                elif self._thickness_rescaling_method == "triangle":
                    current_distance = (current_faces_verts - current_faces_verts.mean(dim=-2, keepdim=True))[self._edition_mask].norm(dim=-1).median(dim=-1).values
                    distance_factor = torch.ones_like(self._edition_mask, dtype=torch.float32)  # n_faces
                    distance_factor[self._edition_mask] = (current_distance / self._reference_distance).nan_to_num()  # n_edited_faces
                    distance_factor = distance_factor.unsqueeze(-1)  # n_faces, 1
                    distance_factor = distance_factor[self._point_cell_indices][self._gaussian_edition_mask]  # n_edited_gaussians, 1
                
                scales[self._gaussian_edition_mask] *= distance_factor
            else:
                if (self.edited_cache is not None) and self.edited_cache.shape[-1]==3:
                    scales = self.edited_cache
                    self.edited_cache = None
                else:
                    quaternions, scales = self.get_edited_quaternions_and_scales()
                    self.edited_cache = quaternions
        else:
            scales = self.scale_activation(self._scales)
        return scales
    
    @property
    def quaternions(self):
        if self.editable:
            if self.use_simple_adapt:
                # Inputs: We have the following transformations:
                # > quaternions (q) sends points from the canonical space `can` to the Gaussians reference world space `w`: q:can->w.
                # > reference_quaternions (g) sends points from `can` to the triangle reference space `tri`: g:can->tri.
                # > face_quaternions (g') sends points from `can` to the triangle modified space `tri'`: g':can->tri'.
                #
                # Output: We are looking for modified quaternions (q'), which send points from `can` to the Gaussians modified space `w'`: q':can->w'.
                #
                # We denote by h:tri->w the transformation from the triangle space to the world space. 
                # This transformation h does not depend on the scene editing, so h' = h for h':tri'->w'.
                # We have q' = h' o g' = h o g' = q o g^-1 o g' where "o" denotes the composition of transformations.
                # Consequently, the final quaternions can be computed as the following product of quaternions:
                # new_quaternions = face_quaternions x reference_quaternions^-1 x quaternions
                reference_quaternions = self._reference_quaternions[self._point_cell_indices[self._gaussian_edition_mask]]  # n_gaussians_in_final_scene, 4
                face_quaternions = self.compute_face_quaternions()[self._point_cell_indices[self._gaussian_edition_mask]]  # n_gaussians_in_final_scene, 4
                
                quaternions = torch.nn.functional.normalize(self._quaternions, dim=-1)  # n_gaussians_in_final_scene, 4
                tri2w_quaternions = quaternion_multiply(quaternion_invert(reference_quaternions), quaternions[self._gaussian_edition_mask])
                quaternions[self._gaussian_edition_mask] = quaternion_multiply(face_quaternions, tri2w_quaternions)
            else:
                if (self.edited_cache is not None) and self.edited_cache.shape[-1]==4:
                    quaternions = self.edited_cache
                    self.edited_cache = None
                else:
                    quaternions, scales = self.get_edited_quaternions_and_scales()
                    self.edited_cache = scales
        else:
            quaternions = torch.nn.functional.normalize(self._quaternions, dim=-1)
        return quaternions
    
    @property
    def bg_points(self):
        bg_points = torch.zeros(0, 3, dtype=torch.float, device=self.device)
        if self.use_background_gaussians:
            bg_points = torch.cat([bg_points, self._bg_points], dim=0)
        if self.use_background_sphere:
            bg_points = torch.cat([bg_points, self._bg_sphere_points], dim=0)
        return bg_points
    
    @property
    def bg_strengths(self):
        bg_strengths = torch.zeros(0, 1, dtype=torch.float, device=self.device)
        if self.use_background_gaussians:
            bg_strengths = torch.cat([bg_strengths, self._bg_opacities], dim=0)
        if self.use_background_sphere:
            bg_strengths = torch.cat([bg_strengths, self._bg_sphere_opacities], dim=0)
        return torch.sigmoid(bg_strengths)
    
    @property
    def bg_sh_coordinates(self):
        bg_sh_coordinates = torch.zeros(0, self.sh_levels**2, 3, dtype=torch.float, device=self.device)
        if self.use_background_gaussians:
            bg_sh_coordinates = torch.cat(
                [bg_sh_coordinates, 
                 torch.cat([self._bg_sh_coordinates_dc, 
                            self._bg_sh_coordinates_rest], 
                           dim=1)], 
                dim=0)
        if self.use_background_sphere:
            bg_sh_coordinates = torch.cat(
                [bg_sh_coordinates, 
                 torch.cat([self._bg_sphere_sh_coordinates_dc, 
                            self._bg_sphere_sh_coordinates_rest], 
                           dim=1)], 
                dim=0)
        return bg_sh_coordinates
    
    @property
    def bg_scaling(self):
        bg_scaling = torch.zeros(0, 3, dtype=torch.float, device=self.device)
        if self.use_background_gaussians:
            bg_scaling = torch.cat([bg_scaling, self.scale_activation(self._bg_scales)], dim=0)
        if self.use_background_sphere:
            bg_sphere_scaling = torch.cat(
                [self.bg_sphere_thickness * torch.ones(len(self._bg_sphere_scales), 1, dtype=torch.float, device=self.device),
                self.scale_activation(self._bg_sphere_scales),],
                dim=1)
            bg_scaling = torch.cat([bg_scaling, bg_sphere_scaling], dim=0)
        return bg_scaling
    
    @property
    def bg_quaternions(self):
        bg_quaternions = torch.zeros(0, 4, dtype=torch.float, device=self.device)
        if self.use_background_gaussians:
            bg_quaternions = torch.cat([bg_quaternions, torch.nn.functional.normalize(self._bg_quaternions, dim=-1)], dim=0)
        if self.use_background_sphere:
            # We compute the base rotation matrix
            base_matrix = quaternion_to_matrix(self._bg_sphere_base_quaternions)
            
            # We adjust the rotation using the learned 2D rotation
            complex_numbers = torch_normalize(self._bg_sphere_complex, dim=-1)
            R_1 = complex_numbers[..., 0:1] * base_matrix[..., 1] + complex_numbers[..., 1:2] * base_matrix[..., 2]
            R_2 = -complex_numbers[..., 1:2] * base_matrix[..., 1] + complex_numbers[..., 0:1] * base_matrix[..., 2]

            # We concatenate the three vectors to get the rotation matrix
            R = torch.cat(
                [
                    base_matrix[..., 0:1].clone(),
                    R_1[..., None],
                    R_2[..., None]
                ],
                dim=-1
            ).view(-1, 3, 3)
            bg_sphere_quaternions = torch_normalize(matrix_to_quaternion(R), dim=-1)
            bg_quaternions = torch.cat([bg_quaternions, bg_sphere_quaternions], dim=0)
        return bg_quaternions
    
    def make_positions_absolute(self):
        self.absolute_positions = self.points
        self.positions_are_absolute = True
    
    def make_positions_relative(self):
        self.absolute_positions = None
        self.positions_are_absolute = False
    
    # WARNING: The normal of the triangle is set as the first axis of the rotation matrix
    def compute_face_quaternions(self):
        faces_verts = self._shell_base_verts[self._shell_base_faces]  # n_faces, 3, 3
        R_0 = torch.nn.functional.normalize(self.shell_base.faces_normals_list()[0], dim=-1)  # n_faces, 3
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)  # n_faces, 3
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))  # n_faces, 3
        face_quaternions = torch.cat([
            R_0[..., None].clone(),
            base_R_1[..., None].clone(),
            base_R_2[..., None].clone()
        ], dim=-1)
        face_quaternions = torch_normalize(matrix_to_quaternion(face_quaternions), dim=-1)  # n_faces, 4
        return face_quaternions
    
    def compute_cell_transformations(self, orthogonalize=True, faces_mask=None):
        """Compute the transformations from a canonical triangle/cell space to the world space.

        Args:
            orthogonalize (bool, optional): If True, orthogonalize the resulting base. Defaults to True.

        Returns:
            torch.Tensor: The transformations matrices from the canonical triangle/cell space to the world space. Shape: (n_faces, 3, 3)
        """
        axis_bary_shifts = torch.tensor([[
            [-np.sqrt(2)/2, np.sqrt(2)/2, 0.],
            [-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
        ]], device=self.device, dtype=torch.float32)  # (1, 2, 3)
        
        faces_verts = self._shell_base_verts[self._shell_base_faces]  # (n_faces, 3, 3)
        if faces_mask is not None:
            faces_verts = faces_verts[faces_mask]
        
        if self._thickness_rescaling_method == "median":
            distance_factor = (faces_verts - faces_verts.mean(dim=-2, keepdim=True)).norm(dim=-1).median().item()
            
        elif self._thickness_rescaling_method == "triangle":
            distance_factor = torch.ones(len(self._shell_base_faces), dtype=torch.float32, device=self.device)  # n_faces
            if faces_mask is not None:
                distance_factor[faces_mask] = (faces_verts - faces_verts.mean(dim=-2, keepdim=True)).norm(dim=-1).median(dim=-1).values  # n_faces_in_mask
            else:
                distance_factor = (faces_verts - faces_verts.mean(dim=-2, keepdim=True)).norm(dim=-1).median(dim=-1).values  # n_faces
            distance_factor = distance_factor.unsqueeze(-1)  # n_faces, 1
        
        # If the distance factor is zero, we set it to 1 as no scaling should be applied
        if isinstance(distance_factor, torch.Tensor):
            distance_factor[distance_factor == 0.] = 1.
        transformed_normal = torch_normalize(self.shell_base.faces_normals_packed(), dim=-1) * distance_factor  # n_faces, 3
        if faces_mask is not None:
            transformed_normal = transformed_normal[faces_mask]
        transformed_axis = (axis_bary_shifts[..., None] * faces_verts[:, None]).sum(dim=-2)  # n_faces, 2, 3
        
        if orthogonalize:
            use_biggest_axis_as_first_axis_for_gram_schmidt = True
            if use_biggest_axis_as_first_axis_for_gram_schmidt:
                # TODO: Is torch.gather too slow?
                sorted_transformed_idx = torch.argsort(transformed_axis.norm(dim=-1, keepdim=True), dim=1, descending=True).repeat(1, 1, transformed_axis.shape[-1])
                transformed_axis = transformed_axis.gather(dim=1, index=sorted_transformed_idx).contiguous()
                second_ortho_axis = transformed_axis[:, 1] - (
                    (transformed_axis[:, 1] * transformed_axis[:, 0]).sum(dim=-1, keepdim=True) 
                    * transformed_axis[:, 0] / (transformed_axis[:, 0] ** 2).sum(dim=-1, keepdim=True)
                )
                transformed_axis = torch.cat(
                    [transformed_axis[:, 0:1], second_ortho_axis[:, None]], 
                    dim=1
                ).gather(dim=1, index=sorted_transformed_idx)
            else:
                transformed_axis[:, 1] = transformed_axis[:, 1] - (
                    (transformed_axis[:, 1] * transformed_axis[:, 0]).sum(dim=-1, keepdim=True) 
                    * transformed_axis[:, 0] / (transformed_axis[:, 0] ** 2).sum(dim=-1, keepdim=True)
                )
                
        transformed_axis = torch.cat([
            transformed_axis,  # n_faces, 2, 3, 
            transformed_normal[:, None]  # n_faces, 1, 3
            ], dim=-2)  # n_faces, 3, 3
        
        if faces_mask is not None:
            transformations = torch.empty((len(self._shell_base_faces), 3, 3), device=self.device)
            transformations[faces_mask] = transformed_axis.transpose(-1, -2)
        else:
            transformations = transformed_axis.transpose(-1, -2)  # n_faces, 3, 3
        return transformations
    
    def project_gaussians_on_mesh(self, thickness=None, change_values=False):
        if thickness is None:
            thickness = self.nerfmodel.training_cameras.get_spatial_extent() / 1_000_000

        # Change positions by changing the inner and outer distances
        self._inner_dist[...] = thickness / 100.
        self._outer_dist[...] = -thickness / 100.
        
        normals = torch.nn.functional.normalize(self.get_points_normals())  # n_gaussians_in_final_scene, 3
        rotation_matrices = quaternion_to_matrix(self.quaternions)[..., :3]  # n_gaussians_in_final_scene, 3, n_vec
        scaled_rotation_matrics = rotation_matrices * self.scaling[:, None]
        scalar_prod = (scaled_rotation_matrics * normals[..., None]).sum(dim=-2, keepdim=True)  # n_gaussians_in_final_scene, 1, n_vec
        proj_rot = scaled_rotation_matrics - scalar_prod * normals[..., None]  # n_gaussians_in_final_scene, 3, n_vec

        proj_norms = proj_rot.norm(dim=-2)  # n_gaussians_in_final_scene, n_vec
        sorted_proj_norms, sorted_proj_indices = proj_norms.sort(dim=-1, descending=True)  # n_gaussians_in_final_scene, n_vec
        sorted_proj_rots = proj_rot.gather(dim=-1, index=sorted_proj_indices[:, None, :3].repeat(1, 3, 1))  # n_gaussians_in_final_scene, 3, n_vec
        
        s_0 = torch.ones_like(sorted_proj_norms[:, 0]) * thickness
        R_0 = normals
        
        s_1 = sorted_proj_norms[:, 0]
        R_1 = torch.nn.functional.normalize(sorted_proj_rots[:, :, 0])
        
        R_2 = torch.cross(normals, R_1, dim=-1)
        s_2 = (sorted_proj_rots[:, :, 1] * R_2).sum(dim=-1).abs()
        
        new_quaternions = torch.cat([
            R_0[..., None].clone(),
            R_1[..., None].clone(),
            R_2[..., None].clone()
        ], dim=-1)
        new_quaternions = torch_normalize(matrix_to_quaternion(new_quaternions), dim=-1)  # n_gaussians_in_final_scene, 4
        new_scales = torch.stack([s_0, s_1, s_2], dim=-1)  # n_gaussians_in_final_scene, 3
        new_scales = new_scales
        
        if change_values:
            with torch.no_grad():
                self._scales[...] = scale_inverse_activation(new_scales)
                self._quaternions[...] = new_quaternions
        
        return new_scales, new_quaternions
    
    @torch.no_grad()
    def make_editable(
        self, 
        use_simple_adapt:bool=False, 
        edition_mask:torch.Tensor=None, 
        thickness_rescaling_method='median',
    ):
        """_summary_

        Args:
            use_simple_adapt (bool, optional): Defaults to False.
            edition_mask (torch.Tensor, optional): Defaults to None.
            thickness_rescaling_method (str, optional): Describes how the thickness of the frosting is rescaled when editing the mesh.
                Can be 'median' or 'triangle':
                > If 'triangle', the thickness is adapted for each triangle of the mesh, depending on how the triangle is modified.
                > If 'median', the thickness is adapted globally, depending on the median rescaling of all triangles. 
        """
        if self.editable:
            print("The model is already editable.")
        else:
            self.use_simple_adapt = use_simple_adapt
            
            if edition_mask is None:
                edition_mask = torch.ones(len(self._shell_base_faces), dtype=torch.bool, device=self.device)
            self._edition_mask = edition_mask
            self._verts_edition_mask = self._shell_base_faces[self._edition_mask].unique()
            self._gaussian_edition_mask = self._edition_mask[self._point_cell_indices]
            self._thickness_rescaling_method = thickness_rescaling_method
            
            faces_verts = self._shell_base_verts[self._shell_base_faces]  # n_faces, 3, 3
            self._reference_verts = self._shell_base_verts.clone().detach()
            
            if thickness_rescaling_method == 'median':
                self._reference_distance = (faces_verts - faces_verts.mean(dim=-2, keepdim=True))[edition_mask].norm(dim=-1).median().item()
            elif thickness_rescaling_method == 'triangle':
                self._reference_distance = (faces_verts - faces_verts.mean(dim=-2, keepdim=True))[edition_mask].norm(dim=-1).median(dim=-1).values
            else:
                raise ValueError(f"Thickness rescaling method '{thickness_rescaling_method}' is not recognized.")
            
            if self.use_simple_adapt:
                print(f"Making the model editable with the simple adaptation method and {thickness_rescaling_method} rescaling.")
                print("The simple adaptation method is faster but can be less accurate.")
                self._reference_quaternions = self.compute_face_quaternions()
            else:
                self.edited_cache = None
                print(f"Making the model editable with the complex adaptation method and {thickness_rescaling_method} rescaling.")
                print("The complex adaptation method is slower but more accurate.")
                # Transformations from canonical triangle space to world space:
                # These transformations are the ones that change when editing the scene.
                tri_transformations = self.compute_cell_transformations(faces_mask=self._edition_mask)  # n_faces, 3, 3

                # Traditional transformations from the "Gaussian input space" to world space:
                # We want to make these transformations invariant to the editing of the scene.    
                gaussian_transformations = quaternion_to_matrix(self.quaternions[self._gaussian_edition_mask])  # n_gaussians_in_final_scene, 3, 3
                gaussian_transformations = gaussian_transformations * self.scaling[self._gaussian_edition_mask][..., None, :]

                # So we compute the Transformations from the "Gaussian input space" to the canonical triangle space:
                # These transformations stay valid when editing the scene.
                # We will just need to multiply them by the edited transformations from triangle space to world space
                # to get the edited transformations (quaternion + scaling) from the "Gaussian input space" to world space.
                
                non_invertible_mask = torch.det(tri_transformations[self._edition_mask]) == 0.
                if non_invertible_mask.sum() > 0:
                    print(f"[WARNING] Found {non_invertible_mask.sum()} non-invertible cell transformation(s). Fixing this...")
                    invertible_matrices = tri_transformations[self._edition_mask][non_invertible_mask]
                    
                    if False:
                        columns_1_and_2_are_colinear = torch.isclose(
                            (invertible_matrices[..., 1] *  invertible_matrices[..., 2]).sum(dim=-1).abs(),  # scalar product
                            invertible_matrices[..., 1].norm(dim=-1) * invertible_matrices[..., 2].norm(dim=-1),  # product of norms
                            atol=1e-7,
                        )  # Equality case in Cauchy-Schwarz inequality
                        invertible_matrices[columns_1_and_2_are_colinear] = torch.cat([
                            invertible_matrices[columns_1_and_2_are_colinear][..., 0:1],
                            torch.cross(invertible_matrices[columns_1_and_2_are_colinear][..., 2], invertible_matrices[columns_1_and_2_are_colinear][..., 0], dim=-1)[..., None],
                            invertible_matrices[columns_1_and_2_are_colinear][..., 2:3],
                        ], dim=-1)
                        invertible_matrices[~columns_1_and_2_are_colinear] = torch.cat([
                            torch.cross(invertible_matrices[~columns_1_and_2_are_colinear][..., 1], invertible_matrices[~columns_1_and_2_are_colinear][..., 2], dim=-1)[..., None],
                            invertible_matrices[~columns_1_and_2_are_colinear][..., 1:2],
                            invertible_matrices[~columns_1_and_2_are_colinear][..., 2:3],
                        ], dim=-1)
                    else:
                        mask_0 = (invertible_matrices[..., 0]==0.).all(dim=-1)
                        invertible_matrices[mask_0] = torch.cat([
                            torch.cross(invertible_matrices[mask_0][..., 1], invertible_matrices[mask_0][..., 2], dim=-1)[..., None],
                            invertible_matrices[mask_0][..., 1:2],
                            invertible_matrices[mask_0][..., 2:3],
                        ], dim=-1)
                        
                        mask_1 = (invertible_matrices[..., 1]==0.).all(dim=-1)
                        invertible_matrices[mask_1] = torch.cat([
                            invertible_matrices[mask_1][..., 0:1],
                            torch.cross(invertible_matrices[mask_1][..., 2], invertible_matrices[mask_1][..., 0], dim=-1)[..., None],
                            invertible_matrices[mask_1][..., 2:3],
                        ], dim=-1)
                    
                    # invertible_matrices[..., 0] = torch.cross(invertible_matrices[..., 1], invertible_matrices[..., 2], dim=-1)
                    new_invertible_transfo = tri_transformations[self._edition_mask].clone()
                    new_invertible_transfo[non_invertible_mask] = invertible_matrices
                    tri_transformations[self._edition_mask] = new_invertible_transfo
                
                self.canonical_gaussian_transformations = (
                    torch.linalg.solve(
                        tri_transformations[self._point_cell_indices[self._gaussian_edition_mask]],  # world space > triangle space
                        gaussian_transformations  # input space > world space
                    )
                )
                
            self.editable = True
        
    @torch.no_grad()
    def update_parameters_from_edited_mesh(self):
        # WARNING: The SH parameters are not updated
        if not self.editable:
            raise ValueError("The model is not editable, so it cannot be updated.")
        
        new_scales = scale_inverse_activation(self.scaling)
        new_quaternions = self.quaternions
        new_inner_dist = (
            ((self.inner_verts - self._shell_base_verts) * self.shell_base_normals).sum(dim=-1) 
            / (self.shell_base_normals * self.shell_base_normals).sum(dim=-1)
        ).nan_to_num()
        new_outer_dist = (
            ((self.outer_verts - self._shell_base_verts) * self.shell_base_normals).sum(dim=-1)
            / (self.shell_base_normals * self.shell_base_normals).sum(dim=-1)
        ).nan_to_num()
        
        self._scales[...] = new_scales
        self._quaternions[...] = new_quaternions
        self._inner_dist[...] = new_inner_dist
        self._outer_dist[...] = new_outer_dist
        
        self.editable = False
        if not self.use_simple_adapt:
            self.edited_cache = None
    
    def get_edited_quaternions_and_scales(self):
        # TODO: Handle differently editable gaussians and the rest
        if not self.editable:
            raise ValueError("The model is not editable. Call make_editable() first.")
        if self.use_simple_adapt:
            raise ValueError("The model is editable with simple adaptation.")
        
        scales = torch.empty_like(self._scales)
        quaternions = torch.empty_like(self._quaternions)
        
        # ---Non edited gaussians---
        scales[~self._gaussian_edition_mask] = scale_activation(self._scales[~self._gaussian_edition_mask])
        quaternions[~self._gaussian_edition_mask] = torch_normalize(self._quaternions[~self._gaussian_edition_mask], dim=-1)
        
        # ---Edited Gaussians---
        tri_transformations = self.compute_cell_transformations(faces_mask=self._edition_mask)
        gaussian_transformations = (
            tri_transformations[self._point_cell_indices[self._gaussian_edition_mask]]  # triangle space > world space
            @ self.canonical_gaussian_transformations  # input space > triangle space
        )  # input space > world space
        
        # Scaling factors are the norms of the columns of the transformation matrices
        edited_scales = torch.norm(gaussian_transformations, dim=-2)
        scales[self._gaussian_edition_mask] = edited_scales
        scales = scales.nan_to_num()
        
        # Rotations are the normalized columns of the transformation matrices
        quaternions[self._gaussian_edition_mask] = torch_normalize(
            matrix_to_quaternion(gaussian_transformations / edited_scales[..., None, :].clamp_min(1e-10)), 
            dim=-1
        )
        nan_mask = quaternions.isnan().any(dim=-1)
        quaternions[nan_mask] = torch.tensor([1., 0., 0., 0.], device=self.device, dtype=torch.float32)

        return quaternions, scales
    
    def adapt_to_cameras(self, cameras:CamerasWrapper):        
        self.image_height = int(cameras.height[0].item())
        self.image_width = int(cameras.width[0].item())
        self.fx = cameras.fx[0].item()
        self.fy = cameras.fy[0].item()
        self.fov_x = focal2fov(self.fx, self.image_width)
        self.fov_y = focal2fov(self.fy, self.image_height)
        self.tanfovx = math.tan(self.fov_x * 0.5)
        self.tanfovy = math.tan(self.fov_y * 0.5)
        self.occlusion_culling_rasterizer = None
        
    def get_cameras_spatial_extent(self, nerf_cameras:CamerasWrapper=None, return_average_xyz=False):
        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras
        
        camera_centers = nerf_cameras.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)  # Should it be replaced by the center of camera bbox, i.e. (min + max) / 2?
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        if return_average_xyz:
            return radius, avg_camera_center
        else:
            return radius
        
    def get_points_frosting_size(self, normalize:bool=False, quantile_to_normalize:float=0.85, 
                                 contract_points=False, k_neighbors_for_smoothing=0, 
                                 L_factor_for_contracting=1.):
        """Returns the size of the frosting at each Gaussian center.
        
        Args:
            normalize (bool, optional): If True, the sizes are normalized between 0 and 1. 
                Useful for visualization. Defaults to False.
            quantile_to_normalize (float, optional): The quantile to use as max value for normalization. 
                Defaults to 0.85.
        """
        frosting_shell_size = (self._inner_dist - self._outer_dist).abs()  # (n_verts, )
        
        if contract_points:
            camera_center = self.nerfmodel.training_cameras.p3d_cameras.get_camera_center().mean(dim=0, keepdim=True)
            camera_bbox_half_diag = L_factor_for_contracting * self.nerfmodel.training_cameras.get_spatial_extent() 
            dists = (self._shell_base_verts - camera_center).norm(dim=-1)  # n_verts
            mask = dists > camera_bbox_half_diag  # n_verts
            frosting_shell_size[mask] = frosting_shell_size[mask] * (camera_bbox_half_diag / dists[mask])**2  # n_verts
            
        if k_neighbors_for_smoothing > 0:
            verts_neighbors_idx = knn_points(
                self._shell_base_verts[None], 
                self._shell_base_verts[None], 
                K=k_neighbors_for_smoothing).idx[0]
            frosting_shell_size = frosting_shell_size[verts_neighbors_idx].mean(dim=1)
        
        shell_cell_sizes = torch.cat(
            [frosting_shell_size[:, None], frosting_shell_size[:, None]], dim=1
            )[self._shell_base_faces].transpose(-1, -2)  # (n_faces, 2, 3)
        shell_cell_sizes = shell_cell_sizes.reshape(-1, 6)  # (n_faces, 6)
        gaussian_shell_size = (self.bary_coords * shell_cell_sizes[self._point_cell_indices]).sum(dim=-1)  # (n_gaussians, )
        
        if normalize:
            min_value = gaussian_shell_size.min().item()
            max_value = gaussian_shell_size.quantile(quantile_to_normalize).item()
            gaussian_shell_size = gaussian_shell_size.clamp_max(max_value)
            gaussian_shell_size =  (gaussian_shell_size - min_value) / (max_value - min_value)
            
        return gaussian_shell_size
    
    def get_points_normals(self, normalize:bool=False, normals_to_use:str='base',
                           k_neighbors_for_smoothing:int=0):
        """Returns the normals of the frosting at each Gaussian center.
        
        Args:
            normalize (bool, optional): If True, the normals are normalized between 0 and 1. 
                Useful for visualization. Defaults to False.
            normals_to_use (str, optional): The type of normals to use. Can be 'base', 'outer' or 'inner'.
        """
        if normals_to_use == 'base':
            frosting_normals = self.shell_base_normals  # (n_verts, 3)
        elif normals_to_use == 'outer':
            frosting_normals = self.shell_outer_normals
        elif normals_to_use == 'inner':
            frosting_normals = self.shell_inner_normals
        else:
            raise ValueError("The argument 'normals_to_use' must be 'base', 'outer' or 'inner'.")
        
        if k_neighbors_for_smoothing > 0:
            verts_neighbors_idx = knn_points(
                self._shell_base_verts[None], 
                self._shell_base_verts[None], 
                K=k_neighbors_for_smoothing).idx[0]
            frosting_normals = frosting_normals[verts_neighbors_idx].mean(dim=1)

        shell_cell_normals = torch.cat(
            [frosting_normals[:, None], frosting_normals[:, None]], dim=1
            )[self._shell_base_faces].transpose(-2, -3)  # (n_faces, 2, 3, 3)
        shell_cell_normals = shell_cell_normals.reshape(-1, 6, 3)  # (n_faces, 6, 3)
        gaussian_shell_normals = (self.bary_coords[..., None] * shell_cell_normals[self._point_cell_indices]).sum(dim=-2)  # (n_gaussians, 3)
        
        if normalize:
            gaussian_shell_normals = (gaussian_shell_normals + 1.) / 2.
            
        return gaussian_shell_normals
    
    def get_points_projections_on_mesh(self):
        """Returns the projections of the gaussians on the mesh."""
        # Compute the projections of the gaussians on the mesh
        shell_cell_point_on_mesh = self._shell_base_verts[self._shell_base_faces][:, 0]  # (n_faces, 3)
        ref_point_on_mesh = shell_cell_point_on_mesh[self._point_cell_indices]  # (n_gaussians, 3)
        shell_face_normals = self.shell_base.faces_normals_list()[0]  # (n_faces, 3)
        point_face_normals = shell_face_normals[self._point_cell_indices]  # (n_gaussians, 3)
        point_normals = self.get_points_normals(normalize=False, normals_to_use='base')  # (n_gaussians, 3)
        
        normal_proj = (point_normals * point_face_normals).sum(dim=-1, keepdim=True)  # (n_gaussians, 1)
        proj_mask = normal_proj[..., 0].abs() > 1e-8
        proj_t = torch.zeros(len(point_normals), 1, device=self.device)
        proj_t[proj_mask] = - ((self.points - ref_point_on_mesh) * point_face_normals)[proj_mask].sum(dim=-1, keepdim=True) / normal_proj[proj_mask]
        
        proj_points = ref_point_on_mesh + proj_t * point_normals  # (n_gaussians, 3)
        return proj_points
    
    def get_points_rgb(
        self,
        positions:torch.Tensor=None,
        camera_centers:torch.Tensor=None,
        directions:torch.Tensor=None,
        sh_levels:int=None,
        sh_coordinates:torch.Tensor=None,
        ):
        """Returns the RGB color of the points for the given camera pose.

        Args:
            positions (torch.Tensor, optional): Shape (n_pts, 3). Defaults to None.
            camera_centers (torch.Tensor, optional): Shape (n_pts, 3) or (1, 3). Defaults to None.
            directions (torch.Tensor, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
            
        if positions is None:
            positions = self.points
            if self.use_background_sphere or self.use_background_gaussians:
                positions = torch.cat([positions, self.bg_points], dim=0)

        if camera_centers is not None:
            render_directions = torch.nn.functional.normalize(positions - camera_centers, dim=-1)
        elif directions is not None:
            render_directions = directions
        else:
            raise ValueError("Either camera_centers or directions must be provided.")

        if sh_coordinates is None:
            sh_coordinates = self.sh_coordinates
            if self.use_background_sphere or self.use_background_gaussians:
                sh_coordinates = torch.cat([sh_coordinates, self.bg_sh_coordinates], dim=0)
            
        if sh_levels is None:
            sh_coordinates = sh_coordinates
        else:
            sh_coordinates = sh_coordinates[:, :sh_levels**2]

        shs_view = sh_coordinates.transpose(-1, -2).view(-1, 3, sh_levels**2)
        sh2rgb = eval_sh(sh_levels-1, shs_view, render_directions)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0).view(-1, 3)
        
        return colors
    
    def render_image_gaussian_rasterizer(
        self, 
        nerf_cameras:CamerasWrapper=None, 
        camera_indices:int=0,
        verbose=False,
        bg_color = None,
        sh_deg:int=None,
        sh_rotations:torch.Tensor=None,
        compute_color_in_rasterizer=False,
        compute_covariance_in_rasterizer=True,
        return_2d_radii = False,
        quaternions=None,
        use_same_scale_in_all_directions=False,
        return_opacities:bool=False,
        return_colors:bool=False,
        positions:torch.Tensor=None,
        point_colors=None,
        point_opacities=None,
        point_scaling=None,
        point_sh_coordinates=None,
        use_occlusion_culling=False,
        # For Ablation
        depth_for_filtering=None,  # Shape (H, W)
        depth_output_package=None,
        filtering_tolerance=0.,
        face_idx_to_render=None,
        ):
        """Render an image using the Gaussian Splatting Rasterizer.

        Args:
            nerf_cameras (CamerasWrapper, optional): _description_. Defaults to None.
            camera_indices (int, optional): _description_. Defaults to 0.
            verbose (bool, optional): _description_. Defaults to False.
            bg_color (_type_, optional): _description_. Defaults to None.
            sh_deg (int, optional): _description_. Defaults to None.
            sh_rotations (torch.Tensor, optional): _description_. Defaults to None.
            compute_color_in_rasterizer (bool, optional): _description_. Defaults to False.
            compute_covariance_in_rasterizer (bool, optional): _description_. Defaults to True.
            return_2d_radii (bool, optional): _description_. Defaults to False.
            quaternions (_type_, optional): _description_. Defaults to None.
            use_same_scale_in_all_directions (bool, optional): _description_. Defaults to False.
            return_opacities (bool, optional): _description_. Defaults to False.
            return_colors (bool, optional): _description_. Defaults to False.
            positions (torch.Tensor, optional): _description_. Defaults to None.
            point_colors (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if sh_deg is None:
            sh_deg = self.sh_levels - 1

        if nerf_cameras is None:
            nerf_cameras = self.nerfmodel.training_cameras

        p3d_camera = nerf_cameras.p3d_cameras[camera_indices]

        if bg_color is None:
            bg_color = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
            
        if positions is None:
            positions = self.points
            if self.use_background_sphere or self.use_background_gaussians:
                positions = torch.cat([positions, self.bg_points], dim=0)

        use_torch = False
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = nerf_cameras.camera_to_worlds[camera_indices]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(self.device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        world_view_transform = torch.Tensor(getWorld2View(
            R=R, t=T, tensor=use_torch)).transpose(0, 1).cuda()
        
        proj_transform = getProjectionMatrix(
            p3d_camera.znear.item(), 
            p3d_camera.zfar.item(), 
            self.fov_x, 
            self.fov_y).transpose(0, 1).cuda()
        # TODO: THE TWO FOLLOWING LINES ARE IMPORTANT! IT'S NOT HERE IN 3DGS CODE! Should make a PR when I have time
        proj_transform[..., 2, 0] = - p3d_camera.K[0, 0, 2]
        proj_transform[..., 2, 1] = - p3d_camera.K[0, 1, 2]
        
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_transform.unsqueeze(0))).squeeze(0)
        

        camera_center = p3d_camera.get_camera_center()
        if verbose:
            print("p3d camera_center", camera_center)
            print("ns camera_center", nerf_cameras.camera_to_worlds[camera_indices][..., 3])

        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.image_height),
            image_width=int(self.image_width),
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=sh_deg,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )
    
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # TODO: Change color computation to match 3DGS paper (remove sigmoid)
        if point_colors is None:
            if not compute_color_in_rasterizer:
                if sh_rotations is None:
                    splat_colors = self.get_points_rgb(
                        positions=positions, 
                        camera_centers=camera_center,
                        sh_levels=sh_deg+1,
                        sh_coordinates=point_sh_coordinates)
                else:
                    splat_colors = self.get_points_rgb(
                        positions=positions, 
                        camera_centers=None,
                        directions=(torch.nn.functional.normalize(positions - camera_center, dim=-1).unsqueeze(1) @ sh_rotations)[..., 0, :],
                        sh_levels=sh_deg+1,
                        sh_coordinates=point_sh_coordinates)
                shs = None
            else:
                if point_sh_coordinates is None:
                    shs = self.sh_coordinates
                    if self.use_background_sphere or self.use_background_gaussians:
                        shs = torch.cat([shs, self.bg_sh_coordinates], dim=0)
                else:
                    shs = point_sh_coordinates
                splat_colors = None
        else:
            splat_colors = point_colors
            shs = None
        
        if point_opacities is None:
            splat_opacities = self.strengths.view(-1, 1)
            if self.use_background_sphere or self.use_background_gaussians:
                splat_opacities = torch.cat([splat_opacities, self.bg_strengths.view(-1, 1)], dim=0)
        else:
            splat_opacities = point_opacities
        
        if quaternions is None:
            quaternions = self.quaternions
            if self.use_background_sphere or self.use_background_gaussians:
                quaternions = torch.cat([quaternions, self.bg_quaternions], dim=0)
        
        if not use_same_scale_in_all_directions:
            if point_scaling is None:
                scales = self.scaling
                if self.use_background_sphere or self.use_background_gaussians:
                    scales = torch.cat([scales, self.bg_scaling], dim=0)
            else:
                scales = point_scaling
        else:
            scales = self.scaling.mean(dim=-1, keepdim=True).expand(-1, 3)
            scales = scales.squeeze(0)
        
        if verbose:
            print("Scales:", scales.shape, scales.min(), scales.max())
            
        if use_occlusion_culling and (face_idx_to_render is None):
            if self.occlusion_culling_rasterizer is None:
                print("Initializing occlusion culling rasterizer...")
                self.occlusion_culling_rasterizer = OCMeshRasterizer(
                    cameras=None,
                    raster_settings=OCRasterizationSettings(
                        image_size=(nerf_cameras.gs_cameras[camera_indices].image_height, nerf_cameras.gs_cameras[camera_indices].image_width),
                    ),
                    use_nvdiffrast=True,
                )
            face_idx_to_render = self.occlusion_culling_rasterizer(
                self.shell_base,
                cameras=nerf_cameras,
                cam_idx=camera_indices,
                return_only_pix_to_face=True,
            ).unique()
            
        if (depth_for_filtering is not None) or (face_idx_to_render is not None):
            # Filter Gaussians located behind depth map, only where depth > 0.
            if verbose:
                _n_gaussians_before_filtering = positions.shape[0]
                print(f"\nGaussians before filtering: {_n_gaussians_before_filtering}")
            
            with torch.no_grad():
                if depth_for_filtering is not None:
                    if depth_output_package is None:
                        depth_output_pkg = get_points_depth_in_depthmaps(
                            points_in_world_space=positions,
                            depths=depth_for_filtering,
                            p3d_cameras=p3d_camera,
                            already_in_camera_space=False,
                            return_whole_package=True,
                        )
                    else:
                        depth_output_pkg = depth_output_package
                    
                    in_front_of_depth_mask = depth_output_pkg['real_z'] < depth_output_pkg['map_z'] + filtering_tolerance * self.nerfmodel.training_cameras.get_spatial_extent()
                    no_depth_mask = depth_output_pkg['map_z'] <= 0.
                    render_mask = (in_front_of_depth_mask + no_depth_mask) * depth_output_pkg['inside_mask']
                
                elif face_idx_to_render is not None:
                    _index_mask = torch.zeros(
                        len(self._shell_base_faces), 
                        device=self.device,
                        dtype=torch.bool,
                    )
                    _index_mask[face_idx_to_render] = True
                    render_mask = torch.cat(
                        [
                            _index_mask[self._point_cell_indices], 
                            torch.ones(len(positions) - len(self._point_cell_indices), device=self.device, dtype=torch.bool)
                        ],
                        dim=0
                    )
            
            positions = positions[render_mask]
            if splat_colors is None:
                shs = shs[render_mask]
            else:
                splat_colors = splat_colors[render_mask]
            splat_opacities = splat_opacities[render_mask]
            scales = scales[render_mask]
            quaternions = quaternions[render_mask]
            
            if verbose:
                _n_gaussians_after_filtering = positions.shape[0]
                _n_gaussians_in_fov = depth_output_pkg['inside_mask'].sum().item()
                print(f"Gaussians inside field of view: {_n_gaussians_in_fov}")
                print(f"Gaussians left after filtering: {_n_gaussians_after_filtering}")
                print(f"Proportion of Gaussians kept: {100. * _n_gaussians_after_filtering / _n_gaussians_before_filtering} %")
                print(f"Proportion of Gaussians filtered: {100. - 100. * _n_gaussians_after_filtering / _n_gaussians_before_filtering} %\n")
                print(f"Proportion of Gaussians kept in FoV: {100. * _n_gaussians_after_filtering / _n_gaussians_in_fov} %")
                print(f"Proportion of Gaussians filtered in FoV: {100. - 100. * _n_gaussians_after_filtering / _n_gaussians_in_fov} %\n")

        if not compute_covariance_in_rasterizer:            
            cov3Dmatrix = torch.zeros((scales.shape[0], 3, 3), dtype=torch.float, device=self.device)
            rotation = quaternion_to_matrix(quaternions)

            cov3Dmatrix[:,0,0] = scales[:,0]**2
            cov3Dmatrix[:,1,1] = scales[:,1]**2
            cov3Dmatrix[:,2,2] = scales[:,2]**2
            cov3Dmatrix = rotation @ cov3Dmatrix @ rotation.transpose(-1, -2)
            # cov3Dmatrix = rotation @ cov3Dmatrix
            
            cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=self.device)

            cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
            cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
            cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
            cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
            cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
            cov3D[:, 5] = cov3Dmatrix[:, 2, 2]
            
            quaternions = None
            scales = None
        else:
            cov3D = None
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(self._points, dtype=self._points.dtype, requires_grad=True, device=self.device) + 0
        screenspace_points = torch.zeros(len(positions), 3, dtype=self._bary_coords.dtype, requires_grad=True, device=self.device)
        if return_2d_radii:
            try:
                screenspace_points.retain_grad()
            except:
                print("WARNING: return_2d_radii is True, but failed to retain grad of screenspace_points!")
                pass
        means2D = screenspace_points
        
        if verbose:
            print("points", positions.shape)
            if not compute_color_in_rasterizer:
                print("splat_colors", splat_colors.shape)
            print("splat_opacities", splat_opacities.shape)
            if not compute_covariance_in_rasterizer:
                print("cov3D", cov3D.shape)
                print(cov3D[0])
            else:
                print("quaternions", quaternions.shape)
                print("scales", scales.shape)
            print("screenspace_points", screenspace_points.shape)
        
        if self.project_gaussians_on_base_mesh:
            scales, quaternions = self.project_gaussians_on_mesh()
        
        rendered_image, radii = rasterizer(
            means3D = positions,
            means2D = means2D,
            shs = shs,
            colors_precomp = splat_colors,
            opacities = splat_opacities,
            scales = scales,
            rotations = quaternions,
            cov3D_precomp = cov3D)
        
        if not(return_2d_radii or return_opacities or return_colors):
            return rendered_image.transpose(0, 1).transpose(1, 2)
        
        else:
            outputs = {
                "image": rendered_image.transpose(0, 1).transpose(1, 2),
                "radii": radii,
                "viewspace_points": screenspace_points,
            }
            if return_opacities:
                outputs["opacities"] = splat_opacities
            if return_colors:
                outputs["colors"] = splat_colors
        
            return outputs
        
    def is_inside_frosting(self, points, k_neighbors_to_use=8, proj_th=1e-8):
        scene_scale = self.nerfmodel.training_cameras.get_spatial_extent()
        
        frosting_cell_verts = self.shell_cells_verts  # (n_cells, 2, 3, 3)
        
        # Face normals
        frosting_cell_face_normals_0 = torch_normalize(torch.cross(
            frosting_cell_verts[:, 0, 1] - frosting_cell_verts[:, 0, 0],
            frosting_cell_verts[:, 1, 0] - frosting_cell_verts[:, 0, 0],
        ), dim=-1)[:, None]  # OK
        frosting_cell_face_normals_1 = torch_normalize(torch.cross(
            frosting_cell_verts[:, 0, 2] - frosting_cell_verts[:, 0, 1],
            frosting_cell_verts[:, 1, 1] - frosting_cell_verts[:, 0, 1],
        ), dim=-1)[:, None]  # OK
        frosting_cell_face_normals_2 = torch_normalize(torch.cross(
            frosting_cell_verts[:, 1, 0] - frosting_cell_verts[:, 0, 0],
            frosting_cell_verts[:, 0, 2] - frosting_cell_verts[:, 0, 0],
        ), dim=-1)[:, None]  # OK
        frosting_cell_face_normals_3 = torch_normalize(torch.cross(
            frosting_cell_verts[:, 0, 2] - frosting_cell_verts[:, 0, 0],
            frosting_cell_verts[:, 0, 1] - frosting_cell_verts[:, 0, 0],
        ), dim=-1)[:, None]  # OK
        frosting_cell_face_normals_4 = torch_normalize(torch.cross(
            frosting_cell_verts[:, 1, 1] - frosting_cell_verts[:, 1, 0],
            frosting_cell_verts[:, 1, 2] - frosting_cell_verts[:, 1, 0],
        ), dim=-1)[:, None]  # OK
        frosting_cell_face_normals = [
            frosting_cell_face_normals_0,
            frosting_cell_face_normals_1,
            frosting_cell_face_normals_2,
            frosting_cell_face_normals_3,
            frosting_cell_face_normals_4,
        ]
        frosting_cell_face_normals = torch.cat(frosting_cell_face_normals, dim=1)  # (n_cells, 5, 3)
        
        # Face points
        frosting_cell_face_points_0 = frosting_cell_verts[:, 0, 0][:, None]
        frosting_cell_face_points_1 = frosting_cell_verts[:, 0, 1][:, None]
        frosting_cell_face_points_2 = frosting_cell_verts[:, 0, 0][:, None]
        frosting_cell_face_points_3 = frosting_cell_verts[:, 0, 0][:, None]
        frosting_cell_face_points_4 = frosting_cell_verts[:, 1, 0][:, None]
        frosting_cell_face_points = [
            frosting_cell_face_points_0,
            frosting_cell_face_points_1,
            frosting_cell_face_points_2,
            frosting_cell_face_points_3,
            frosting_cell_face_points_4,
        ]
        frosting_cell_face_points = torch.cat(frosting_cell_face_points, dim=1)  # (n_cells, 5, 3)
        
        # Cell centers
        frosting_cell_centers = self.shell_cells_verts.reshape(-1, 6, 3).mean(dim=1)  # (n_cells, 3)
        
        # Compute cells closest to the points
        knn_idx = knn_points(points[None], frosting_cell_centers[None], K=k_neighbors_to_use).idx[0]  # (n_points, k_neighbors_to_use)
        
        # Compute the projection of the points
        proj = (points[:, None, None] - frosting_cell_face_points[knn_idx]) * frosting_cell_face_normals[knn_idx]  # (n_points, k_neighbors_to_use, 5, 3)
        proj = proj.sum(dim=-1)  # (n_points, k_neighbors_to_use, 5)
        inside = (proj > scene_scale * proj_th).all(dim=-1)  # (n_points, k_neighbors_to_use)
        inside = inside.any(dim=-1)  # (n_points, )
        
        return inside
        
    def save_model(self, path, **kwargs):
        checkpoint = {}
        checkpoint['state_dict'] = self.state_dict()
        for k, v in kwargs.items():
            checkpoint[k] = v
        torch.save(checkpoint, path) 
    

def compute_level_surface_points_and_range_from_camera(
    self:SuGaR,
    nerf_cameras=None,
    cam_idx=0,
    rasterizer=None,
    surface_levels=[0.1],  # 0.5 or 0.1
    n_surface_points=-1,
    primitive_types=None,  # Should be 'diamond'
    triangle_scale=None,  # Should be 2.
    splat_mesh=True,  # True
    n_points_in_range=21,  # 21
    range_size=3.,  # 3.
    n_points_per_pass=2_000_000,
    density_factor=1.,
    return_pixel_idx=False,
    return_gaussian_idx=False,
    return_normals=False,
    compute_flat_normals=False,
    compute_intersection_for_flat_gaussian=False,  # Should be False
    use_gaussian_depth=False,  # False until now. TODO: Test with True
    just_use_depth_as_level=False, # Should be False
    use_last_intersection_as_inner_level_point=False,  # If False, uses the 2nd crossing point as inner level point
    ):
    # Remember to reset neighbors and update texture features before calling this function
    if nerf_cameras is None:
        nerf_cameras = self.nerfmodel.training_cameras
    
    if primitive_types is not None:
        self.primitive_types = primitive_types
        
    if triangle_scale is not None:
        self.triangle_scale = triangle_scale
        
    if rasterizer is None:
        faces_per_pixel = 10
        max_faces_per_bin = 50_000

        mesh_raster_settings = RasterizationSettings(
            image_size=(self.image_height, self.image_width),
            blur_radius=0.0, 
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=max_faces_per_bin
        )
        rasterizer = MeshRasterizer(
                cameras=nerf_cameras.p3d_cameras[cam_idx], 
                raster_settings=mesh_raster_settings,
            )
        
    p3d_cameras = nerf_cameras.p3d_cameras[cam_idx]
    
    # Compute splatted depth
    # (either using Gaussian Splatting rasterizer, 
    # or using PyTorch3D's triangle rasterizer for sharper results 
    # and instant access to closest gaussian index for each pixel.)
    if use_gaussian_depth:
        point_depth = p3d_cameras.get_world_to_view_transform().transform_points(self.points)[..., 2:].expand(-1, 3)
        depth = self.render_image_gaussian_rasterizer( 
                camera_indices=cam_idx,
                bg_color=torch.Tensor([-1., -1., -1.]).to(self.device),
                sh_deg=0,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=False,
                use_same_scale_in_all_directions=False,
                point_colors=point_depth,
            ).contiguous()[..., 0]
    else:
        if True:
            textures_img = self.get_texture_img(
                nerf_cameras=nerf_cameras, 
                cam_idx=cam_idx,
                sh_levels=self.sh_levels,
                )

        if splat_mesh:
            mesh = self.splat_mesh(p3d_cameras)
        else:
            mesh = self.mesh
        if True:
            mesh.textures._maps_padded = textures_img[None]

        fragments = rasterizer(mesh, cameras=p3d_cameras)
        depth = fragments.zbuf[0, ..., 0]
    no_depth_mask = depth < 0.
    depth[no_depth_mask] = depth.max() * 1.05
    
    # We backproject the points in world space
    batch_size = 1
    x_tab = torch.Tensor([[i for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
    y_tab = torch.Tensor([[j for j in range(self.image_width)] for i in range(self.image_height)]).to(self.device)
    ndc_x_tab = self.image_width / min(self.image_width,
                                            self.image_height) - (y_tab / (min(self.image_width,
                                                                                self.image_height) - 1)) * 2
    ndc_y_tab = self.image_height / min(self.image_width,
                                                self.image_height) - (x_tab / (min(self.image_width,
                                                                                self.image_height) - 1)) * 2

    ndc_points = torch.cat((ndc_x_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                            ndc_y_tab.view(1, -1, 1).expand(batch_size, -1, -1),
                            depth.view(batch_size, -1, 1)),
                            dim=-1
                            ).view(batch_size, self.image_height * self.image_width, 3)
    
    fov_cameras = nerf_cameras.p3d_cameras[cam_idx]
    no_proj_mask = no_depth_mask.view(-1)
    ndc_points = ndc_points[0][~no_proj_mask][None]  # Remove pixels with no projection
    if n_surface_points == -1:
        n_surface_points = ndc_points.shape[1]
        ndc_points_idx = torch.arange(n_surface_points)
    else:
        n_surface_points = min(n_surface_points, ndc_points.shape[1])
        ndc_points_idx = torch.randperm(ndc_points.shape[1])[:n_surface_points]
        ndc_points = ndc_points[:, ndc_points_idx]
    all_world_points = fov_cameras.unproject_points(ndc_points, scaled_depth_input=False).view(-1, 3)
    
    # Gather info about gaussians surrounding each 3D point
    if use_gaussian_depth:
        closest_gaussians_idx = self.get_gaussians_closest_to_samples(all_world_points)
        gaussian_idx = closest_gaussians_idx[..., 0]
    else:
        gaussian_idx = fragments.pix_to_face[..., 0].view(-1) // self.n_triangles_per_gaussian
        gaussian_idx = gaussian_idx[~no_proj_mask][ndc_points_idx]
        closest_gaussians_idx = self.knn_idx[gaussian_idx]
    
    # We compute the standard deviation of the gaussian at each point
    gaussian_to_camera = torch.nn.functional.normalize(fov_cameras.get_camera_center() - self.points, dim=-1)
    gaussian_standard_deviations = (self.scaling * quaternion_apply(quaternion_invert(self.quaternions), gaussian_to_camera)).norm(dim=-1)
    points_stds = gaussian_standard_deviations[closest_gaussians_idx[..., 0]]
    
    # We compute ray samples
    points_range = torch.linspace(-range_size, range_size, n_points_in_range).to(self.device).view(1, -1, 1)  # (1, n_points_in_range, 1)
    points_range = points_range * points_stds[..., None, None].expand(-1, n_points_in_range, 1)  # (n_points, n_points_in_range, 1)
    camera_to_samples = torch.nn.functional.normalize(all_world_points - fov_cameras.get_camera_center(), dim=-1)  # (n_points, 3)
    samples = (all_world_points[:, None, :] + points_range * camera_to_samples[:, None, :]).view(-1, 3)  # (n_points * n_points_in_range, 3)
    samples_closest_gaussians_idx = closest_gaussians_idx[:, None, :].expand(-1, n_points_in_range, -1).reshape(-1, self.knn_to_track)
    
    # Compute densities of all samples
    densities = torch.zeros(len(samples), dtype=torch.float, device=self.device)
    gaussian_strengths = self.strengths
    gaussian_centers = self.points
    gaussian_inv_scaled_rotation = self.get_covariance(return_full_matrix=True, return_sqrt=True, inverse_scales=True)
    
    for i in range(0, len(samples), n_points_per_pass):
        i_start = i
        i_end = min(len(samples), i + n_points_per_pass)
        
        pass_closest_gaussians_idx = samples_closest_gaussians_idx[i_start:i_end]
        
        closest_gaussian_centers = gaussian_centers[pass_closest_gaussians_idx]
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[pass_closest_gaussians_idx]
        closest_gaussian_strengths = gaussian_strengths[pass_closest_gaussians_idx]

        shift = (samples[i_start:i_end, None] - closest_gaussian_centers)
        if not compute_intersection_for_flat_gaussian:
            warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
            neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        else:
            closest_gaussian_normals = self.get_normals()[pass_closest_gaussians_idx]
            closest_gaussian_min_scales = self.scaling.min(dim=-1)[0][pass_closest_gaussians_idx]
            neighbor_opacities = (shift * closest_gaussian_normals).sum(dim=-1).pow(2)  / (closest_gaussian_min_scales).pow(2)
        neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
        pass_densities = neighbor_opacities.sum(dim=-1)
        pass_density_mask = pass_densities >= 1.
        pass_densities[pass_density_mask] = pass_densities[pass_density_mask] / (pass_densities[pass_density_mask].detach() + 1e-12)
        
        densities[i_start:i_end] = pass_densities
    densities = densities.reshape(-1, n_points_in_range)
    
    # Compute isosurface intersection points
    all_outputs = {}
    for surface_level in surface_levels:
        outputs = {}
                    
        under_level = (densities - surface_level < 0)
        above_level = (densities - surface_level > 0)

        _, first_point_above_level = above_level.max(dim=-1, keepdim=True)
        if use_last_intersection_as_inner_level_point:
            # Use last crossing point as first point above level
            last_point_above_level = (n_points_in_range - 1) - above_level.flip(dims=(-1,)).max(dim=-1, keepdim=True)[1]  # TO ADD
        else:
            # Use second crossing point as last point above level
            last_point_above_level = under_level[..., 1:] * above_level[..., :-1]
            _, last_point_above_level = last_point_above_level.max(dim=-1, keepdim=True)  # There is a "+ 1 - 1" hidden here
            # The second crossing point can't be the first point above level.
            # If the value is 0, it means that there is no second crossing point and we should use the last point.
            last_point_above_level[last_point_above_level == 0] = n_points_in_range-1

        empty_pixels = ~under_level[..., 0] + (first_point_above_level[..., 0] == 0)

        if not just_use_depth_as_level:
            valid_densities = densities[~empty_pixels]
            valid_range = points_range[~empty_pixels][..., 0]
            valid_first_point_above_level = first_point_above_level[~empty_pixels]

            first_value_above_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level).view(-1)
            value_before_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

            first_t_above_level = valid_range.gather(dim=-1, index=valid_first_point_above_level).view(-1)
            t_before_level = valid_range.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

            intersection_t = (surface_level - value_before_level) / (first_value_above_level - value_before_level) * (first_t_above_level - t_before_level) + t_before_level
            intersection_points = (all_world_points[~empty_pixels] + intersection_t[:, None] * camera_to_samples[~empty_pixels])
            
            # ---TO ADD
            last_point_above_level_is_not_bound = (last_point_above_level[..., 0] < n_points_in_range-1)
            valid_densities_last = densities[(~empty_pixels) * last_point_above_level_is_not_bound]
            valid_range_last = points_range[(~empty_pixels) * last_point_above_level_is_not_bound][..., 0]            
            valid_last_point_above_level = last_point_above_level[(~empty_pixels) * last_point_above_level_is_not_bound]
            
            last_value_above_level = valid_densities_last.gather(dim=-1, index=valid_last_point_above_level).view(-1)
            value_after_level = valid_densities_last.gather(dim=-1, index=valid_last_point_above_level+1).view(-1)
            
            last_t_above_level = valid_range_last.gather(dim=-1, index=valid_last_point_above_level).view(-1)
            t_after_level = valid_range_last.gather(dim=-1, index=valid_last_point_above_level+1).view(-1)
            
            inner_intersection_t = 0. + valid_range[..., -1]
            inner_intersection_t[last_point_above_level_is_not_bound[~empty_pixels]] = (surface_level - last_value_above_level) / (value_after_level - last_value_above_level) * (t_after_level - last_t_above_level) + last_t_above_level
            inner_intersection_points = (all_world_points[~empty_pixels] + inner_intersection_t[:, None] * camera_to_samples[~empty_pixels])
            # ---End TO ADD
        else:
            empty_pixels = torch.zeros_like(empty_pixels, dtype=torch.bool)
            intersection_points = all_world_points[~empty_pixels]
            inner_intersection_points = all_world_points[~empty_pixels]  # TO ADD
        outputs['intersection_points'] = intersection_points
        outputs['inner_intersection_points'] = inner_intersection_points  # TO ADD
        
        if return_pixel_idx:
            pixel_idx = torch.arange(self.image_height * self.image_width, dtype=torch.long, device=self.device)
            pixel_idx = pixel_idx[~no_proj_mask][ndc_points_idx][~empty_pixels]
            outputs['pixel_idx'] = pixel_idx
            
        if return_gaussian_idx:
            outputs['gaussian_idx'] = gaussian_idx[~empty_pixels]
        
        if return_normals:                
            points_closest_gaussians_idx = closest_gaussians_idx[~empty_pixels]

            closest_gaussian_centers = gaussian_centers[points_closest_gaussians_idx]
            closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[points_closest_gaussians_idx]
            closest_gaussian_strengths = gaussian_strengths[points_closest_gaussians_idx]

            shift = (intersection_points[:, None] - closest_gaussian_centers)
            if not compute_intersection_for_flat_gaussian:
                warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
            else:
                closest_gaussian_normals = self.get_normals()[points_closest_gaussians_idx]
                closest_gaussian_min_scales = self.scaling.min(dim=-1)[0][points_closest_gaussians_idx]
                neighbor_opacities = (shift * closest_gaussian_normals).sum(dim=-1).pow(2)  / (closest_gaussian_min_scales).pow(2)
            neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
            
            if not compute_flat_normals:
                density_grad = (neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]).sum(dim=-2)
            else:
                closest_gaussian_normals = self.get_normals()[points_closest_gaussians_idx]
                closest_gaussian_min_scales = self.scaling.min(dim=-1, keepdim=True)[0][points_closest_gaussians_idx]
                density_grad = (
                    neighbor_opacities[..., None] * 
                    1. / (closest_gaussian_min_scales).pow(2)  * (shift * closest_gaussian_normals).sum(dim=-1, keepdim=True) * closest_gaussian_normals
                    ).sum(dim=-2)
            
            intersection_normals = -torch.nn.functional.normalize(density_grad, dim=-1)
            outputs['normals'] = intersection_normals
                
        all_outputs[surface_level] = outputs

    return all_outputs


def compute_level_points_along_normals(
    gaussian_model: Union[GaussianSplattingWrapper, SuGaR],
    mesh_verts: torch.Tensor,
    mesh_verts_normals: torch.Tensor,
    inner_range: torch.Tensor,
    outer_range: torch.Tensor,
    n_samples_per_vertex: int=21,
    n_points_per_pass: int=2_000_000,
    n_closest_gaussians_to_use: int=16,
    level: float=0.1,
    smooth_points: bool=True,
    n_neighbors_for_smoothing: int=4,
    use_last_intersection_as_inner_level_point: bool=True,
    min_clamping_inner_dist: torch.Tensor=None,
    max_clamping_outer_dist: torch.Tensor=None,
    min_layer_size: float=0.,
    ):
    """
    Computes level points of a given Gaussian Splatting model along the vertex normals of a given mesh.
    The model can be either a GaussianSplattingWrapper (vanilla 3DGS) or a SuGaR model.
    
    Args:
    - gaussian_model: A GaussianSplattingWrapper or a SuGaR model.
    - mesh_verts: A tensor of shape (n_verts, 3) containing the vertices of the mesh.
    - mesh_verts_normals: A tensor of shape (n_verts, 3) containing the normals of the mesh vertices.
    - inner_range: A tensor of shape (n_verts, ) containing the inner range for looking for level points along the normals. Should be positive.
    - outer_range: A tensor of shape (n_verts, ) containing the outer range for looking for level points along the normals. Should be negative.
    - n_samples_per_vertex: An integer representing the number of samples to take along the normals of each vertex.
    - n_points_per_pass: An integer
    - n_closest_gaussians_to_use: An integer
    - level: A float
    - smooth_points: A boolean
    - n_neighbors_for_smoothing: An integer
    - use_last_intersection_as_inner_level_point: A boolean
    - clamping_inner_dist: A tensor of shape (n_verts, ) or None
    - clamping_outer_dist: A tensor of shape (n_verts, ) or None
    - min_layer_size: A float
    """
    
    # Get gaussian model parameters
    if type(gaussian_model) == GaussianSplattingWrapper:
        xyz = gaussian_model.gaussians.get_xyz
        scaling = gaussian_model.gaussians.get_scaling
        rotation = gaussian_model.gaussians.get_rotation
        strengths = gaussian_model.gaussians.get_opacity
        training_cameras = gaussian_model.training_cameras
    elif type(gaussian_model) == SuGaR:
        xyz = gaussian_model.points
        scaling = gaussian_model.scaling
        rotation = gaussian_model.quaternions
        strengths = gaussian_model.strengths
        training_cameras = gaussian_model.nerfmodel.training_cameras
    else:
        raise ValueError("gaussian_model should be either a SuGaR or a GaussianSplattingWrapper.")
    
    # Compute closest unconstrained gaussians to each vertex
    closest_gaussians_idx = knn_points(
        mesh_verts[None], 
        xyz[None], 
        K=n_closest_gaussians_to_use
    ).idx[0]  # (n_verts, n_closest_gaussians_to_use)
    
    # Compute samples
    verts_range = torch.linspace(0., 1., n_samples_per_vertex).to(gaussian_model.device).view(1, -1, 1)  # (1, n_samples_per_vertex, 1)
    verts_range = verts_range * (inner_range - outer_range)[..., None, None] + outer_range[..., None, None]  # (n_verts, n_samples_per_vertex, 1)
    samples = (mesh_verts[:, None, :] + verts_range * mesh_verts_normals[:, None, :]).view(-1, 3)  # (n_verts * n_samples_per_vertex, 3)
    samples_closest_gaussians_idx = closest_gaussians_idx[:, None, :].expand(
        -1, n_samples_per_vertex, -1
        ).reshape(-1, n_closest_gaussians_to_use
                    )  # (n_verts * n_samples_per_vertex, n_closest_gaussians_to_use)
        
    # Compute densities of all samples using unconstrained gaussians
    densities = torch.zeros(len(samples), dtype=torch.float, device=gaussian_model.device)
    gaussian_strengths = strengths  # (n_points, 1)
    gaussian_centers = xyz  # (n_points, 3)
    gaussian_inv_scaling = 1. / scaling.clamp(min=1e-8)  # (n_points, 3)
    gaussian_inv_scaled_rotation = quaternion_to_matrix(rotation) * gaussian_inv_scaling[:, None]  # (n_points, 3, 3)
    
    for i in range(0, len(samples), n_points_per_pass):
        i_start = i
        i_end = min(len(samples), i + n_points_per_pass)
        
        pass_closest_gaussians_idx = samples_closest_gaussians_idx[i_start:i_end]  # (n_points_per_pass, n_closest_gaussians_to_use)
        
        closest_gaussian_centers = gaussian_centers[pass_closest_gaussians_idx]  # (n_points_per_pass, n_closest_gaussians_to_use, 3)
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[pass_closest_gaussians_idx]  # (n_points_per_pass, n_closest_gaussians_to_use, 3, 3)
        closest_gaussian_strengths = gaussian_strengths[pass_closest_gaussians_idx]  # (n_points_per_pass, n_closest_gaussians_to_use, 1)
        
        shift = (samples[i_start:i_end, None] - closest_gaussian_centers)  # (n_points_per_pass, n_closest_gaussians_to_use, 3)
        warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]  # (n_points_per_pass, n_closest_gaussians_to_use, 3, 1)
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)  # (n_points_per_pass, n_closest_gaussians_to_use)

        neighbor_opacities = closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
        pass_densities = neighbor_opacities.sum(dim=-1)
        pass_density_mask = pass_densities >= 1.
        pass_densities[pass_density_mask] = pass_densities[pass_density_mask] / (pass_densities[pass_density_mask].detach() + 1e-12)
        
        densities[i_start:i_end] = pass_densities
    densities = densities.reshape(-1, n_samples_per_vertex)  # (n_verts, n_samples_per_vertex)
        
    # Compute shell extreme points
    under_level = (densities - level < 0)
    above_level = (densities - level > 0)

    _, first_point_above_level = above_level.max(dim=-1, keepdim=True)
    if use_last_intersection_as_inner_level_point:
        last_point_above_level = (n_samples_per_vertex - 1) - above_level.flip(dims=(-1,)).max(dim=-1, keepdim=True)[1]
    else:
        # Use second crossing point as last point above level
        last_point_above_level = under_level[..., 1:] * above_level[..., :-1]
        _, last_point_above_level = last_point_above_level.max(dim=-1, keepdim=True)  # There is a "+ 1 - 1" hidden here
        # The second crossing point can't be the first point above level.
        # If the value is 0, it means that there is no second crossing point and we should use the last point.
        last_point_above_level[last_point_above_level == 0] = n_samples_per_vertex-1
    
    # Outer point
    outer_point_above_level_is_not_bound = (first_point_above_level[..., 0] > 0)
    valid_densities = densities[outer_point_above_level_is_not_bound]
    valid_range = verts_range[outer_point_above_level_is_not_bound][..., 0]
    valid_first_point_above_level = first_point_above_level[outer_point_above_level_is_not_bound]

    first_value_above_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level).view(-1)
    value_before_level = valid_densities.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

    first_t_above_level = valid_range.gather(dim=-1, index=valid_first_point_above_level).view(-1)
    t_before_level = valid_range.gather(dim=-1, index=valid_first_point_above_level-1).view(-1)

    outer_dist = 0. + verts_range[..., 0, 0]
    outer_dist[outer_point_above_level_is_not_bound] = (level - value_before_level) / (first_value_above_level - value_before_level) * (first_t_above_level - t_before_level) + t_before_level
    print("Outer point is bound:", (~outer_point_above_level_is_not_bound).sum().item())
    
    # Inner point
    last_point_above_level_is_not_bound = (last_point_above_level[..., 0] < n_samples_per_vertex-1)
    valid_densities_last = densities[last_point_above_level_is_not_bound]
    valid_range_last = verts_range[last_point_above_level_is_not_bound][..., 0]            
    valid_last_point_above_level = last_point_above_level[last_point_above_level_is_not_bound]
    
    last_value_above_level = valid_densities_last.gather(dim=-1, index=valid_last_point_above_level).view(-1)
    value_after_level = valid_densities_last.gather(dim=-1, index=valid_last_point_above_level+1).view(-1)
    
    last_t_above_level = valid_range_last.gather(dim=-1, index=valid_last_point_above_level).view(-1)
    t_after_level = valid_range_last.gather(dim=-1, index=valid_last_point_above_level+1).view(-1)
    
    inner_dist = 0. + verts_range[..., -1, 0]
    inner_dist[last_point_above_level_is_not_bound] = (level - last_value_above_level) / (value_after_level - last_value_above_level) * (t_after_level - last_t_above_level) + last_t_above_level
    print("Inner point is bound:", (~last_point_above_level_is_not_bound).sum().item())
    
    # Points for which all densities are under the shell level
    empty_mask = (~outer_point_above_level_is_not_bound) * (~last_point_above_level_is_not_bound) * (under_level[..., 0])
    outer_dist[empty_mask] = (inner_range + outer_range)[empty_mask] / 2
    inner_dist[empty_mask] = (inner_range + outer_range)[empty_mask] / 2
    print("Range is empty:", empty_mask.sum().item())
    
    # Set minimal frosting size
    if min_layer_size > 0:
        print(f"Setting a minimal layer size of {min_layer_size} * spatial extent...")
        layer_size = (inner_dist - outer_dist).abs()  # (n_verts, )
        flat_layer_mask = layer_size < min_layer_size * training_cameras.get_spatial_extent()
        print("Layer size smaller than minimal layer size:", flat_layer_mask.sum().item())
        flat_layer_center_dist = (inner_range + outer_range)[flat_layer_mask] / 2
        outer_dist[flat_layer_mask] = flat_layer_center_dist - 0.5 * min_layer_size * training_cameras.get_spatial_extent()
        inner_dist[flat_layer_mask] = flat_layer_center_dist + 0.5 * min_layer_size * training_cameras.get_spatial_extent()
    
    # Clamp inner and outer distances if needed
    if min_clamping_inner_dist is not None:
        print(f"Clamping inner distance...")
        print("Inner distance clamped:", (inner_dist < min_clamping_inner_dist).sum().item())
        inner_dist = inner_dist.clamp_min(min_clamping_inner_dist)
    if max_clamping_outer_dist is not None:
        print(f"Clamping outer distance...")
        print("Outer distance clamped:", (outer_dist > max_clamping_outer_dist).sum().item())
        outer_dist = outer_dist.clamp_max(max_clamping_outer_dist)
    
    # Smooth initial shell
    if smooth_points:
        print(f"Smoothing layer using neighborhood of {n_neighbors_for_smoothing} vertices...")
        verts_neighbors_idx = knn_points(
            mesh_verts[None], 
            mesh_verts[None], 
            K=n_neighbors_for_smoothing).idx[0]
        outer_dist = outer_dist[verts_neighbors_idx].mean(dim=1)
        inner_dist = inner_dist[verts_neighbors_idx].mean(dim=1)

    outer_verts = mesh_verts + outer_dist[:, None] * mesh_verts_normals
    inner_verts = mesh_verts + inner_dist[:, None] * mesh_verts_normals
    
    outputs = {
        "outer_verts": outer_verts,
        "inner_verts": inner_verts,
        "outer_dist": outer_dist,
        "inner_dist": inner_dist,
    }
    return outputs
    
    
def convert_frosting_into_gaussians(
    frosting:Frosting,
    means=None,
    quaternions=None,
    scales=None,
    opacities=None,
    sh_coordinates_dc=None,
    sh_coordinates_rest=None,
    ):
    new_gaussians = GaussianModel(frosting.sh_levels - 1)
    
    with torch.no_grad():
        # Means
        if means is None:
            points = frosting.points
            if frosting.use_background_gaussians or frosting.use_background_sphere:
                points = torch.cat([points, frosting.bg_points], dim=0)
        else:
            points = means
            
        # Opacities
        if opacities is None:
            opacities = inverse_sigmoid(frosting.strengths)
            if frosting.use_background_gaussians or frosting.use_background_sphere:
                opacities = torch.cat([opacities, inverse_sigmoid(frosting.bg_strengths)], dim=0)
        else:
            opacities = inverse_sigmoid(opacities)
            
        # Scaling
        if scales is None:
            scales = scale_inverse_activation(frosting.scaling)
            if frosting.use_background_gaussians or frosting.use_background_sphere:
                scales = torch.cat([scales, scale_inverse_activation(frosting.bg_scaling)], dim=0)
        else:
            scales = scale_inverse_activation(scales)
        
        # Quaternions
        if quaternions is None:
            rots = frosting.quaternions
            if frosting.use_background_gaussians or frosting.use_background_sphere:
                rots = torch.cat([rots, frosting.bg_quaternions], dim=0)
        else:
            rots = quaternions
        
        # Color features
        if sh_coordinates_dc is None:
            features_dc = frosting._sh_coordinates_dc.permute(0, 2, 1)
            if frosting.use_background_gaussians or frosting.use_background_sphere:
                features_dc = torch.cat([features_dc, frosting._bg_sh_coordinates_dc.permute(0, 2, 1)], dim=0)
        else:
            features_dc = sh_coordinates_dc
        if sh_coordinates_rest is None:
            features_extra = frosting._sh_coordinates_rest.permute(0, 2, 1)
            if frosting.use_background_gaussians or frosting.use_background_sphere:
                features_extra = torch.cat([features_extra, frosting._bg_sh_coordinates_rest.permute(0, 2, 1)], dim=0)
        else:
            features_extra = sh_coordinates_rest
        
        xyz = points.cpu().numpy()
        opacities = opacities.cpu().numpy()
        features_dc = features_dc.cpu().numpy()
        features_extra = features_extra.cpu().numpy()
        scales = scales.cpu().numpy()
        rots = rots.cpu().numpy()

    new_gaussians._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    new_gaussians._features_dc = torch.nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    new_gaussians._features_rest = torch.nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    new_gaussians._opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    new_gaussians._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    new_gaussians._rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    new_gaussians.active_sh_degree = new_gaussians.max_sh_degree
    
    return new_gaussians


def load_frosting_model(
    frosting_checkpoint_path, 
    nerfmodel=None,
    device=None,
    **kwargs,  # Just for compatibility with the old function
):
    if device is None:
        if nerfmodel is None:
            raise ValueError("You must provide a device if no NeRFmodel is provided.")
        device = nerfmodel.device
        
    print("Creating Frosting model from checkpoint:", frosting_checkpoint_path)
    checkpoint = torch.load(frosting_checkpoint_path, map_location=device)
    n_gaussians_in_frosting = checkpoint['state_dict']['_bary_coords'].shape[0]
    sh_levels = int(np.sqrt(checkpoint['state_dict']['_sh_coordinates_rest'].shape[1] + 1))
    
    if "_bg_points" in checkpoint['state_dict']:
        use_background_gaussians = True
    else:
        use_background_gaussians = False
        print("[WARNING] No background Gaussians found.")
    
    # create a opend3d mesh from the verts and faces
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(checkpoint['state_dict']['_shell_base_verts'].detach().cpu().numpy()),
        triangles=o3d.utility.Vector3iVector(checkpoint['state_dict']['_shell_base_faces'].detach().cpu().numpy()),
    )
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
        o3d.utility.Vector3dVector(0.5 * torch.ones_like(checkpoint['state_dict']['_shell_base_verts'].detach()).cpu().numpy())
    )
    
    frosting = Frosting(
        nerfmodel,
        coarse_sugar=None,
        # sh_levels=nerfmodel.gaussians.max_sh_degree+1,
        sh_levels=sh_levels,
        shell_base_to_bind=o3d_mesh,  # Open3D mesh
        learn_shell=False,  # False
        min_frosting_factor=-0.5,  # -0.5
        max_frosting_factor=1.5,  # 1.5
        n_gaussians_in_frosting=n_gaussians_in_frosting,  # 2_000_000
        # Frosting initialization
        n_closest_gaussians_to_use_for_initializing_frosting=1,  # 16
        n_points_per_pass_for_initializing_frosting=2_000_000,  # 2_000_000
        n_samples_per_vertex_for_initializing_frosting=2,  # 51?
        frosting_level=0.01,  # 0.01
        smooth_initial_frosting=False,  # True
        n_neighbors_for_smoothing_initial_frosting=4,  # 4
        # Edition
        editable=False,
        use_softmax_for_bary_coords=True,
        min_frosting_range=0.001,
        min_frosting_size=0.001,
        initial_proposal_std_range=3.,
        final_proposal_range=3.,
        final_clamping_range=0.1,
        use_background_sphere=False,
        use_background_gaussians=False,
        contract_points=False,
        avoid_self_intersections=False,
        use_constant_frosting_size=False,
        device=device,
    )
    
    # Retrieve the background Gaussians parameters
    if use_background_gaussians:
        frosting._bg_points = torch.nn.Parameter(checkpoint['state_dict']['_bg_points'], requires_grad=True).to(device)
        frosting._bg_opacities = torch.nn.Parameter(checkpoint['state_dict']['_bg_opacities'], requires_grad=True).to(device)
        frosting._bg_sh_coordinates_dc = torch.nn.Parameter(checkpoint['state_dict']['_bg_sh_coordinates_dc'], requires_grad=True).to(device)
        frosting._bg_sh_coordinates_rest = torch.nn.Parameter(checkpoint['state_dict']['_bg_sh_coordinates_rest'], requires_grad=True).to(device)
        frosting._bg_scales = torch.nn.Parameter(checkpoint['state_dict']['_bg_scales'], requires_grad=True).to(device)
        frosting._bg_quaternions = torch.nn.Parameter(checkpoint['state_dict']['_bg_quaternions'], requires_grad=True).to(device)

    frosting.n_points = frosting._bary_coords.shape[0]
    if use_background_gaussians:
        frosting.use_background_gaussians = True
        frosting.n_points = frosting.n_points + frosting._bg_points.shape[0]
    
    # Load the frosting
    frosting.load_state_dict(checkpoint['state_dict'])
    print("Frosting loaded.")
    
    return frosting


def create_frosting_model_from_sugar(
    sugar_checkpoint_path, nerfmodel,
    ):
    
    print("Creating Frosting model from checkpoint:", sugar_checkpoint_path)
    from frosting_scene.sugar_model import load_refined_model as load_sugar_model
    
    sugar = load_sugar_model(sugar_checkpoint_path, nerfmodel)    
    use_constant_frosting_size = True
     
    # create a opend3d mesh from the verts and faces
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(sugar.surface_mesh.verts_list()[0].detach().cpu().numpy()),
        triangles=o3d.utility.Vector3iVector(sugar.surface_mesh.faces_list()[0].detach().cpu().numpy()),
    )
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
        o3d.utility.Vector3dVector(0.5 * torch.ones_like(sugar.surface_mesh.verts_list()[0].detach()).cpu().numpy())
    )
    n_gaussians_in_frosting = sugar._quaternions.shape[0]
    
    # Initialize the frosting
    frosting = Frosting(
        nerfmodel,
        coarse_sugar=None,
        sh_levels=nerfmodel.gaussians.max_sh_degree+1,
        shell_base_to_bind=o3d_mesh,  # Open3D mesh
        learn_shell=False,  # False
        min_frosting_factor=-0.5,  # -0.5
        max_frosting_factor=1.5,  # 1.5
        n_gaussians_in_frosting=n_gaussians_in_frosting,  # 2_000_000
        # Frosting initialization
        n_closest_gaussians_to_use_for_initializing_frosting=1,  # 16
        n_points_per_pass_for_initializing_frosting=2_000_000,  # 2_000_000
        n_samples_per_vertex_for_initializing_frosting=11,  # 51?
        frosting_level=0.01,  # 0.01
        smooth_initial_frosting=False,  # True
        n_neighbors_for_smoothing_initial_frosting=4,  # 4
        # Edition
        editable=False,
        use_softmax_for_bary_coords=True,
        min_frosting_range=0.001,
        min_frosting_size=0.001,
        initial_proposal_std_range=3.,
        final_proposal_range=3.,
        final_clamping_range=0.1,
        use_background_sphere=False,
        use_background_gaussians=False,
        avoid_self_intersections=False,
        use_constant_frosting_size=use_constant_frosting_size,
    )
    
    with torch.no_grad():
        n_gaussians_per_triangle = sugar.surface_triangle_bary_coords.shape[0]
        frosting._bary_coords[...] = sugar.surface_triangle_bary_coords[None][..., 0].repeat(
            len(sugar._surface_mesh_faces), 1, 1
            ).reshape(-1, 1, 3).repeat(1, 2, 1).reshape(-1, 6) / 2.
        
        frosting._inner_dist[...] = (1e-9) * torch.ones_like(frosting._inner_dist)
        frosting._outer_dist[...] = -(1e-9) * torch.ones_like(frosting._outer_dist)

        frosting._quaternions[...] = sugar.quaternions
        frosting._scales[...] = scale_inverse_activation(sugar.scaling)
        frosting._opacities[...] = inverse_sigmoid(sugar.strengths)
        frosting._sh_coordinates_dc[...] = sugar._sh_coordinates_dc
        frosting._sh_coordinates_rest[...] = sugar._sh_coordinates_rest
        frosting._point_cell_indices[...] = torch.arange(len(sugar._surface_mesh_faces), device=frosting.device)[:, None].repeat(1, n_gaussians_per_triangle).reshape(-1)
        
    print("Frosting loaded.")
    
    return frosting