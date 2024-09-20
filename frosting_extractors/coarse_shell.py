import os
import numpy as np
import open3d as o3d
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.ops import knn_points
from frosting_scene.gs_model import GaussianSplattingWrapper
from frosting_scene.sugar_model import SuGaR
from frosting_utils.general_utils import str2bool
from frosting_utils.spherical_harmonics import SH2RGB
from frosting_scene.frosting_model import compute_level_surface_points_and_range_from_camera

from rich.console import Console


def compute_optimal_poisson_depth(
    sugar:SuGaR,
    cell_size_nn_distance_ratio:float=100,
    max_poisson_depth:int=10,
    fg_bbox_factor=1.,
    opacity_threshold=0.5,
    quantile_to_use=0.1,
    ):
    
    # Compute foreground bbox
    _cameras_spatial_extent, _camera_average_xyz = sugar.get_cameras_spatial_extent(return_average_xyz=True)
    fg_bbox_min_tensor = _camera_average_xyz - fg_bbox_factor * _cameras_spatial_extent * torch.ones(1, 3, device=sugar.device)
    fg_bbox_max_tensor = _camera_average_xyz + fg_bbox_factor * _cameras_spatial_extent * torch.ones(1, 3, device=sugar.device)
    fg_mask = (sugar.points > fg_bbox_min_tensor).all(dim=-1) * (sugar.points < fg_bbox_max_tensor).all(dim=-1)

    # Remove transparent gaussians
    opacity_mask = sugar.strengths[..., 0] > opacity_threshold
    mask = opacity_mask * fg_mask
    
    # Compute poisson bbox
    bbox_size = 1.1 * (sugar.points[mask].max(dim=0)[0] - sugar.points[mask].min(dim=0)[0]).max().item()
    
    # Compute nearest neighbor distances and normalize by bbox size
    nn_dists = knn_points(sugar.points[mask][None], sugar.points[mask][None], K=2).dists[0, ..., 1]
    norm_nn_dists = nn_dists / bbox_size
    quantile_dist = norm_nn_dists.quantile(quantile_to_use).item()
    
    # Compute optimal poisson depth
    poisson_depth = -np.log2(cell_size_nn_distance_ratio * quantile_dist)
    poisson_depth = np.floor(poisson_depth).astype(int)
    poisson_depth = min(poisson_depth, max_poisson_depth)
    
    return poisson_depth


def extract_shell_base_from_coarse_sugar(args):
    CONSOLE = Console(width=120)
    
    all_sugar_mesh_paths = []

    # ========== Parameters ==========

    use_train_test_split = True
    n_skip_images_for_eval_split = 8
    low_opacity_gaussian_pruning_threshold = 0.5

    # Surface level extraction parameters
    n_total_points = 10_000_000
    use_gaussian_depth_for_surface_levels = False  # False until now
    surface_level_triangle_scale = 2.  # 2.
    # surface_level_triangle_scale = -2 * np.log(surface_level)
    surface_level_primitive_types = 'diamond'  # 'diamond'
    surface_level_splat_mesh = True  # True
    surface_level_n_points_in_range = 21  # 21
    surface_level_range_size = 3.0  # 3.0
    surface_level_n_points_per_pass = 2_000_000  # '2_000_000'
    surface_level_knn_to_track = 16  # 16
    flat_surface_level_normals = False  # False


    # Mesh computation parameters
    fg_bbox_factor = 1.  # 1.
    bg_bbox_factor = 4.  # 4.
    poisson_depth = args.poisson_depth  # 10 for most real scenes. 7 to 9 work well for most synthetic scenes
    vertices_density_quantile = args.cleaning_quantile  # 0.1 for most real scenes. 0. works well for most synthetic scenes
    decimate_mesh = True
    clean_mesh = True
    remove_nonvisible_connected_components = True  # ADDED
    cluster_visibility_threshold = args.connected_components_vis_th  # ADDED  -- 0.5 or 0.001
    
    project_mesh_on_surface_points = args.project_mesh_on_surface_points  # ADDED
    
    # Vanilla 3DGS data
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load
    use_train_test_split = args.eval
    
    # Coarse model path
    sugar_checkpoint_path = args.coarse_model_path
            
    # Surface levels to extract
    if args.surface_level is None:
        surface_levels = [0.1, 0.3, 0.5]
    else:
        surface_levels = [args.surface_level]
        
    # Decimation targets
    if args.decimation_target is None:
        decimation_targets = [200_000, 1_000_000]
    else:
        decimation_targets = [args.decimation_target]
    
    # Mesh output dir
    if args.mesh_output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.mesh_output_dir = os.path.join("./output/coarse_shell", args.scene_path.split("/")[-1])
        else:
            args.mesh_output_dir = os.path.join("./output/coarse_shell", args.scene_path.split("/")[-2])
    mesh_output_dir = args.mesh_output_dir
    os.makedirs(mesh_output_dir, exist_ok=True)
            
    # Bounding box
    if (args.bboxmin is None) or (args.bboxmin == 'None'):
        use_custom_bbox = False
    else:
        if (args.bboxmax is None) or (args.bboxmax == 'None'):
            raise ValueError("You need to specify both bboxmin and bboxmax.")
        use_custom_bbox = True
        
        # Parse bboxmin
        if args.bboxmin[0] == '(':
            args.bboxmin = args.bboxmin[1:]
        if args.bboxmin[-1] == ')':
            args.bboxmin = args.bboxmin[:-1]
        args.bboxmin = tuple([float(x) for x in args.bboxmin.split(",")])
        
        # Parse bboxmax
        if args.bboxmax[0] == '(':
            args.bboxmax = args.bboxmax[1:]
        if args.bboxmax[-1] == ')':
            args.bboxmax = args.bboxmax[:-1]
        args.bboxmax = tuple([float(x) for x in args.bboxmax.split(",")])
        
        fg_bbox_min = args.bboxmin
        fg_bbox_max = args.bboxmax
    center_bbox = args.center_bbox
        
    use_centers_to_extract_mesh = args.use_centers_to_extract_mesh
    use_marching_cubes = args.use_marching_cubes
    use_vanilla_3dgs = args.use_vanilla_3dgs
            
    CONSOLE.print("-----Parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("Gaussian Splatting Checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("Coarse model Checkpoint path:", sugar_checkpoint_path)
    CONSOLE.print("Shell output path:", mesh_output_dir)
    CONSOLE.print("Surface levels:", surface_levels)
    CONSOLE.print("Decimation targets:", decimation_targets)
    CONSOLE.print("Poisson depth:", poisson_depth)
    CONSOLE.print("Cleaning quantile:", vertices_density_quantile)
    CONSOLE.print("Connected components visibility threshold:", cluster_visibility_threshold)
    CONSOLE.print("Project mesh on surface points:", project_mesh_on_surface_points)
    CONSOLE.print("Use custom bbox:", use_custom_bbox)
    CONSOLE.print("Use eval split:", use_train_test_split)
    CONSOLE.print("GPU:", args.gpu)
    CONSOLE.print("Use centers to extract mesh:", use_centers_to_extract_mesh)
    CONSOLE.print("Use marching cubes:", use_marching_cubes)
    CONSOLE.print("Use vanilla 3DGS:", use_vanilla_3dgs)
    CONSOLE.print("--------------------")
    
    # Set the GPU
    torch.cuda.set_device(args.gpu)
    
    # Load the initial 3DGS model
    CONSOLE.print(f"Loading the initial 3DGS model from path {gs_checkpoint_path}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=False,
        eval_split=use_train_test_split,
        eval_split_interval=n_skip_images_for_eval_split,
        )
    
    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')
    
    # Load the coarse model
    if use_vanilla_3dgs:
        CONSOLE.print(f"\nUsing the vanilla 3DGS model for meshing...")
        with torch.no_grad():    
            print("Initializing model from trained 3DGS...")
            points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
            colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
        sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=points,
            colors=colors,
            initialize=True,
            sh_levels=nerfmodel.gaussians.active_sh_degree+1,
            keep_track_of_knn=True,
            knn_to_track=16,
            beta_mode='average',  # 'learnable', 'average', 'weighted_average'
            primitive_types='diamond',  # 'diamond', 'square'
            surface_mesh_to_bind=None,  # Open3D mesh
            )
        with torch.no_grad():
            sugar._scales[...] = nerfmodel.gaussians._scaling.detach()
            sugar._quaternions[...] = nerfmodel.gaussians._rotation.detach()
            sugar.all_densities[...] = nerfmodel.gaussians._opacity.detach()
            sugar._sh_coordinates_dc[...] = nerfmodel.gaussians._features_dc.detach()
            sugar._sh_coordinates_rest[...] = nerfmodel.gaussians._features_rest.detach()
    else:
        CONSOLE.print(f"\nLoading the coarse SuGaR model from path {sugar_checkpoint_path}...")
        checkpoint = torch.load(sugar_checkpoint_path, map_location=nerfmodel.device)
        colors = SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :])
        sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=checkpoint['state_dict']['_points'],
            colors=colors,
            initialize=True,
            sh_levels=nerfmodel.gaussians.active_sh_degree+1,
            keep_track_of_knn=True,
            knn_to_track=16,
            beta_mode='average',  # 'learnable', 'average', 'weighted_average'
            primitive_types='diamond',  # 'diamond', 'square'
            surface_mesh_to_bind=None,  # Open3D mesh
            )
        sugar.load_state_dict(checkpoint['state_dict'])
    sugar.eval()
    
    CONSOLE.print("Coarse model loaded.")
    CONSOLE.print("Coarse model parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)
        
    if poisson_depth == -1:
        CONSOLE.print("Computing optimal poisson depth...")
        poisson_depth = compute_optimal_poisson_depth(sugar)
        poisson_depth_str = 'auto'
        CONSOLE.print("Optimal poisson depth:", poisson_depth)
    else:
        poisson_depth_str = str(poisson_depth)
    
    # Pruning low opacity gaussians
    with torch.no_grad():
        CONSOLE.print("Number of gaussians:", sugar.n_points)
        CONSOLE.print("Opacities min/max/mean:", sugar.strengths.min(), sugar.strengths.max(), sugar.strengths.mean())
        n_quantiles = 10
        for i in range(n_quantiles):
            CONSOLE.print(f'Quantile {i/n_quantiles}:', sugar.strengths.quantile(i/n_quantiles).item())
            
        CONSOLE.print("\nStarting pruning low opacity gaussians...")
        sugar.drop_low_opacity_points(low_opacity_gaussian_pruning_threshold)

        CONSOLE.print("Number of gaussians left:", sugar.n_points)
        CONSOLE.print("Opacities min/max/mean:", sugar.strengths.min(), sugar.strengths.max(), sugar.strengths.mean())
        n_quantiles = 10
        for i in range(n_quantiles):
            CONSOLE.print(f'Quantile {i/n_quantiles}:', sugar.strengths.quantile(i/n_quantiles).item())
            
    # Build the triangle soup that will be used for splatting
    # sugar.primitive_types = 'square'
    sugar.primitive_types = 'diamond'
    sugar.triangle_scale = 2.
    sugar.update_texture_features()
    mesh = sugar.mesh
    
    # Create a mesh renderer
    faces_per_pixel = 10
    max_faces_per_bin = 50_000

    mesh_raster_settings = RasterizationSettings(
        image_size=(sugar.image_height, sugar.image_width),
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel,
        max_faces_per_bin=max_faces_per_bin
    )
    rasterizer = MeshRasterizer(
            cameras=nerfmodel.training_cameras.p3d_cameras[0], 
            raster_settings=mesh_raster_settings,
    )
    
    if not use_marching_cubes:
        if not use_centers_to_extract_mesh:
            # Compute surface levels point clouds
            n_pts_per_frame = int(n_total_points / len(nerfmodel.training_cameras)) + 1
            sugar.knn_to_track = surface_level_knn_to_track

            surface_levels_outputs = {}
            for surface_level in surface_levels:
                surface_levels_outputs[surface_level] = {
                    'points': torch.zeros(0, 3, device=sugar.device),
                    'inner_points': torch.zeros(0, 3, device=sugar.device),  # Added
                    'colors': torch.zeros(0, 3, device=sugar.device),
                    'view_directions': torch.zeros(0, 3, device=sugar.device),
                    'pix_to_gaussians': torch.zeros(0, dtype=torch.long, device=sugar.device),
                    'normals': torch.zeros(0, 3, device=sugar.device),
                }

            with torch.no_grad():
                cameras_to_use = nerfmodel.training_cameras
                    
                for cam_idx in range(len(nerfmodel.training_cameras)):
                    if cam_idx % 30 == 0:
                        CONSOLE.print(f"Processing frame {cam_idx}/{len(nerfmodel.training_cameras)}...")
                        for surface_level in surface_levels:
                            CONSOLE.print(f"Current point cloud for level {surface_level} has {len(surface_levels_outputs[surface_level]['points'])} points.")
                    
                    point_depth = cameras_to_use.p3d_cameras[cam_idx].get_world_to_view_transform().transform_points(sugar.points)[..., 2:].expand(-1, 3)
                    
                    # Render RGB image with Gaussian splatting
                    rgb = sugar.render_image_gaussian_rasterizer(
                        nerf_cameras=cameras_to_use, 
                        camera_indices=cam_idx,
                        bg_color = None,
                        sh_deg=0,  # nerfmodel.gaussians.active_sh_degree,
                        compute_color_in_rasterizer=True,
                        compute_covariance_in_rasterizer=True,
                        return_2d_radii=False,
                        use_same_scale_in_all_directions=False,
                    ).clamp(min=0., max=1.).contiguous()
                    
                    # Compute surface level points for the current frame
                    if cam_idx == 0:
                        sugar.reset_neighbors(knn_to_track=surface_level_knn_to_track)
                    with torch.no_grad():
                        frame_surface_level_outputs = compute_level_surface_points_and_range_from_camera(  # Added
                            sugar,
                            cam_idx=cam_idx,
                            rasterizer=rasterizer,
                            surface_levels=surface_levels, 
                            n_surface_points=2*n_pts_per_frame,  # TODO: 2*n_pts_per_frame is safe to avoid empty pixels
                            primitive_types=surface_level_primitive_types, 
                            triangle_scale=surface_level_triangle_scale,
                            splat_mesh=surface_level_splat_mesh,
                            n_points_in_range=surface_level_n_points_in_range,
                            range_size=surface_level_range_size,
                            n_points_per_pass=surface_level_n_points_per_pass,
                            density_factor=1.,
                            return_pixel_idx=True,
                            return_gaussian_idx=True,
                            return_normals=True,
                            compute_flat_normals=flat_surface_level_normals,
                            use_gaussian_depth=use_gaussian_depth_for_surface_levels,)
                        
                        
                        for surface_level in surface_levels:
                            img_surface_points = frame_surface_level_outputs[surface_level]['intersection_points']
                            img_inner_points = frame_surface_level_outputs[surface_level]['inner_intersection_points']  # Added
                            surface_gaussian_idx = frame_surface_level_outputs[surface_level]['gaussian_idx']
                            img_surface_normals = frame_surface_level_outputs[surface_level]['normals']
                            
                            pixel_idx = frame_surface_level_outputs[surface_level]['pixel_idx']
                            img_surface_colors = rgb.view(-1, 3)[pixel_idx]
                            
                            img_surface_view_directions = torch.nn.functional.normalize(cameras_to_use.p3d_cameras[cam_idx].get_camera_center() - img_surface_points)
                            img_surface_pix_to_gaussians = surface_gaussian_idx.view(-1)
                            
                            idx = torch.randperm(len(img_surface_points), device=sugar.device)[:n_pts_per_frame]
                            
                            surface_levels_outputs[surface_level]['points'] = torch.cat([surface_levels_outputs[surface_level]['points'], img_surface_points[idx]], dim=0)
                            surface_levels_outputs[surface_level]['inner_points'] = torch.cat([surface_levels_outputs[surface_level]['inner_points'], img_inner_points[idx]], dim=0)  # Added
                            surface_levels_outputs[surface_level]['colors'] = torch.cat([surface_levels_outputs[surface_level]['colors'], img_surface_colors[idx]], dim=0)
                            surface_levels_outputs[surface_level]['view_directions'] = torch.cat([surface_levels_outputs[surface_level]['view_directions'], img_surface_view_directions[idx]], dim=0)
                            surface_levels_outputs[surface_level]['pix_to_gaussians'] = torch.cat([surface_levels_outputs[surface_level]['pix_to_gaussians'], img_surface_pix_to_gaussians[idx]], dim=0)
                            surface_levels_outputs[surface_level]['normals'] = torch.cat([surface_levels_outputs[surface_level]['normals'], img_surface_normals[idx]], dim=0)

            # -----Processing surface levels-----
            for surface_level in surface_levels:
                CONSOLE.print("\n========== Processing surface level", surface_level, "==========")
                CONSOLE.print(f"Final point cloud for level {surface_level} has {len(surface_levels_outputs[surface_level]['points'])} points.")
                surface_points = surface_levels_outputs[surface_level]['points']
                surface_inner_points = surface_levels_outputs[surface_level]['inner_points']  # Added
                surface_colors = surface_levels_outputs[surface_level]['colors']
                surface_normals = surface_levels_outputs[surface_level]['normals']
                # ---Added: Use distance to inner points as colors---
                surface_colors = (surface_inner_points - surface_points).norm(dim=-1, keepdim=True)
                max_surface_color = surface_colors.max().item()
                min_surface_color = surface_colors.min().item()
                surface_colors = (surface_colors - min_surface_color) / (max_surface_color - min_surface_color)
                surface_colors = surface_colors.repeat(1, 3)
                # ---End of added---

                if use_custom_bbox:
                    CONSOLE.print("Using provided bounding box.")
                    fg_bbox_min_tensor = torch.tensor(fg_bbox_min).to(sugar.device)
                    fg_bbox_max_tensor = torch.tensor(fg_bbox_max).to(sugar.device)
                else:
                    CONSOLE.print("Using default, camera based bounding box.")
                    fg_bbox_min_tensor = - fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)
                    fg_bbox_max_tensor = fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)
                    
                if center_bbox:
                    _cameras_spatial_extent, _camera_average_xyz = sugar.get_cameras_spatial_extent(return_average_xyz=True)
                    with torch.no_grad():
                        CONSOLE.print("Centering bounding box.")
                        fg_bbox_min_tensor = fg_bbox_min_tensor + _camera_average_xyz
                        fg_bbox_max_tensor = fg_bbox_max_tensor + _camera_average_xyz

                points_idx = torch.arange(len(surface_points))
                
                fg_mask = (surface_points[points_idx] > fg_bbox_min_tensor).all(dim=-1) * (surface_points[points_idx] < fg_bbox_max_tensor).all(dim=-1)
                if center_bbox:
                    bg_mask = ((surface_points[points_idx] - _camera_average_xyz).abs().max(dim=-1)[0]
                               < bg_bbox_factor * _cameras_spatial_extent) * ~fg_mask
                else:
                    bg_mask = (surface_points[points_idx].abs().max(dim=-1)[0] < bg_bbox_factor * sugar.get_cameras_spatial_extent()) * ~fg_mask

                fg_points = surface_points[points_idx][fg_mask]
                fg_inner_points = surface_inner_points[points_idx][fg_mask]  # Added
                fg_colors = surface_colors[points_idx][fg_mask]
                fg_normals = surface_normals[points_idx][fg_mask]

                bg_points = surface_points[points_idx][bg_mask]
                bg_inner_points = surface_inner_points[points_idx][bg_mask]  # Added
                bg_colors = surface_colors[points_idx][bg_mask]
                bg_normals = surface_normals[points_idx][bg_mask]

                CONSOLE.print("Foreground points:", fg_points.shape, fg_colors.shape, fg_normals.shape)
                CONSOLE.print("Background points:", bg_points.shape, bg_colors.shape, bg_normals.shape)
                
                # ---Compute foreground mesh---
                CONSOLE.print("\n-----Foreground mesh-----")
                if fg_points.shape[0] > 0:
                    CONSOLE.print("Computing points, colors and normals...")
                    fg_pcd = o3d.geometry.PointCloud()
                    fg_pcd.points = o3d.utility.Vector3dVector(fg_points.double().cpu().numpy())
                    fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors.double().cpu().numpy())
                    fg_pcd.normals = o3d.utility.Vector3dVector(fg_normals.double().cpu().numpy())

                    # outliers removal
                    cl, ind = fg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
                    CONSOLE.print("Cleaning Point Cloud...")
                    fg_pcd = fg_pcd.select_by_index(ind)

                    CONSOLE.print("Finished computing points, colors and normals.")

                    CONSOLE.print("Now computing mesh...")
                    o3d_fg_mesh, o3d_fg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        fg_pcd, depth=poisson_depth) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

                    if vertices_density_quantile > 0.:
                        CONSOLE.print("Removing vertices with low densities...")
                        vertices_to_remove = o3d_fg_densities < np.quantile(o3d_fg_densities, vertices_density_quantile)
                        o3d_fg_mesh.remove_vertices_by_mask(vertices_to_remove)
                    
                else:
                    CONSOLE.print("\n[WARNING] Foreground is empty.")
                    o3d_fg_mesh = None
                
                # ---Compute background mesh---
                CONSOLE.print("\n-----Background mesh-----")
                if bg_points.shape[0] > 0:
                    CONSOLE.print("Computing points, colors and normals...")
                    bg_pcd = o3d.geometry.PointCloud()
                    bg_pcd.points = o3d.utility.Vector3dVector(bg_points.double().cpu().numpy())
                    bg_pcd.colors = o3d.utility.Vector3dVector(bg_colors.double().cpu().numpy())
                    bg_pcd.normals = o3d.utility.Vector3dVector(bg_normals.double().cpu().numpy())

                    # outliers removal
                    cl, ind = bg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
                    CONSOLE.print("Cleaning Point Cloud...")
                    bg_pcd = bg_pcd.select_by_index(ind)

                    CONSOLE.print("Finished computing points, colors and normals.")

                    CONSOLE.print("Now computing mesh...")
                    o3d_bg_mesh, o3d_bg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        bg_pcd, depth=poisson_depth) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

                    if vertices_density_quantile > 0.:
                        CONSOLE.print("Removing vertices with low densities...")
                        vertices_to_remove = o3d_bg_densities < np.quantile(o3d_bg_densities, vertices_density_quantile)
                        o3d_bg_mesh.remove_vertices_by_mask(vertices_to_remove)
                else:
                    CONSOLE.print("\n[WARNING] Background is empty.")
                    o3d_bg_mesh = None
                
                CONSOLE.print("Finished computing meshes.")
                CONSOLE.print("Foreground mesh:", o3d_fg_mesh)
                CONSOLE.print("Background mesh:", o3d_bg_mesh)
                
                # ---Decimate and clean meshes---
                CONSOLE.print("\n-----Decimating and cleaning meshes-----")
                for decimation_target in decimation_targets:
                    CONSOLE.print("\nProcessing decimation target:", decimation_target)
                    if decimate_mesh:
                        if o3d_fg_mesh is not None:
                            CONSOLE.print("Decimating foreground mesh...")
                            decimated_o3d_fg_mesh = o3d_fg_mesh.simplify_quadric_decimation(decimation_target)
                            CONSOLE.print("Finished decimating foreground mesh.")
                        else:
                            decimated_o3d_fg_mesh = None
                            
                        if o3d_bg_mesh is not None:                            
                            CONSOLE.print("Decimating background mesh...")
                            decimated_o3d_bg_mesh = o3d_bg_mesh.simplify_quadric_decimation(decimation_target)
                            CONSOLE.print("Finished decimating background mesh.")
                        else:
                            decimated_o3d_bg_mesh = None

                    if clean_mesh:
                        CONSOLE.print("Cleaning mesh...")
                        if decimated_o3d_fg_mesh is not None:
                            decimated_o3d_fg_mesh.remove_duplicated_vertices()
                            decimated_o3d_fg_mesh.remove_degenerate_triangles()
                            decimated_o3d_fg_mesh.remove_duplicated_triangles()
                            decimated_o3d_fg_mesh.remove_non_manifold_edges()
                        
                        if decimated_o3d_bg_mesh is not None:
                            decimated_o3d_bg_mesh.remove_duplicated_vertices()
                            decimated_o3d_bg_mesh.remove_degenerate_triangles()
                            decimated_o3d_bg_mesh.remove_duplicated_triangles()
                            decimated_o3d_bg_mesh.remove_non_manifold_edges()
                    
                    if (decimated_o3d_fg_mesh is not None) and (decimated_o3d_bg_mesh is not None):
                        CONSOLE.print("Merging foreground and background meshes.")
                        decimated_o3d_mesh = decimated_o3d_fg_mesh + decimated_o3d_bg_mesh
                    elif decimated_o3d_fg_mesh is not None:
                        CONSOLE.print("Using foreground mesh only, since background mesh is empty.")
                        decimated_o3d_mesh = decimated_o3d_fg_mesh
                    elif decimated_o3d_bg_mesh is not None:
                        CONSOLE.print("Using background mesh only, since foreground mesh is empty.")
                        decimated_o3d_mesh = decimated_o3d_bg_mesh
                    else:
                        raise ValueError("Both foreground and background meshes are empty. Please provide a valid bounding box for the scene.")
                    
                    # Project Mesh on extracted surface points for better details
                    if project_mesh_on_surface_points:
                        CONSOLE.print("Projecting mesh on surface points to recover better details...")
                        mesh_verts = torch.tensor(np.asarray(decimated_o3d_mesh.vertices), device=sugar.device, dtype=torch.float32)
                        mesh_faces = torch.tensor(np.asarray(decimated_o3d_mesh.triangles), device=sugar.device, dtype=torch.int64)

                        proj_knn_idx = knn_points(
                            mesh_verts[None], 
                            surface_points[None], 
                            K=1,
                        ).idx[0][..., 0]
                        
                        new_mesh_verts = surface_points[proj_knn_idx]
                        new_mesh_faces = mesh_faces
                        new_mesh_colors = surface_colors[proj_knn_idx]
                        
                        decimated_o3d_mesh = o3d.geometry.TriangleMesh()
                        decimated_o3d_mesh.vertices = o3d.utility.Vector3dVector(new_mesh_verts.cpu().numpy())
                        decimated_o3d_mesh.triangles = o3d.utility.Vector3iVector(new_mesh_faces.cpu().numpy())
                        decimated_o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(new_mesh_colors.cpu().numpy())
                        decimated_o3d_mesh.compute_vertex_normals()
                        
                        # ADDED
                        CONSOLE.print("Cleaning projected mesh...")
                        decimated_o3d_mesh.remove_duplicated_vertices()
                        decimated_o3d_mesh.remove_degenerate_triangles()
                        decimated_o3d_mesh.remove_duplicated_triangles()
                        decimated_o3d_mesh.remove_non_manifold_edges()
                        CONSOLE.print("Projection done.")
                    
                    # Denormalize colors
                    final_colors = np.asarray(decimated_o3d_mesh.vertex_colors)
                    final_colors = final_colors * (max_surface_color - min_surface_color) + min_surface_color
                    
                    # Renormalize colors based on the size of the bounding box of mesh vertices
                    verts_bbox_diag = np.linalg.norm(decimated_o3d_mesh.get_max_bound() - decimated_o3d_mesh.get_min_bound())
                    final_colors = final_colors / verts_bbox_diag
                    decimated_o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(final_colors))
                    
                    # Remove non-visible connected components
                    if remove_nonvisible_connected_components:
                        CONSOLE.print("Removing non-visible connected components. \nIterating over all training viewpoints...")
                        # Create a PyTorch3D mesh
                        p3d_mesh = Meshes(
                            verts=[torch.tensor(decimated_o3d_mesh.vertices).float().to(sugar.device)],
                            faces=[torch.tensor(decimated_o3d_mesh.triangles).long().to(sugar.device)],
                        )
                        face_visible_count = torch.zeros(len(p3d_mesh.faces_list()[0]), dtype=torch.long, device=sugar.device)
                        
                        # Create a new PyTorch3D mesh rasterizer
                        faces_per_pixel = 1
                        max_faces_per_bin = 50_000
                        mesh_raster_settings = RasterizationSettings(
                            image_size=(sugar.image_height, sugar.image_width),
                            blur_radius=0.0, 
                            faces_per_pixel=faces_per_pixel,
                            # max_faces_per_bin=max_faces_per_bin
                        )
                        rasterizer = MeshRasterizer(
                            cameras=nerfmodel.training_cameras.p3d_cameras[0], 
                            raster_settings=mesh_raster_settings,
                        )
                        
                        # Iterate over all training viewpoints and count the number of times each face is visible
                        for cam_idx in range(len(nerfmodel.training_cameras)):
                            if cam_idx % 30 == 0:
                                CONSOLE.print(f"Processing frame {cam_idx}/{len(nerfmodel.training_cameras)}...")
                            p3d_cameras = nerfmodel.training_cameras.p3d_cameras[cam_idx]
                            fragments = rasterizer(p3d_mesh, cameras=p3d_cameras)
                            pix_to_face = fragments.pix_to_face[0, ..., 0]
                            frame_faces = pix_to_face[pix_to_face > -1].unique()
                            face_visible_count[frame_faces] = face_visible_count[frame_faces] + 1
                        
                        # Compute connected components
                        face_visible_count = face_visible_count.clamp(max=1)
                        face_clusters, cluster_n_faces, _ = decimated_o3d_mesh.cluster_connected_triangles()
                        face_clusters = torch.tensor(np.asarray(face_clusters), dtype=torch.long, device=sugar.device)
                        cluster_n_faces = torch.tensor(np.asarray(cluster_n_faces), dtype=torch.long, device=sugar.device)
                        
                        # Identify visible faces in connected components
                        visible_clusters, n_visible_faces_per_visible_cluster = (face_visible_count * (face_clusters+1)).unique(return_counts=True)
                        visible_clusters = visible_clusters - 1
                        n_visible_faces_per_visible_cluster = n_visible_faces_per_visible_cluster[visible_clusters > -1]
                        visible_clusters = visible_clusters[visible_clusters > -1]
                        cluster_n_visible_faces = torch.zeros_like(cluster_n_faces)
                        cluster_n_visible_faces[visible_clusters] = n_visible_faces_per_visible_cluster
                        cluster_visibility_ratio = cluster_n_visible_faces.float() / cluster_n_faces.float()
                        
                        # Remove non-visible connected components
                        triangles_to_remove = np.array((cluster_visibility_ratio[face_clusters] < cluster_visibility_threshold).cpu())
                        decimated_o3d_mesh.remove_triangles_by_mask(triangles_to_remove)
                        decimated_o3d_mesh.remove_unreferenced_vertices()
                    
                    if use_vanilla_3dgs:
                        frosting_mesh_path = 'frostingshellbase_vanilla3dgs_levelZZ_decimAA_depthBB_quantileCC.ply'
                    else:
                        frosting_mesh_path = 'frostingshellbase_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + '_levelZZ_decimAA_depthBB_quantileCC.ply'
                    frosting_mesh_path = frosting_mesh_path.replace(
                        'ZZ', str(surface_level).replace('.', '')
                        ).replace(
                            'AA', str(decimation_target).replace('.', '')
                            ).replace(
                                'BB', poisson_depth_str.replace('.', '')
                                ).replace(
                                    'CC', str(vertices_density_quantile).replace('.', '')
                                    )
                    frosting_mesh_path = os.path.join(mesh_output_dir, frosting_mesh_path)
                    o3d.io.write_triangle_mesh(frosting_mesh_path, decimated_o3d_mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                    CONSOLE.print("Mesh saved at", frosting_mesh_path)
                    all_sugar_mesh_paths.append(frosting_mesh_path)
                    
        else:
            CONSOLE.print("\nWARNING: Using centers of gaussians to extract mesh.")
            CONSOLE.print("Results will look bad, this is not the best way to extract a mesh.")
            CONSOLE.print("You should use this option only for ablation.")
            
            with torch.no_grad():
                surface_points = sugar.points
                surface_colors = SH2RGB(sugar._sh_coordinates_dc[:, 0, :])
                surface_normals = sugar.get_normals(estimate_from_points=True)
                
                if use_custom_bbox:
                    CONSOLE.print("Using provided bounding box.")
                    fg_bbox_min_tensor = torch.tensor(fg_bbox_min).to(sugar.device)
                    fg_bbox_max_tensor = torch.tensor(fg_bbox_max).to(sugar.device)
                else:
                    CONSOLE.print("Using default, camera based bounding box.")
                    fg_bbox_min_tensor = - fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)
                    fg_bbox_max_tensor = fg_bbox_factor * sugar.get_cameras_spatial_extent() * torch.ones(1, 3, device=sugar.device)

                points_idx = torch.arange(len(surface_points))
                fg_mask = (surface_points[points_idx] > fg_bbox_min_tensor).all(dim=-1) * (surface_points[points_idx] < fg_bbox_max_tensor).all(dim=-1)
                bg_mask = (surface_points[points_idx].abs().max(dim=-1)[0] < bg_bbox_factor * sugar.get_cameras_spatial_extent()) * ~fg_mask

                fg_points = surface_points[points_idx][fg_mask]
                fg_colors = surface_colors[points_idx][fg_mask]
                fg_normals = surface_normals[points_idx][fg_mask]

                bg_points = surface_points[points_idx][bg_mask]
                bg_colors = surface_colors[points_idx][bg_mask]
                bg_normals = surface_normals[points_idx][bg_mask]

                CONSOLE.print("Foreground points:", fg_points.shape, fg_colors.shape, fg_normals.shape)
                CONSOLE.print("Background points:", bg_points.shape, bg_colors.shape, bg_normals.shape)
                
                # ---Compute foreground mesh---
                CONSOLE.print("\n-----Foreground mesh-----")
                CONSOLE.print("Computing points, colors and normals...")
                fg_pcd = o3d.geometry.PointCloud()
                fg_pcd.points = o3d.utility.Vector3dVector(fg_points.double().cpu().numpy())
                fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors.double().cpu().numpy())
                fg_pcd.normals = o3d.utility.Vector3dVector(fg_normals.double().cpu().numpy())

                # outliers removal
                cl, ind = fg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
                CONSOLE.print("Cleaning Point Cloud...")
                fg_pcd = fg_pcd.select_by_index(ind)

                CONSOLE.print("Finished computing points, colors and normals.")

                CONSOLE.print("Now computing mesh...")
                o3d_fg_mesh, o3d_fg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    fg_pcd, depth=poisson_depth) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

                if vertices_density_quantile > 0.:
                    CONSOLE.print("Removing vertices with low densities...")
                    vertices_to_remove = o3d_fg_densities < np.quantile(o3d_fg_densities, vertices_density_quantile)
                    o3d_fg_mesh.remove_vertices_by_mask(vertices_to_remove)
                
                # ---Compute background mesh---
                if bg_points.shape[0] > 0:
                    CONSOLE.print("\n-----Background mesh-----")
                    CONSOLE.print("Computing points, colors and normals...")
                    bg_pcd = o3d.geometry.PointCloud()
                    bg_pcd.points = o3d.utility.Vector3dVector(bg_points.double().cpu().numpy())
                    bg_pcd.colors = o3d.utility.Vector3dVector(bg_colors.double().cpu().numpy())
                    bg_pcd.normals = o3d.utility.Vector3dVector(bg_normals.double().cpu().numpy())

                    # outliers removal
                    cl, ind = bg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.)
                    CONSOLE.print("Cleaning Point Cloud...")
                    bg_pcd = bg_pcd.select_by_index(ind)

                    CONSOLE.print("Finished computing points, colors and normals.")

                    CONSOLE.print("Now computing mesh...")
                    o3d_bg_mesh, o3d_bg_densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        bg_pcd, depth=poisson_depth) #, width=0, scale=1.1, linear_fit=False)  # depth=10 should be the default value? 11 is good to (but it starts to make a big number of triangles)

                    if vertices_density_quantile > 0.:
                        CONSOLE.print("Removing vertices with low densities...")    
                        vertices_to_remove = o3d_bg_densities < np.quantile(o3d_bg_densities, vertices_density_quantile)
                        o3d_bg_mesh.remove_vertices_by_mask(vertices_to_remove)
                else:
                    o3d_bg_mesh = None
                
                CONSOLE.print("Finished computing meshes.")
                CONSOLE.print("Foreground mesh:", o3d_fg_mesh)
                CONSOLE.print("Background mesh:", o3d_bg_mesh)
                
                # ---Decimate and clean meshes---
                CONSOLE.print("\n-----Decimating and cleaning meshes-----")
                for decimation_target in decimation_targets:
                    CONSOLE.print("\nProcessing decimation target:", decimation_target)
                    if decimate_mesh:
                        CONSOLE.print("Decimating foreground mesh...")
                        decimated_o3d_fg_mesh = o3d_fg_mesh.simplify_quadric_decimation(decimation_target)
                        CONSOLE.print("Finished decimating foreground mesh.")
                        
                        if o3d_bg_mesh is not None:                                                        
                            CONSOLE.print("Decimating background mesh...")
                            decimated_o3d_bg_mesh = o3d_bg_mesh.simplify_quadric_decimation(decimation_target)
                            CONSOLE.print("Finished decimating background mesh.")

                    if clean_mesh:
                        CONSOLE.print("Cleaning mesh...")
                        decimated_o3d_fg_mesh.remove_degenerate_triangles()
                        decimated_o3d_fg_mesh.remove_duplicated_triangles()
                        decimated_o3d_fg_mesh.remove_duplicated_vertices()
                        decimated_o3d_fg_mesh.remove_non_manifold_edges()
                        
                        if decimated_o3d_bg_mesh is not None:
                            decimated_o3d_bg_mesh.remove_degenerate_triangles()
                            decimated_o3d_bg_mesh.remove_duplicated_triangles()
                            decimated_o3d_bg_mesh.remove_duplicated_vertices()
                            decimated_o3d_bg_mesh.remove_non_manifold_edges()
                        
                    if decimated_o3d_bg_mesh is not None:
                        decimated_o3d_mesh = decimated_o3d_fg_mesh + decimated_o3d_bg_mesh
                    else:
                        decimated_o3d_mesh = decimated_o3d_fg_mesh
                    
                    if remove_nonvisible_connected_components:
                        # TODO
                        pass
                    
                    if use_vanilla_3dgs:
                        sugar_mesh_path = 'sugarmesh_vanilla3dgs_poissoncenters_decimAA.ply'
                    else:
                        sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + '_poissoncenters_decimAA.ply'
                    sugar_mesh_path = sugar_mesh_path.replace(
                            'AA', str(decimation_target).replace('.', '')
                            )
                    sugar_mesh_path = os.path.join(mesh_output_dir, sugar_mesh_path)
                    o3d.io.write_triangle_mesh(sugar_mesh_path, decimated_o3d_mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                    CONSOLE.print("Mesh saved at", sugar_mesh_path)
                    all_sugar_mesh_paths.append(sugar_mesh_path)
    else:
        CONSOLE.print("\nWARNING: Using marching cubes to extract mesh.")
        import mcubes
        
        sugar.reset_neighbors(knn_to_track=16)
        
        resolution = 512
        surface_level = surface_levels[0]
        decimation_target = decimation_targets[0]

        # Foreground mesh
        CONSOLE.print("\n-----Foreground mesh-----")
        X = torch.linspace(-1, 1, resolution) * sugar.get_cameras_spatial_extent()
        Y = torch.linspace(-1, 1, resolution) * sugar.get_cameras_spatial_extent()
        Z = torch.linspace(-1, 1, resolution) * sugar.get_cameras_spatial_extent()

        xx, yy, zz = torch.meshgrid(X, Y, Z)
        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(sugar.device)

        xx.shape, yy.shape, zz.shape, pts.shape

        n_pts_per_pass = 2_000_000

        densities = torch.zeros(0, device=sugar.device)

        CONSOLE.print("Computing densities...")
        with torch.no_grad():
            for i in range(0, len(pts), n_pts_per_pass):
                print("\nPts:", i, 'to', i+n_pts_per_pass)
                pts_i = pts[i:i+n_pts_per_pass]
                densities_i = sugar.compute_density(pts_i)
                densities = torch.cat([densities, densities_i], dim=0)
            densities = densities.reshape(resolution, resolution, resolution)
            CONSOLE.print("Finished computing densities.")
            
            density_th = surface_levels[0]  # 1.
            CONSOLE.print(f"Computing mesh for surface level {density_th}...")
            vertices, triangles = mcubes.marching_cubes(densities.cpu().numpy(), density_th)
            verts = -sugar.get_cameras_spatial_extent() + (torch.tensor(vertices) / resolution) * 2 * sugar.get_cameras_spatial_extent()
            faces = torch.tensor(triangles.tolist())
            closest_gaussians = sugar.get_gaussians_closest_to_samples(verts.float().to(sugar.device))
            verts_colors = SH2RGB(sugar._sh_coordinates_dc[closest_gaussians[:, 0]][:, 0, :])
            
            mc_mesh = o3d.geometry.TriangleMesh()
            mc_mesh.vertices = o3d.utility.Vector3dVector(verts.cpu().numpy())
            mc_mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            mc_mesh.vertex_colors = o3d.utility.Vector3dVector(verts_colors.cpu().numpy())
            mc_mesh.compute_vertex_normals()
            CONSOLE.print("Finished computing mesh.")
         
        # Background mesh   
        CONSOLE.print("\n-----Background mesh-----")
        X = torch.linspace(-1, 1, resolution) * 4 * sugar.get_cameras_spatial_extent()
        Y = torch.linspace(-1, 1, resolution) * 4 * sugar.get_cameras_spatial_extent()
        Z = torch.linspace(-1, 1, resolution) * 4 * sugar.get_cameras_spatial_extent()

        xx, yy, zz = torch.meshgrid(X, Y, Z)
        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(sugar.device)

        xx.shape, yy.shape, zz.shape, pts.shape

        n_pts_per_pass = 2_000_000

        densities = torch.zeros(0, device=sugar.device)

        CONSOLE.print("Computing densities...")
        with torch.no_grad():
            for i in range(0, len(pts), n_pts_per_pass):
                print("\nPts:", i, 'to', i+n_pts_per_pass)
                pts_i = pts[i:i+n_pts_per_pass]
                densities_i = sugar.compute_density(pts_i)
                densities = torch.cat([densities, densities_i], dim=0)
            CONSOLE.print("Finished computing densities.")
            
            # Removing pts in foreground
            densities[(pts > -sugar.get_cameras_spatial_extent()).all(dim=-1) * (pts < sugar.get_cameras_spatial_extent()).all(dim=-1)] = 0.
            densities = densities.reshape(resolution, resolution, resolution)

            density_th = surface_levels[0]  # 1.
            CONSOLE.print(f"Computing mesh for surface level {density_th}...")
            bg_vertices, bg_triangles = mcubes.marching_cubes(densities.cpu().numpy(), density_th)
            bg_verts = - 4 * sugar.get_cameras_spatial_extent() + (torch.tensor(bg_vertices) / resolution) * 2 * 4 * sugar.get_cameras_spatial_extent()
            bg_faces = torch.tensor(bg_triangles.tolist())
            closest_gaussians = sugar.get_gaussians_closest_to_samples(bg_verts.float().to(sugar.device))
            bg_verts_colors = SH2RGB(sugar._sh_coordinates_dc[closest_gaussians[:, 0]][:, 0, :])
            
            bg_mc_mesh = o3d.geometry.TriangleMesh()
            bg_mc_mesh.vertices = o3d.utility.Vector3dVector(bg_verts.cpu().numpy())
            bg_mc_mesh.triangles = o3d.utility.Vector3iVector(bg_faces.cpu().numpy())
            bg_mc_mesh.vertex_colors = o3d.utility.Vector3dVector(bg_verts_colors.cpu().numpy())
            bg_mc_mesh.compute_vertex_normals()
            CONSOLE.print("Finished computing mesh.")
            
        # Decimate and clean meshes
        decimate_mesh = True
        decimation_target = decimation_targets[0]

        if decimate_mesh:
            print(f"Decimating mesh to target {decimation_target}...")
            decimated_o3d_fg_mesh = mc_mesh.simplify_quadric_decimation(decimation_target)
            print("Finished decimating mesh.")
            
            print("Decimating mesh...")
            decimated_o3d_bg_mesh = bg_mc_mesh.simplify_quadric_decimation(decimation_target)
            print("Finished decimating mesh.")
        else:
            decimated_o3d_fg_mesh = mc_mesh
            decimated_o3d_bg_mesh = bg_mc_mesh
            
        clean_mesh = True
        if clean_mesh:
            decimated_o3d_fg_mesh.remove_degenerate_triangles()
            decimated_o3d_fg_mesh.remove_duplicated_triangles()
            decimated_o3d_fg_mesh.remove_duplicated_vertices()
            decimated_o3d_fg_mesh.remove_non_manifold_edges()
            
            decimated_o3d_bg_mesh.remove_degenerate_triangles()
            decimated_o3d_bg_mesh.remove_duplicated_triangles()
            decimated_o3d_bg_mesh.remove_duplicated_vertices()
            decimated_o3d_bg_mesh.remove_non_manifold_edges()
            
        decimated_o3d_mesh = decimated_o3d_fg_mesh + decimated_o3d_bg_mesh
        if use_vanilla_3dgs:
            sugar_mesh_path = 'sugarmesh_vanilla3dgsmarchingcubes_levelZZ_decimAA.ply'
        else:
            sugar_mesh_path = 'sugarmesh_' + sugar_checkpoint_path.split('/')[-2].replace('sugarcoarse_', '') + 'marchingcubes_levelZZ_decimAA.ply'
        sugar_mesh_path = sugar_mesh_path.replace(
            'ZZ', str(surface_level).replace('.', '')
            ).replace(
                'AA', str(decimation_target).replace('.', '')
                )
        sugar_mesh_path = os.path.join(mesh_output_dir, sugar_mesh_path)
        o3d.io.write_triangle_mesh(sugar_mesh_path, decimated_o3d_mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
        CONSOLE.print("Mesh saved at", sugar_mesh_path)
        all_sugar_mesh_paths.append(sugar_mesh_path)
        
    return all_sugar_mesh_paths
