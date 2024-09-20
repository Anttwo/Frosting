import os
import numpy as np
import torch
import open3d as o3d
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from pytorch3d.io import save_obj
from frosting_scene.gs_model import GaussianSplattingWrapper, fetchPly
from frosting_scene.sugar_model import SuGaR, SH2RGB
from frosting_scene.frosting_model import Frosting, convert_frosting_into_gaussians, rasterization_with_pix_to_face
from frosting_scene.frosting_optimizer import OptimizationParams, FrostingOptimizer
from frosting_utils.loss_utils import ssim, l1_loss, l2_loss
from frosting_utils.mesh_rasterization import RasterizationSettings, MeshRasterizer
from frosting_utils.texture import compute_textured_mesh_for_frosting_mesh

from rich.console import Console
import time
import gc


def refined_training(args):
    CONSOLE = Console(width=120)

    # ====================Parameters====================

    num_device = args.gpu
    detect_anomaly = False

    # -----Data parameters-----
    downscale_resolution_factor = 1  # 2, 4

    # -----Model parameters-----
    use_eval_split = True
    n_skip_images_for_eval_split = 8

    n_points_at_start = None  # If None, takes all points in the SfM point cloud
    sh_levels = 4  
    
        
    # -----Rendering parameters-----
    compute_color_in_rasterizer = True  # TODO: Try True

    # TODO: New feature
    # use_occlusion_culling = True
    occlusion_culling_type = 'pix_to_face'  # Can be 'pix_to_face' or 'depth'
        
    # -----Optimization parameters-----

    # Learning rates and scheduling
    num_iterations = 15_000  # Changed

    spatial_lr_scale = None
    position_lr_init=0.00016
    position_lr_final=0.0000016
    position_lr_delay_mult=0.01
    position_lr_max_steps=30_000
    feature_lr=0.0025
    opacity_lr=0.05
    scaling_lr=0.005
    rotation_lr=0.001
    # Specific to Frosting. TODO: Adjust
    position_bary_coords_lr_init = 0.005
    position_bary_coords_lr_final = 0.00005

    # Data processing and batching
    n_images_to_use_for_training = -1  # If -1, uses all images
    train_num_images_per_batch = 1  # 1 for full images

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    if loss_function == 'l1+dssim':
        dssim_factor = 0.2

    # Warmup
    do_resolution_warmup = False
    if do_resolution_warmup:
        resolution_warmup_every = 500
        current_resolution_factor = downscale_resolution_factor * 4.
    else:
        current_resolution_factor = downscale_resolution_factor

    do_sh_warmup = False  # Was True for SuGaR, should be False here
    if do_sh_warmup:
        sh_warmup_every = 1000
        current_sh_levels = 1
    else:
        current_sh_levels = sh_levels
        

    # -----Log and save-----
    print_loss_every_n_iterations = 200
    save_model_every_n_iterations = 1_000_000 # 500, 1_000_000  # TODO
    # save_milestones = [2000, 7_000, 15_000]
    save_milestones = []

    # ====================End of parameters====================

    if args.output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.output_dir = os.path.join("./output/refined_frosting", args.scene_path.split("/")[-1])
        else:
            args.output_dir = os.path.join("./output/refined_frosting", args.scene_path.split("/")[-2])
            
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
    
    # Data parameters
    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    sugar_model_path = args.sugar_path
    shell_base_to_bind_path = args.mesh_path
    mesh_name = shell_base_to_bind_path.split("/")[-1].split(".")[0]
    iteration_to_load = args.iteration_to_load   
    
    # Render parameters
    use_occlusion_culling = args.use_occlusion_culling
    use_occlusion_culling_every_n_iterations = 2
    
    # Frosting parameters
    min_frosting_factor = args.min_frosting_factor  # Deprecated
    max_frosting_factor = args.max_frosting_factor  # Deprecated
    min_frosting_range = args.min_frosting_range  # Deprecated
    
    learn_shell = args.learn_shell
    use_surface_mesh_normal_consistency_loss = args.regularize_shell
    n_gaussians_in_frosting = args.gaussians_in_frosting    
    n_samples_per_vertex_for_initializing_frosting = args.n_samples_per_vertex
    frosting_level = args.frosting_level
    smooth_initial_frosting = args.smooth_initial_frosting
    n_neighbors_for_smoothing_initial_frosting = args.n_neighbors_for_smoothing
    min_frosting_size = args.min_frosting_size
    initial_proposal_std_range = args.initial_proposal_std_range
    final_proposal_range = args.final_proposal_range
    final_clamping_range = args.final_clamping_range
    
    use_background_sphere = args.use_background_sphere
    use_background_gaussians = args.use_background_gaussians
    
    # Optimization parameters
    if use_surface_mesh_normal_consistency_loss:
        print("WARNING: Using surface mesh normal consistency loss.")
        surface_mesh_normal_consistency_factor = args.normal_consistency_factor 
    # n_vertices_in_fg = args.n_vertices_in_fg
    num_iterations = args.refinement_iterations

    name_base = 'frostingfine_'
    if learn_shell:
        name_base += 'learnshell_'
    if use_surface_mesh_normal_consistency_loss:
        frosting_checkpoint_path = name_base + mesh_name.replace('frostingshellbase_', '') + '_normalconsistencyXX_gaussYY_frostlevelAA_proposalBB/'
    else:
        frosting_checkpoint_path = name_base + mesh_name.replace('frostingshellbase_', '') + '_gaussYY_frostlevelAA_proposalBB/'
    frosting_checkpoint_path = os.path.join(args.output_dir, frosting_checkpoint_path)
    if use_surface_mesh_normal_consistency_loss:
        frosting_checkpoint_path = frosting_checkpoint_path.replace(
            'XX', str(surface_mesh_normal_consistency_factor).replace('.', '')
            ).replace(
            'YY', str(n_gaussians_in_frosting).replace('.', '')
            ).replace(
                'AA', str(frosting_level).replace('.', '')
            ).replace(
                'BB', str(final_proposal_range).replace('.', '')
            )
    else:
        frosting_checkpoint_path = frosting_checkpoint_path.replace(
            'YY', str(n_gaussians_in_frosting).replace('.', '')
            ).replace(
                'AA', str(frosting_level).replace('.', '')
            ).replace(
                'BB', str(final_proposal_range).replace('.', '')
            )
        
    if use_custom_bbox:
        fg_bbox_min = args.bboxmin
        fg_bbox_max = args.bboxmax
    
    use_eval_split = args.eval
    use_white_background = args.white_background
    
    export_ply_at_the_end = args.export_ply
    export_obj_at_the_end = args.export_obj
    obj_texture_square_size = args.texture_square_size
    
    ply_path = os.path.join(source_path, "sparse/0/points3D.ply")
    
    CONSOLE.print("-----Parsed parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("   > Content:", len(os.listdir(source_path)))
    CONSOLE.print("Gaussian Splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("   > Content:", len(os.listdir(gs_checkpoint_path)))
    CONSOLE.print("Sugar model path:", sugar_model_path)
    CONSOLE.print("Frosting checkpoint path:", frosting_checkpoint_path)
    CONSOLE.print("Shell base mesh to bind to:", shell_base_to_bind_path)
    CONSOLE.print("Iteration to load:", iteration_to_load)
    CONSOLE.print("Use occlusion culling:", use_occlusion_culling)
    if use_occlusion_culling:
        CONSOLE.print("Occlusion culling type:", occlusion_culling_type)
    CONSOLE.print("Shell parameters:")
    CONSOLE.print("   > Learn shell:", learn_shell)
    CONSOLE.print("   > Number of gaussians in frosting:", n_gaussians_in_frosting)
    CONSOLE.print("   > Number of samples per vertex for initializing frosting:", n_samples_per_vertex_for_initializing_frosting)
    CONSOLE.print("   > Frosting level:", frosting_level)
    CONSOLE.print("   > Smooth initial frosting:", smooth_initial_frosting)
    CONSOLE.print("   > Number of neighbors for smoothing initial frosting:", n_neighbors_for_smoothing_initial_frosting)
    CONSOLE.print("   > Min frosting size:", min_frosting_size)
    CONSOLE.print("   > Initial proposal std range:", initial_proposal_std_range)
    CONSOLE.print("   > Final proposal range:", final_proposal_range)
    CONSOLE.print("   > Final clamping range:", final_clamping_range)
    CONSOLE.print("   > Use background sphere:", use_background_sphere)
    CONSOLE.print("   > Use background gaussians:", use_background_gaussians)
    CONSOLE.print("   > (Deprecated) Min frosting factor:", min_frosting_factor)
    CONSOLE.print("   > (Deprecated) Max frosting factor:", max_frosting_factor)
    CONSOLE.print("   > (Deprecated) Min frosting range:", min_frosting_range)
    if use_surface_mesh_normal_consistency_loss:
        CONSOLE.print("Shell Regularization:")
        CONSOLE.print("   > Normal consistency factor:", surface_mesh_normal_consistency_factor)
    # CONSOLE.print("Number of vertices in the foreground:", n_vertices_in_fg)
    if use_custom_bbox:
        CONSOLE.print("Foreground bounding box min:", fg_bbox_min)
        CONSOLE.print("Foreground bounding box max:", fg_bbox_max)
    CONSOLE.print("Use eval split:", use_eval_split)
    CONSOLE.print("Use white background:", use_white_background)
    CONSOLE.print("Export ply at the end:", export_ply_at_the_end)
    CONSOLE.print("Export obj at the end:", export_obj_at_the_end)
    if export_obj_at_the_end:
        CONSOLE.print("Obj texture square size:", obj_texture_square_size)
    CONSOLE.print("----------------------------")
    
    # Setup device
    torch.cuda.set_device(num_device)
    CONSOLE.print("Using device:", num_device)
    device = torch.device(f'cuda:{num_device}')
    CONSOLE.print(torch.cuda.memory_summary())
    
    torch.autograd.set_detect_anomaly(detect_anomaly)
    
    # Creates save directory if it does not exist
    os.makedirs(frosting_checkpoint_path, exist_ok=True)
    
    # ====================Load NeRF model and training data====================

    # Load Gaussian Splatting checkpoint 
    CONSOLE.print(f"\nLoading config {gs_checkpoint_path}...")
    if use_eval_split:
        CONSOLE.print("Performing train/eval split...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=True,
        eval_split=use_eval_split,
        eval_split_interval=n_skip_images_for_eval_split,
        white_background=use_white_background,
        )

    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')

    if downscale_resolution_factor != 1:
       nerfmodel.downscale_output_resolution(downscale_resolution_factor)
    CONSOLE.print(f'\nCamera resolution scaled to '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_height} x '
          f'{nerfmodel.training_cameras.gs_cameras[0].image_width}'
          )
    
    # Load coarse SuGaR model
    CONSOLE.print(f'\nLoading SuGaR model {sugar_model_path}...')
    with torch.no_grad():
        checkpoint = torch.load(sugar_model_path, map_location=nerfmodel.device)
        sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=checkpoint['state_dict']['_points'],
            colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
            initialize=True,
            sh_levels=nerfmodel.gaussians.active_sh_degree+1,
            keep_track_of_knn=True,
            knn_to_track=16,
            beta_mode='average',  # 'learnable', 'average', 'weighted_average'
            primitive_types='diamond',  # 'diamond', 'square'
            surface_mesh_to_bind=None,  # Open3D mesh
            )
        sugar.load_state_dict(checkpoint['state_dict'])
            
    # Load shell base mesh
    shell_base_to_bind_full_path = shell_base_to_bind_path
    CONSOLE.print(f'\nLoading mesh to use as shell base: {shell_base_to_bind_full_path}...')
    o3d_mesh = o3d.io.read_triangle_mesh(shell_base_to_bind_full_path)
    CONSOLE.print("Mesh loaded.")
        
    # Background tensor if needed
    if use_white_background:
        bg_tensor = torch.ones(3, dtype=torch.float, device=nerfmodel.device)
    else:
        bg_tensor = torch.zeros(3, dtype=torch.float, device=nerfmodel.device)
    
    # ====================Initialize Frosting model====================
    # Construct Frosting model
    frosting = Frosting(
        nerfmodel=nerfmodel,
        coarse_sugar=sugar,
        sh_levels=sh_levels,
        shell_base_to_bind=o3d_mesh,  # Open3D mesh
        learn_shell=learn_shell,
        min_frosting_factor=min_frosting_factor,
        max_frosting_factor=max_frosting_factor,
        n_gaussians_in_frosting=n_gaussians_in_frosting,
        # Frosting initialization
        n_closest_gaussians_to_use_for_initializing_frosting=16,
        n_points_per_pass_for_initializing_frosting=2_000_000,
        n_samples_per_vertex_for_initializing_frosting=n_samples_per_vertex_for_initializing_frosting,
        frosting_level=frosting_level,
        smooth_initial_frosting=smooth_initial_frosting,
        n_neighbors_for_smoothing_initial_frosting=n_neighbors_for_smoothing_initial_frosting,
        # Edition
        editable=False,
        use_softmax_for_bary_coords=True,
        min_frosting_range=min_frosting_range,
        min_frosting_size=min_frosting_size,
        initial_proposal_std_range=initial_proposal_std_range,
        final_proposal_range=final_proposal_range,
        final_clamping_range=final_clamping_range,
        use_background_sphere=use_background_sphere,
        use_background_gaussians=use_background_gaussians,
    )
        
    CONSOLE.print(f'\Frosting model has been initialized.')
    CONSOLE.print(frosting)
    CONSOLE.print(f'Number of learnable parameters: {sum(p.numel() for p in frosting.parameters() if p.requires_grad)}')
    CONSOLE.print(f'Checkpoints will be saved in {frosting_checkpoint_path}')
    
    CONSOLE.print("\nModel parameters:")
    for name, param in frosting.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)
    
    # Cleaning memory
    sugar = None
    checkpoint = None
    gc.collect()
    torch.cuda.empty_cache()
    
    # Compute scene extent
    cameras_spatial_extent = frosting.get_cameras_spatial_extent()
    
    
    # ====================Initialize optimizer====================
    if use_custom_bbox:
        bbox_radius = ((torch.tensor(fg_bbox_max) - torch.tensor(fg_bbox_min)).norm(dim=-1) / 2.).item()
    else:
        bbox_radius = cameras_spatial_extent
        
    # TODO: Change spatial_lr_scale and compute better learning rates
    if False:
        spatial_lr_scale = 10. * bbox_radius / torch.tensor(n_vertices_in_fg).pow(1/2).item()
        print("Using as spatial_lr_scale:", spatial_lr_scale, "with bbox_radius:", bbox_radius, "and n_vertices_in_fg:", n_vertices_in_fg)
    else:
        if spatial_lr_scale is None:
            spatial_lr_scale = cameras_spatial_extent
        print("Using camera spatial extent as spatial_lr_scale:", spatial_lr_scale)
    
    opt_params = OptimizationParams(
        iterations=num_iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_bary_coords_lr_init=position_bary_coords_lr_init,
        position_bary_coords_lr_final=position_bary_coords_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
    )
    optimizer = FrostingOptimizer(frosting, opt_params, spatial_lr_scale=spatial_lr_scale)
    CONSOLE.print("Optimizer initialized.")
    CONSOLE.print("Optimization parameters:")
    CONSOLE.print(opt_params)
    
    CONSOLE.print("Optimizable parameters:")
    for param_group in optimizer.optimizer.param_groups:
        CONSOLE.print(param_group['name'], param_group['lr'])
        
    
    # ====================Loss function====================
    if loss_function == 'l1':
        loss_fn = l1_loss
    elif loss_function == 'l2':
        loss_fn = l2_loss
    elif loss_function == 'l1+dssim':
        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    CONSOLE.print(f'Using loss function: {loss_function}')
    
    
    # ====================Occlusion culling====================
    if use_occlusion_culling:
        CONSOLE.print("\nOcclusion culling will be performed.")
        CONSOLE.print(f"Method for performing occlusion culling: {occlusion_culling_type}")
        
        occlusion_culling_rasterizer = MeshRasterizer(
            cameras=frosting.nerfmodel.training_cameras,
            raster_settings=RasterizationSettings(
                image_size=(
                    frosting.nerfmodel.training_cameras.gs_cameras[0].image_height, 
                    frosting.nerfmodel.training_cameras.gs_cameras[0].image_width),
            ),
            use_nvdiffrast=True,
        )
        if occlusion_culling_rasterizer.use_nvdiffrast:
            CONSOLE.print("NVDiffRast is available. Using it for occlusion culling.")
        
        with torch.no_grad():
            if occlusion_culling_type == 'pix_to_face':
                face_idx_to_render_list = []
                for cam_idx in range(len(frosting.nerfmodel.training_cameras)):
                    if cam_idx % 10 == 0:
                        CONSOLE.print(f"Processing image {cam_idx} to prepare occlusion culling...")
                    face_idx_to_render_list.append(
                        occlusion_culling_rasterizer(
                            frosting.shell_base, 
                            cam_idx=cam_idx,
                        ).pix_to_face[0, ..., 0].unique()
                    )
                    
            elif occlusion_culling_type == 'depth':
                depth_for_filtering_list = []
                for cam_idx in range(len(frosting.nerfmodel.training_cameras)):
                    if cam_idx % 10 == 0:
                        CONSOLE.print(f"Processing image {cam_idx} to prepare occlusion culling...")
                    depth_for_filtering_list.append(
                        rasterization_with_pix_to_face(
                            frosting.shell_inner, # TODO: Try base shell
                            frosting.nerfmodel.training_cameras, cam_idx,
                            frosting.image_height, frosting.image_width,
                            ).zbuf[0, ..., 0]  # TODO: check that it is correct
                        )
        CONSOLE.print("Done computing data for occlusion culling.")
    
    # ====================Start training====================
    frosting.train()
    epoch = 0
    iteration = 0
    train_losses = []
    t0 = time.time()
    
    for batch in range(9_999_999):
        if iteration >= num_iterations:
            break
        
        # Shuffle images
        shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
        train_num_images = len(shuffled_idx)
        
        # We iterate on images
        for i in range(0, train_num_images, train_num_images_per_batch):
            iteration += 1
            
            # Update learning rates
            optimizer.update_learning_rate(iteration)
            
            start_idx = i
            end_idx = min(i+train_num_images_per_batch, train_num_images)
            
            camera_indices = shuffled_idx[start_idx:end_idx]
            
            # Computing rgb predictions
            depth_for_filtering = None
            face_idx_to_render = None
            if use_occlusion_culling:
                if iteration % use_occlusion_culling_every_n_iterations == 0:
                    if occlusion_culling_type == 'pix_to_face':
                        face_idx_to_render = face_idx_to_render_list[camera_indices.item()]
                    elif occlusion_culling_type == 'depth':
                        depth_for_filtering = depth_for_filtering_list[camera_indices.item()]
            outputs = frosting.render_image_gaussian_rasterizer( 
                camera_indices=camera_indices.item(),
                bg_color=bg_tensor,
                sh_deg=current_sh_levels-1,
                compute_color_in_rasterizer=compute_color_in_rasterizer,
                compute_covariance_in_rasterizer=True,
                depth_for_filtering=depth_for_filtering,
                face_idx_to_render=face_idx_to_render,
            )
            pred_rgb = outputs.view(-1, frosting.image_height, frosting.image_width, 3)
            pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)  # TODO: Change for torch.permute
            
            # Gather rgb ground truth
            gt_image = nerfmodel.get_gt_image(camera_indices=camera_indices)           
            gt_rgb = gt_image.view(-1, frosting.image_height, frosting.image_width, 3)
            gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)
                
            # Compute loss 
            loss = loss_fn(pred_rgb, gt_rgb)
            
            # Shell base optimization
            if use_surface_mesh_normal_consistency_loss:
                loss = loss + surface_mesh_normal_consistency_factor * mesh_normal_consistency(frosting.shell_base)
            
            # Update parameters
            loss.backward()
            
            # Optimization step
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            
            # Print loss
            if iteration==1 or iteration % print_loss_every_n_iterations == 0:
                CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                train_losses.append(loss.detach().item())
                CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{num_iterations:>5d}]",
                    "computed in", (time.time() - t0) / 60., "minutes.")
                with torch.no_grad():
                    for name, param in frosting.named_parameters():
                        CONSOLE.print(f"> {name}:", "Requires grad:", param.requires_grad)
                        if param.requires_grad:
                            CONSOLE.print(
                                f"     > Min:{param.min().item()}",
                                f"     > Max:{param.max().item()}",
                                f"     > Mean:{param.mean().item()}",
                                f"     > Std:{param.std().item()}"
                            )
                t0 = time.time()
                
            # Save model
            if (iteration % save_model_every_n_iterations == 0) or (iteration in save_milestones):
                CONSOLE.print("Saving model...")
                model_path = os.path.join(frosting_checkpoint_path, f'{iteration}.pt')
                frosting.save_model(path=model_path,
                                train_losses=train_losses,
                                epoch=epoch,
                                iteration=iteration,
                                optimizer_state_dict=optimizer.state_dict(),
                                )
                CONSOLE.print("Model saved.")
            
            if iteration >= num_iterations:
                break
            
            if do_sh_warmup and (iteration > 0) and (current_sh_levels < sh_levels) and (iteration % sh_warmup_every == 0):
                current_sh_levels += 1
                CONSOLE.print("Increasing number of spherical harmonics levels to", current_sh_levels)
            
            if do_resolution_warmup and (iteration > 0) and (current_resolution_factor > 1) and (iteration % resolution_warmup_every == 0):
                current_resolution_factor /= 2.
                nerfmodel.downscale_output_resolution(1/2)
                CONSOLE.print(f'\nCamera resolution scaled to '
                        f'{nerfmodel.training_cameras.ns_cameras.height[0].item()} x '
                        f'{nerfmodel.training_cameras.ns_cameras.width[0].item()}'
                        )
                frosting.adapt_to_cameras(nerfmodel.training_cameras)
                # TODO: resize GT images
        
        epoch += 1

    CONSOLE.print(f"Training finished after {num_iterations} iterations with loss={loss.detach().item()}.")
    CONSOLE.print("Saving final model...")
    model_path = os.path.join(frosting_checkpoint_path, f'{iteration}.pt')
    frosting.save_model(path=model_path,
                    train_losses=train_losses,
                    epoch=epoch,
                    iteration=iteration,
                    optimizer_state_dict=optimizer.state_dict(),
                    )

    CONSOLE.print("Final model saved.")
    
    if export_ply_at_the_end:
        # =====Build path=====
        CONSOLE.print("\nExporting ply file with refined Gaussians...")
        tmp_list = model_path.split(os.sep)
        tmp_list[-4] = 'refined_frosting_ply'
        tmp_list.pop(-1)
        tmp_list[-1] = tmp_list[-1] + '.ply'
        refined_ply_save_dir = os.path.join(*tmp_list[:-1])
        refined_ply_save_path = os.path.join(*tmp_list)
        os.makedirs(refined_ply_save_dir, exist_ok=True)
        
        # Export and save ply
        refined_gaussians = convert_frosting_into_gaussians(frosting)
        refined_gaussians.save_ply(refined_ply_save_path)
        CONSOLE.print("Ply file exported. This file is needed for using the dedicated viewer.")
        
    if export_obj_at_the_end:
        # =====Build path=====
        tmp_list = model_path.split(os.sep)
        tmp_list[-4] = 'refined_frosting_base_mesh'
        tmp_list.pop(-1)
        tmp_list[-1] = tmp_list[-1] + '.obj'
        refined_obj_save_dir = os.path.join(*tmp_list[:-1])
        refined_obj_save_path = os.path.join(*tmp_list)
        os.makedirs(refined_obj_save_dir, exist_ok=True)
        
        # Export and save mesh
        CONSOLE.print("Exporting textured mesh for visualization and edition in Blender...")
        textured_mesh = compute_textured_mesh_for_frosting_mesh(
            frosting,
            square_size=obj_texture_square_size,
            n_sh=0,
            texture_with_gaussian_renders=True,
            bg_color=bg_tensor,
            use_occlusion_culling=use_occlusion_culling,
        )
        CONSOLE.print("Textured mesh computed. Saving OBJ file with PyTorch3D (saving the textured OBJ seems to take a few minutes with PyTorch3D...)")
        with torch.no_grad():
            save_obj(  
                refined_obj_save_path,
                verts=textured_mesh.verts_list()[0],
                faces=textured_mesh.faces_list()[0],
                verts_uvs=textured_mesh.textures.verts_uvs_list()[0],
                faces_uvs=textured_mesh.textures.faces_uvs_list()[0],
                texture_map=textured_mesh.textures.maps_padded()[0].clamp(0., 1.),
            )
        CONSOLE.print("Textured mesh saved at:", refined_obj_save_path)
    
    return model_path