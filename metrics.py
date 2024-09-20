import argparse
import os
import json
import numpy as np
import torch
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.lpipsPyTorch import lpips
from frosting_scene.gs_model import GaussianSplattingWrapper
from frosting_utils.general_utils import str2bool
from frosting_scene.sugar_model import load_refined_model
from frosting_scene.frosting_model import load_frosting_model
from gaussian_splatting.scene.dataset_readers import CameraInfo
from frosting_scene.cameras import CamerasWrapper, GSCamera
from gaussian_splatting.utils.graphics_utils import focal2fov, fov2focal
from pathlib import Path
from PIL import Image

from rich.console import Console
CONSOLE = Console(width=120)

os.makedirs('./lpipsPyTorch/weights/', exist_ok=True)
torch.hub.set_dir('./lpipsPyTorch/weights/')

n_skip_images_for_eval_split = 8


def readCamerasFromTestTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # image_path = os.path.join(path, cam_name)
            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos


if __name__ == "__main__":
    
    # Parser
    parser = argparse.ArgumentParser(description='Script to evaluate Frosting models.')
    
    # Config file for scenes to evaluate
    parser.add_argument('--scenes_config', type=str, 
                        help='(Required) Path to the JSON file containing the config parameters.')
    
    # Device
    parser.add_argument('--gpu', type=int, default=0, 
                        help='Index of GPU to use.')
    
    # (Optional) Additional evaluation parameters
    parser.add_argument('--evaluate_vanilla', type=str2bool, default=False, 
                        help='If True, will also evaluate vanilla 3DGS, in addition to Frosting.')
    
    args = parser.parse_args()
            
    # --- Scenes dict ---
    with open(args.scenes_config, 'r') as f:
        config = json.load(f)
        
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    bg_color = config['bg_color']
    white_background = config['white_background']
    load_training_data = config['load_training_data']
    use_standard_eval_split = config['use_standard_eval_split']
    coarse_sugar_config = config['coarse_sugar']
    if 'frosting_refinement' in config:
        CONSOLE.print("Model type: Frosting")
        model_type = 'frosting'
        frosting_extraction_config = config['frosting_extraction']
        frosting_refinement_config = config['frosting_refinement']
    elif 'sugar_refinement' in config:
        CONSOLE.print("Model type: SuGaR")
        model_type = 'sugar'
        mesh_extraction_config = config['mesh_extraction']
        sugar_refinement_config = config['sugar_refinement']
    else:
        raise ValueError("Unknown model type. Please specify either 'frosting_refinement' or 'sugar_refinement' in the config file.")
    
    gs_checkpoints_eval = config["scenes"]
    
    # --- Coarse model parameters ---
    coarse_iteration_to_load = coarse_sugar_config["iteration_to_load"]
    coarse_estimation_factor = coarse_sugar_config["estimation_factor"]
    estim_method = coarse_sugar_config["regularization_type"]
    coarse_normal_factor = coarse_sugar_config["normal_factor"]
    
    if model_type == 'frosting':
        # --- Frosting extraction parameters ---
        surface_levels = [frosting_extraction_config["surface_level"]]
        decimation_targets = [frosting_extraction_config["n_vertices_in_mesh"]]
        poisson_depth = frosting_extraction_config["poisson_depth"]
        cleaning_quantile = frosting_extraction_config["cleaning_quantile"]
        connected_components_vis_th = frosting_extraction_config["connected_components_vis_th"]
        
        # --- Frosting refinement parameters ---
        use_occlusion_culling = frosting_refinement_config["use_occlusion_culling"]
        gaussians_in_frosting = frosting_refinement_config["gaussians_in_frosting"]
        refinement_iterations_list = [frosting_refinement_config["refinement_iterations"]]
        frosting_level = frosting_refinement_config["frosting_level"]
        min_frosting_size = frosting_refinement_config["min_frosting_size"]
        initial_proposal_std_range = frosting_refinement_config["initial_proposal_std_range"]
        final_proposal_range = frosting_refinement_config["final_proposal_range"]
        final_clamping_range = frosting_refinement_config["final_clamping_range"]
        surface_mesh_normal_consistency_factor = 0.1
        use_background_sphere = frosting_refinement_config["use_background_sphere"]
        use_background_gaussians = frosting_refinement_config["use_background_gaussians"]
    
    elif model_type == 'sugar':
        # Mesh extraction parameters
        surface_levels = [mesh_extraction_config["surface_level"]]
        decimation_targets = [mesh_extraction_config["n_vertices_in_mesh"]]
        
        # Sugar refinement parameters
        # TODO
        normal_consistency_factor = sugar_refinement_config["normal_consistency_factor"]
        n_gaussians_per_face = sugar_refinement_config["n_gaussians_per_face"]
        refinement_iterations_list = [sugar_refinement_config["refinement_iterations"]]
        
    # --- Evaluation parameters ---
    evaluate_vanilla = args.evaluate_vanilla
            
    CONSOLE.print('==================================================')
    CONSOLE.print("Starting evaluation with the following parameters:")
    CONSOLE.print(f"Dataset name: {dataset_name}")
    CONSOLE.print(f"Dataset type: {dataset_type}")
    CONSOLE.print(f"Background color: {bg_color}")
    CONSOLE.print(f"White background: {white_background}")
    CONSOLE.print(f"Load training data: {load_training_data}")
    CONSOLE.print(f"Use standard eval split: {use_standard_eval_split}")
    CONSOLE.print(f"Coarse SuGaR parameters:")
    CONSOLE.print(f"   > Estimation method: {estim_method}")
    CONSOLE.print(f"   > Coarse iteration to load: {coarse_iteration_to_load}")
    CONSOLE.print(f"   > Coarse estimation factor: {coarse_estimation_factor}")
    CONSOLE.print(f"   > Coarse normal factor: {coarse_normal_factor}")
    if model_type == 'frosting':
        CONSOLE.print(f"Frosting extraction parameters:")
        CONSOLE.print(f"   > Surface levels: {surface_levels}")
        CONSOLE.print(f"   > Decimation targets: {decimation_targets}")
        CONSOLE.print(f"   > Poisson depth: {poisson_depth}")
        CONSOLE.print(f"   > Cleaning quantile: {cleaning_quantile}")
        CONSOLE.print(f"   > Connected components visibility threshold: {connected_components_vis_th}")
        CONSOLE.print(f"Frosting refinement parameters:")
        CONSOLE.print(f"   > Use occlusion culling: {use_occlusion_culling}")
        CONSOLE.print(f"   > Gaussians in frosting: {gaussians_in_frosting}")
        CONSOLE.print(f"   > Frosting level: {frosting_level}")
        CONSOLE.print(f"   > Min frosting size: {min_frosting_size}")
        CONSOLE.print(f"   > Initial proposal std range: {initial_proposal_std_range}")
        CONSOLE.print(f"   > Final proposal range: {final_proposal_range}")
        CONSOLE.print(f"   > Final clamping range: {final_clamping_range}")
        CONSOLE.print(f"   > Surface mesh normal consistency factor: {surface_mesh_normal_consistency_factor}")
        CONSOLE.print(f"   > Refinement iterations: {refinement_iterations_list}")
        CONSOLE.print(f"   > Use background sphere: {use_background_sphere}")
        CONSOLE.print(f"   > Use background gaussians: {use_background_gaussians}")
    elif model_type == 'sugar':
        CONSOLE.print(f"Mesh extraction parameters:")
        CONSOLE.print(f"   > Surface levels: {surface_levels}")
        CONSOLE.print(f"   > Decimation targets: {decimation_targets}")
        CONSOLE.print(f"Sugar refinement parameters:")
        CONSOLE.print(f"   > Normal consistency factor: {normal_consistency_factor}")
        CONSOLE.print(f"   > Gaussians per face: {n_gaussians_per_face}")
        CONSOLE.print(f"   > Refinement iterations: {refinement_iterations_list}")
    CONSOLE.print(f"GS checkpoints for evaluation: {gs_checkpoints_eval}")
    CONSOLE.print(f"Evaluate vanilla: {evaluate_vanilla}")
    CONSOLE.print('==================================================')
    
    # Set the GPU
    torch.cuda.set_device(args.gpu)
    device = torch.device(torch.cuda.current_device())
    
    # ==========================

    result_file_dir = './output/metrics/'
    os.makedirs(result_file_dir, exist_ok=True)
    results = {}
    
    for source_path in gs_checkpoints_eval.keys():
        scene_name = source_path.split('/')[-1]
        CONSOLE.print(f"\n===== Processing scene {scene_name}... =====")
        scene_results = {}
        
        # Loading vanilla 3DGS models
        gs_checkpoint_path = gs_checkpoints_eval[source_path]
        bg_tensor = torch.tensor(bg_color).float().cuda()
        
        CONSOLE.print("Source path:", source_path)
        CONSOLE.print("Gaussian splatting checkpoint path:", gs_checkpoint_path)    
        CONSOLE.print(f"\nLoading Vanilla 3DGS model config {gs_checkpoint_path}...")
        
        nerfmodel_7k = GaussianSplattingWrapper(
            source_path=source_path,
            output_path=gs_checkpoint_path,
            iteration_to_load=7000,
            load_gt_images=load_training_data,
            eval_split=use_standard_eval_split,
            eval_split_interval=n_skip_images_for_eval_split,
            background=bg_color,
            white_background=white_background,
            )
        
        try:
            nerfmodel_30k = GaussianSplattingWrapper(
                source_path=source_path,
                output_path=gs_checkpoint_path,
                iteration_to_load=30_000,
                load_gt_images=False,
                eval_split=use_standard_eval_split,
                eval_split_interval=n_skip_images_for_eval_split,
                background=bg_color,
                white_background=white_background,
                )
        except:
            CONSOLE.print("Could not load 30K model. Using only 7K model for evaluation.")
            nerfmodel_30k = None
        
        sh_deg_to_use = nerfmodel_7k.gaussians.active_sh_degree
        CONSOLE.print("Vanilla 3DGS Loaded.")
        CONSOLE.print("Using SH degree:", sh_deg_to_use)
        
        # If dataset is synthetic, use separate test data
        if dataset_type == 'synthetic':
            CONSOLE.print("Loading separate test data for synthetic dataset.")
            transforms_file = 'transforms_test.json'
            
            shelly_list = ['pug', 'kitten', 'woolly', 'horse', 'khady', 'fernvase']
            synthetic_nerf_list = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
            
            if scene_name in shelly_list:
                test_data_extension = ''
            elif scene_name in synthetic_nerf_list:
                test_data_extension = '.png'
            else:
                raise ValueError(f"Unknown synthetic scene name: {scene_name}")
            
            test_cam_infos = readCamerasFromTestTransforms(
                path=source_path,
                transformsfile=transforms_file,
                white_background=white_background,
                extension=test_data_extension,
            )
            cam_indices = [cam_idx for cam_idx in range(len(test_cam_infos))]
            CONSOLE.print("Number of test cameras:", len(test_cam_infos))
            CONSOLE.print(f"Image size: {test_cam_infos[0].width}x{test_cam_infos[0].height}")
            
        elif dataset_type == 'real':    
            CONSOLE.print("Number of test cameras:", len(nerfmodel_7k.test_cameras))
            cam_indices = [cam_idx for cam_idx in range(len(nerfmodel_7k.test_cameras))]
            
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        compute_lpips = True
        
        # Evaluating Vanilla 3DGS
        if evaluate_vanilla:
            CONSOLE.print("\n--- Starting Evaluation of Vanilla 3DGS... ---")

            gs_7k_ssims = []
            gs_7k_psnrs = []
            gs_7k_lpipss = []
            
            if nerfmodel_30k is not None:
                gs_30k_ssims = []
                gs_30k_psnrs = []
                gs_30k_lpipss = []
            
            with torch.no_grad():    
                for cam_i in cam_indices:
                    # GT image
                    if dataset_type == 'real':
                        cam_idx = cam_i
                        gt_img = nerfmodel_7k.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                        test_cameras = nerfmodel_7k.test_cameras
                    elif dataset_type == 'synthetic':
                        cam_idx = 0
                        gt_img = np.array(test_cam_infos[cam_i].image) / 255.
                        gt_img = torch.tensor(gt_img, dtype=torch.float32).cuda().permute(2, 0, 1)
                        gs_cameras = [GSCamera(
                            colmap_id=test_cam_infos[cam_i].uid,
                            R=test_cam_infos[cam_i].R,
                            T=test_cam_infos[cam_i].T,
                            FoVy=test_cam_infos[cam_i].FovY,
                            FoVx=test_cam_infos[cam_i].FovX,
                            image_width=test_cam_infos[cam_i].width,
                            image_height=test_cam_infos[cam_i].height,
                            uid=test_cam_infos[cam_i].uid,
                            image_name=test_cam_infos[cam_i].image_name,
                            image=gt_img,
                            gt_alpha_mask=None,
                        )]
                        test_cameras = CamerasWrapper(gs_cameras)
                    else:
                        raise ValueError(f"Unknown dataset type: {dataset_type}")
                    
                    # Vanilla 3DGS image (30K)
                    if nerfmodel_30k is not None:
                        gs_30k_img = nerfmodel_30k.render_image(
                            nerf_cameras=test_cameras,
                            camera_indices=cam_idx).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)                        
                        gs_30k_ssims.append(ssim(gs_30k_img, gt_img))
                        gs_30k_psnrs.append(psnr(gs_30k_img, gt_img))
                        gs_30k_lpipss.append(lpips(gs_30k_img, gt_img, net_type='vgg'))
                    
                    # Vanilla 3DGS image (7K)
                    gs_7k_img = nerfmodel_7k.render_image(
                        nerf_cameras=test_cameras,
                        camera_indices=cam_idx).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                    gs_7k_ssims.append(ssim(gs_7k_img, gt_img))
                    gs_7k_psnrs.append(psnr(gs_7k_img, gt_img))
                    gs_7k_lpipss.append(lpips(gs_7k_img, gt_img, net_type='vgg'))    
                    
            CONSOLE.print("Evaluation of Vanilla 3DGS finished.")
            scene_results['3dgs_7k'] = {}
            scene_results['3dgs_7k']['ssim'] = torch.tensor(gs_7k_ssims).mean().item()
            scene_results['3dgs_7k']['psnr'] = torch.tensor(gs_7k_psnrs).mean().item()
            scene_results['3dgs_7k']['lpips'] = torch.tensor(gs_7k_lpipss).mean().item()
            
            CONSOLE.print(f"\nVanilla 3DGS results (7K iterations):")
            CONSOLE.print("SSIM:", torch.tensor(gs_7k_ssims).mean())
            CONSOLE.print("PSNR:", torch.tensor(gs_7k_psnrs).mean())
            CONSOLE.print("LPIPS:", torch.tensor(gs_7k_lpipss).mean())
            
            if nerfmodel_30k is not None:
                scene_results['3dgs_30k'] = {}
                scene_results['3dgs_30k']['ssim'] = torch.tensor(gs_30k_ssims).mean().item()
                scene_results['3dgs_30k']['psnr'] = torch.tensor(gs_30k_psnrs).mean().item()
                scene_results['3dgs_30k']['lpips'] = torch.tensor(gs_30k_lpipss).mean().item()
            
                CONSOLE.print(f"\bVanilla 3DGS results (30K iterations):")
                CONSOLE.print("SSIM:", torch.tensor(gs_30k_ssims).mean())
                CONSOLE.print("PSNR:", torch.tensor(gs_30k_psnrs).mean())
                CONSOLE.print("LPIPS:", torch.tensor(gs_30k_lpipss).mean())
        
        # Evaluating Frosting models
        if model_type == 'frosting':
            with torch.no_grad():
                CONSOLE.print("\n--- Starting Evaluation of Frosting... ---")
                for surface_level in surface_levels:
                    for decimation_target in decimation_targets:
                        for refinement_iterations in refinement_iterations_list:
                            estim_factor_str = str(coarse_estimation_factor).replace('.', '')
                            normal_factor_str = str(coarse_normal_factor).replace('.', '')
                            surface_level_str = str(surface_level).replace('.', '')
                            cleaning_quantile_str = str(cleaning_quantile).replace('.', '')
                            frosting_level_str = str(frosting_level).replace('.', '')
                            final_proposal_range_str = str(final_proposal_range).replace('.', '')
                            if estim_method == 'dn_consistency':
                                estim_method_str = 'density'
                            else:
                                estim_method_str = estim_method
                            refined_frosting_path = f"./output/refined_frosting/{scene_name}/frostingfine_3Dgs{coarse_iteration_to_load}_{estim_method_str}estim{estim_factor_str}_sdfnorm{normal_factor_str}_level{surface_level_str}_decim{decimation_target}_depth{poisson_depth}_quantile{cleaning_quantile_str}_gauss{gaussians_in_frosting}_frostlevel{frosting_level_str}_proposal{final_proposal_range_str}/{refinement_iterations}.pt"
                            refined_frosting_str = f"{estim_method}estim{estim_factor_str}_sdfnorm{normal_factor_str}_level{surface_level_str}_decim{decimation_target}_depth{poisson_depth}_quantile{cleaning_quantile_str}_gauss{gaussians_in_frosting}_frostlevel{frosting_level_str}_proposal{final_proposal_range_str}_refined{refinement_iterations}"
                            
                            # Loading refined Frosting model
                            CONSOLE.print(f"Loading Frosting model config {refined_frosting_path}...")
                            
                            frosting = load_frosting_model(refined_frosting_path, nerfmodel_7k,
                                learn_shell=False,
                                n_gaussians_in_frosting=gaussians_in_frosting,
                                frosting_level=frosting_level,
                                min_frosting_size=min_frosting_size,
                                initial_proposal_std_range=initial_proposal_std_range,
                                final_proposal_range=final_proposal_range,
                                final_clamping_range=final_clamping_range,
                                use_background_sphere=use_background_sphere,
                                use_background_gaussians=use_background_gaussians,
                            )
                            frosting.eval()
                            
                            # Evaluating Frosting
                            with torch.no_grad():                
                                frosting_ssims = []
                                frosting_psnrs = []
                                frosting_lpipss = []
                                
                                for cam_i in cam_indices:                                
                                    # GT image
                                    if dataset_type == 'real':
                                        cam_idx = cam_i
                                        gt_img = nerfmodel_7k.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                                        test_cameras = nerfmodel_7k.test_cameras
                                    elif dataset_type == 'synthetic':
                                        cam_idx = 0
                                        gt_img = np.array(test_cam_infos[cam_i].image) / 255.
                                        gt_img = torch.tensor(gt_img, dtype=torch.float32).cuda().permute(2, 0, 1)
                                        gs_cameras = [GSCamera(
                                            colmap_id=test_cam_infos[cam_i].uid,
                                            R=test_cam_infos[cam_i].R,
                                            T=test_cam_infos[cam_i].T,
                                            FoVy=test_cam_infos[cam_i].FovY,
                                            FoVx=test_cam_infos[cam_i].FovX,
                                            image_width=test_cam_infos[cam_i].width,
                                            image_height=test_cam_infos[cam_i].height,
                                            uid=test_cam_infos[cam_i].uid,
                                            image_name=test_cam_infos[cam_i].image_name,
                                            image=gt_img,
                                            gt_alpha_mask=None,
                                        )]
                                        test_cameras = CamerasWrapper(gs_cameras)
                                    else:
                                        raise ValueError(f"Unknown dataset type: {dataset_type}")
                                    
                                    # Frosting image
                                    frosting_img = frosting.render_image_gaussian_rasterizer(
                                        nerf_cameras=test_cameras,
                                        camera_indices=cam_idx,
                                        bg_color=bg_tensor,
                                        sh_deg=sh_deg_to_use,
                                        compute_color_in_rasterizer=True,
                                        use_occlusion_culling=use_occlusion_culling,
                                    ).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                                    
                                    frosting_ssims.append(ssim(frosting_img, gt_img))
                                    frosting_psnrs.append(psnr(frosting_img, gt_img))
                                    frosting_lpipss.append(lpips(frosting_img, gt_img, net_type='vgg'))
                            
                            CONSOLE.print(f"Evaluation of Frosting finished, with config {refined_frosting_str}.")
                            scene_results[f'frosting'] = {}
                            scene_results[f'frosting']['ssim'] = torch.tensor(frosting_ssims).mean().item()
                            scene_results[f'frosting']['psnr'] = torch.tensor(frosting_psnrs).mean().item()
                            scene_results[f'frosting']['lpips'] = torch.tensor(frosting_lpipss).mean().item()
                            
                            CONSOLE.print(f"Frosting results:")
                            CONSOLE.print("SSIM:", torch.tensor(frosting_ssims).mean())
                            CONSOLE.print("PSNR:", torch.tensor(frosting_psnrs).mean())
                            CONSOLE.print("LPIPS:", torch.tensor(frosting_lpipss).mean())
        
        elif model_type == 'sugar':
            with torch.no_grad():
                CONSOLE.print("\n--- Starting Evaluation of SuGaR... ---")
                for surface_level in surface_levels:
                    for decimation_target in decimation_targets:
                        for refinement_iterations in refinement_iterations_list:
                            estim_factor_str = str(coarse_estimation_factor).replace('.', '')
                            normal_factor_str = str(coarse_normal_factor).replace('.', '')
                            surface_level_str = str(surface_level).replace('.', '')
                            normal_consistency_str = str(normal_consistency_factor).replace('.', '')
                            if estim_method == 'dn_consistency':
                                estim_method_str = 'density'
                            else:
                                estim_method_str = estim_method
                            refined_sugar_path = f"./output/refined/{scene_name}/sugarfine_3Dgs{coarse_iteration_to_load}_{estim_method_str}estim{estim_factor_str}_sdfnorm{normal_factor_str}_level{surface_level_str}_decim{decimation_target}_normalconsistency{normal_consistency_str}_gaussperface{n_gaussians_per_face}/{refinement_iterations}.pt"
                            refined_sugar_str = f"{estim_method}estim{estim_factor_str}_sdfnorm{normal_factor_str}_level{surface_level_str}_decim{decimation_target}_normalconsistency{normal_consistency_str}_gaussperface{n_gaussians_per_face}_refined{refinement_iterations}"
                            
                            # Loading refined SuGaR model
                            CONSOLE.print(f"Loading SuGaR model config {refined_sugar_path}...")
                            sugar = load_refined_model(refined_sugar_path, nerfmodel_7k)
                            sugar.eval()
                            
                            # Evaluating SuGaR
                            with torch.no_grad():                
                                sugar_ssims = []
                                sugar_psnrs = []
                                sugar_lpipss = []
                                
                                for cam_i in cam_indices:                                
                                    # GT image
                                    if dataset_type == 'real':
                                        cam_idx = cam_i
                                        gt_img = nerfmodel_7k.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                                        test_cameras = nerfmodel_7k.test_cameras
                                    elif dataset_type == 'synthetic':
                                        cam_idx = 0
                                        gt_img = np.array(test_cam_infos[cam_i].image) / 255.
                                        gt_img = torch.tensor(gt_img, dtype=torch.float32).cuda().permute(2, 0, 1)
                                        gs_cameras = [GSCamera(
                                            colmap_id=test_cam_infos[cam_i].uid,
                                            R=test_cam_infos[cam_i].R,
                                            T=test_cam_infos[cam_i].T,
                                            FoVy=test_cam_infos[cam_i].FovY,
                                            FoVx=test_cam_infos[cam_i].FovX,
                                            image_width=test_cam_infos[cam_i].width,
                                            image_height=test_cam_infos[cam_i].height,
                                            uid=test_cam_infos[cam_i].uid,
                                            image_name=test_cam_infos[cam_i].image_name,
                                            image=gt_img,
                                            gt_alpha_mask=None,
                                        )]
                                        test_cameras = CamerasWrapper(gs_cameras)
                                    else:
                                        raise ValueError(f"Unknown dataset type: {dataset_type}")
                                    
                                    # SuGaR image
                                    sugar_img = sugar.render_image_gaussian_rasterizer(
                                        nerf_cameras=test_cameras,
                                        camera_indices=cam_idx,
                                        bg_color=bg_tensor,
                                        sh_deg=sh_deg_to_use,
                                        compute_color_in_rasterizer=True,#compute_color_in_rasterizer,
                                    ).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                                    
                                    sugar_ssims.append(ssim(sugar_img, gt_img))
                                    sugar_psnrs.append(psnr(sugar_img, gt_img))
                                    sugar_lpipss.append(lpips(sugar_img, gt_img, net_type='vgg'))
                            
                            CONSOLE.print(f"Evaluation of SuGaR finished, with config {refined_sugar_str}.")
                            scene_results[f'sugar'] = {}
                            scene_results[f'sugar']['ssim'] = torch.tensor(sugar_ssims).mean().item()
                            scene_results[f'sugar']['psnr'] = torch.tensor(sugar_psnrs).mean().item()
                            scene_results[f'sugar']['lpips'] = torch.tensor(sugar_lpipss).mean().item()
                            
                            CONSOLE.print(f"SuGaR results:")
                            CONSOLE.print("SSIM:", torch.tensor(sugar_ssims).mean())
                            CONSOLE.print("PSNR:", torch.tensor(sugar_psnrs).mean())
                            CONSOLE.print("LPIPS:", torch.tensor(sugar_lpipss).mean())
        
        # Saves results to JSON file   
        results[scene_name] = scene_results
        if model_type == 'frosting':        
            result_file_name = f'results_{dataset_name}_{refined_frosting_str}.json'
        elif model_type == 'sugar':
            result_file_name = f'results_sugar_{dataset_name}_{refined_sugar_str}.json'
        result_file_name = os.path.join(result_file_dir, result_file_name)

        CONSOLE.print(f"Saving results to {result_file_name}...")
        with open(result_file_name, 'w') as f:
            json.dump(results, f, indent=4)