import argparse
import os
import torch
import numpy as np
from PIL import Image
from frosting_utils.general_utils import str2bool
from blender.frosting_utils import (
    load_blender_package, 
    load_frosting_models_from_blender_package, 
    load_cameras_from_blender_package, 
    build_composite_scene,
    render_composited_image,
) 
from rich.console import Console


if __name__ == "__main__":
    print_every_n_frames = 5
    
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to render Frosting scenes edited or animated with Blender.')
    
    parser.add_argument('-p', '--package_path',
                        type=str, 
                        help='(Required) path to the Blender data package to use for rendering.')
    
    parser.add_argument('-o', '--output_path',
                        type=str, 
                        default=None,
                        help='Path to the output folder where to save the rendered images. \
                        If None, images will be saved in ./output/blender/renders/{package_name}.')
    
    parser.add_argument('--thickness_rescaling_method', type=str, default='median', 
                        help="Method to use for rescaling the thickness of the Frosting. Can be 'median' or 'triangle'."
                        "The 'triangle' method may be more accurate but also more unstable and can lead to more artifacts.")
    
    parser.add_argument('--adaptation_method', type=str, default='complex', 
                        help="Method to use for automatically adapting the parameters of the Gaussians. Can be 'simple' or 'complex'."
                        "The 'simple' method is faster and more stable but may be less accurate.")
    
    parser.add_argument('--deformation_threshold', type=float, default=2., 
                        help='Threshold for the deformation of the Frosting. '
                        'A face is considered too much deformed if its size increases by a ratio greater than this threshold.'
                        'The faces with a deformation greater than this threshold will be not be rendered.')
    
    parser.add_argument('--occlusion_culling', type=str2bool, default=False, 
                        help='Use occlusion culling for rendering. This should be set to True only if the scenes have been trained with occlusion culling.')
    
    parser.add_argument('--render_background_gaussians', type=str2bool, default=True, 
                        help='Render the background Gaussians for better quality. '
                        'Only the background Gaussians of the Frosting model with the highest number of Gaussians will be rendered, '
                        'as it is assumed to be the background scene.')
    
    parser.add_argument('--export_frame_as_ply', type=int, default=0, 
                        help='Export the Frosting representation of the scene at the specified frame as a PLY file. '
                        'If 0, no PLY file will be exported and all frames will be rendered.')
    
    parser.add_argument('--sh_degree', type=int, default=None, help='SH degree to use.')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    
    CONSOLE = Console(width=120)
    
    args = parser.parse_args()
    scene_name = os.path.splitext(os.path.basename(args.package_path))[0]
    output_path = args.output_path if args.output_path else f"./output/blender/renders/{scene_name}"
    sh_degree = args.sh_degree
    use_simple_adapt = True if args.adaptation_method == 'simple' else False
    thickness_rescaling_method = args.thickness_rescaling_method
    deformation_threshold = args.deformation_threshold
    use_occlusion_culling = args.occlusion_culling
    use_background_gaussians = args.render_background_gaussians
    frame_to_export_as_ply = args.export_frame_as_ply - 1
    
    # ----- Setup -----
    torch.cuda.set_device(args.gpu)
    device = torch.device(torch.cuda.current_device())
    CONSOLE.print("[INFO] Using device: ", device)
    CONSOLE.print("[INFO] Images will be saved in: ", output_path)
    
    # ----- Load Blender package -----
    CONSOLE.print("\nLoading Blender package...")
    package = load_blender_package(args.package_path, device)
    CONSOLE.print("Blender package loaded.")
    
    # ----- Load Frosting models -----
    CONSOLE.print("\nLoading Frosting models...")
    frosting_models, scene_paths = load_frosting_models_from_blender_package(package, device)
    
    # ----- Build composite scene -----
    CONSOLE.print("\nBuilding composite scene...")
    frosting_comp = build_composite_scene(
        frosting_models, scene_paths, package, 
        use_simple_adapt=use_simple_adapt,
        thickness_rescaling_method=thickness_rescaling_method,
        deformation_threshold=deformation_threshold,
    )
    if frosting_comp.editable:
        CONSOLE.print(f"{frosting_comp._edition_mask.sum()} / {frosting_comp._edition_mask.shape[0]} faces of the mesh are editable.")
    
    # ----- Build cameras -----
    CONSOLE.print("\nLoading cameras...")
    render_cameras = load_cameras_from_blender_package(package, device=device)
    
    # ----- Render and saving images -----
    CONSOLE.print("\nLoading successful. Rendering and saving images...")
    if use_occlusion_culling:
        CONSOLE.print("Using occlusion culling...")
    n_frames = len(package['camera']['lens'])
    os.makedirs(output_path, exist_ok=True)
    
    frosting_comp.eval()
    frosting_comp.adapt_to_cameras(render_cameras)
    frosting_comp.use_background_gaussians = frosting_comp.use_background_gaussians and use_background_gaussians
    
    with torch.no_grad():
        if frame_to_export_as_ply == -1:
            for i_frame in range(n_frames):
                # Change pose of meshes if needed and render the scene
                rgb_render = render_composited_image(
                    package=package,
                    frosting=frosting_comp, 
                    render_cameras=render_cameras, 
                    i_frame=i_frame,
                    sh_degree=sh_degree,
                    deformation_threshold=deformation_threshold,
                    use_occlusion_culling=use_occlusion_culling,
                ).nan_to_num().clamp(min=0, max=1)
            
                # Save image
                save_path = os.path.join(output_path, f"{i_frame+1:04d}.png")
                img = Image.fromarray((rgb_render.cpu().numpy() * 255).astype(np.uint8))
                img.save(save_path)
                
                # Info
                if i_frame % print_every_n_frames == 0:
                    print(f"Saved frame {i_frame} to {save_path}")
                    
                torch.cuda.empty_cache()
        else:
            # Export PLY file
            ply_save_path = os.path.join(output_path, f"{frame_to_export_as_ply+1:04d}.ply")
            render_composited_image(
                package=package,
                frosting=frosting_comp, 
                render_cameras=render_cameras, 
                i_frame=frame_to_export_as_ply,
                sh_degree=sh_degree,
                deformation_threshold=deformation_threshold,
                use_occlusion_culling=use_occlusion_culling,
                return_GS_model=True,
            ).save_ply(ply_save_path)
            CONSOLE.print(f"Exported PLY file of frame {frame_to_export_as_ply+1} to {ply_save_path}")

CONSOLE.print("Rendering completed.")
