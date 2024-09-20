import os
import numpy as np
import torch
import open3d as o3d
# from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.io import save_obj
from frosting_scene.gs_model import GaussianSplattingWrapper
from frosting_scene.frosting_model import load_frosting_model
from frosting_utils.texture import compute_textured_mesh_for_frosting_mesh

from rich.console import Console
import time
import gc


def extract_mesh_and_texture_from_frosting(args):
    CONSOLE = Console(width=120)

    # ====================Parameters====================

    # Main
    num_device = args.gpu
    detect_anomaly = False
    
    # Data
    source_path = args.scene_path
    use_eval_split = args.eval
    use_white_background = args.white_background
    n_skip_images_for_eval_split = 8
    
    # 3DGS parameters
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load
    
    # Frosting parameters
    frosting_path = args.frosting_model_path
    use_occlusion_culling = args.use_occlusion_culling
    
    # Textured mesh extraction parameters
    obj_texture_square_size = args.texture_square_size
    obj_save_path = args.mesh_output_dir
    if obj_save_path is None:
        CONSOLE.print("No output directory provided. Using default directory.")
        if False:
            obj_save_path = os.path.join(os.path.dirname(frosting_path), 'textured_mesh.obj')
        else:
            tmp_list = frosting_path.split(os.sep)
            tmp_list[-4] = 'refined_frosting_base_mesh'
            tmp_list.pop(-1)
            tmp_list[-1] = tmp_list[-1] + '.obj'
            obj_save_dir = os.path.join(*tmp_list[:-1])
            obj_save_path = os.path.join(*tmp_list)
            os.makedirs(obj_save_dir, exist_ok=True)
    
    CONSOLE.print("\n-----Parsed parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("   > Content:", len(os.listdir(source_path)))
    CONSOLE.print("Gaussian Splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("   > Content:", len(os.listdir(gs_checkpoint_path)))
    CONSOLE.print("Frosting checkpoint path:", frosting_path)
    CONSOLE.print("Output path:", obj_save_path)
    CONSOLE.print("Data parameters:")
    CONSOLE.print("   > Use white background:", use_white_background)
    CONSOLE.print("   > Using eval split:", use_eval_split)
    CONSOLE.print("Frosting parameters:")
    CONSOLE.print("   > Use occlusion culling:", use_occlusion_culling)
    CONSOLE.print("Textured mesh extraction parameters:")
    CONSOLE.print("   > Texture square size:", obj_texture_square_size)
    CONSOLE.print("----------------------------")
    
    torch.cuda.set_device(num_device)
    
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
    
    # ====================Load Frosting model====================
    CONSOLE.print("\nLoading Frosting model...")
    frosting = load_frosting_model(
        frosting_path, 
        nerfmodel,
        learn_shell=False,
        frosting_level=0.01,
        min_frosting_size=0.001,
        use_background_sphere=False,
    )
    frosting.eval()
    CONSOLE.print("Frosting model loaded.")
    
    # ====================Extract mesh and texture====================
    CONSOLE.print("\nExtracting textured mesh...")
    textured_mesh = compute_textured_mesh_for_frosting_mesh(
        frosting,
        square_size=obj_texture_square_size,
        n_sh=0,
        texture_with_gaussian_renders=True,
        bg_color=[0., 0., 0.],
        use_occlusion_culling=use_occlusion_culling,
    )
    CONSOLE.print("Textured mesh extracted.")
    
    # ====================Save textured mesh====================
    CONSOLE.print("\nSaving textured mesh...")
    save_obj(  
        obj_save_path,
        verts=textured_mesh.verts_list()[0],
        faces=textured_mesh.faces_list()[0],
        verts_uvs=textured_mesh.textures.verts_uvs_list()[0],
        faces_uvs=textured_mesh.textures.faces_uvs_list()[0],
        texture_map=textured_mesh.textures.maps_padded()[0].clamp(0., 1.),
    )
    CONSOLE.print("Textured mesh saved.")
    
    return obj_save_path