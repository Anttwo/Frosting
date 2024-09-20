import argparse
from frosting_utils.general_utils import str2bool
from frosting_extractors.textured_mesh import extract_mesh_and_texture_from_frosting


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to train a full macarons model in large 3D scenes.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  # --OK
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')  # --OK
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    parser.add_argument('-m', '--frosting_model_path',
                        type=str, 
                        help='(Required) Path to the Frosting model checkpoint.')  # --OK
    parser.add_argument('-o', '--mesh_output_dir',
                        type=str, 
                        default=None, 
                        help='path to the output directory.')  # --OK

    parser.add_argument('--use_occlusion_culling', type=str2bool, default=False, 
                        help='If True, uses occlusion culling for rendering and extracting texture.')
    parser.add_argument('--texture_square_size',
                        default=8, type=int, help='Size of the square to use for the texture.')  # --OK
    
    parser.add_argument('--eval', type=str2bool, default=False, help='Use eval split.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Index of GPU to use.')
    
    # Optional postprocessing
    args = parser.parse_args()
    
    # Call function
    extract_mesh_and_texture_from_frosting(args)
    