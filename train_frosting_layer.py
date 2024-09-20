import argparse
from frosting_utils.general_utils import str2bool
from frosting_trainers.refine import refined_training

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to train a Frosting model.')
    
    # Data parameters
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='path to the scene data to use.')  
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')  
    parser.add_argument('--sugar_path', type=str,
                        help='Path to the coarse sugar model to use.')
    parser.add_argument('-m', '--mesh_path', 
                        type=str, 
                        help='Path to the extracted mesh file to use as a shell base for Frosting optimization.')  
    parser.add_argument('-o', '--output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')  
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    # Render parameters
    parser.add_argument('--use_occlusion_culling', type=str2bool, default=False, 
                        help='If True, uses occlusion culling during training.')
    
    # Frosting parameters
    parser.add_argument('--learn_shell', type=str2bool, default=False, 
                        help='If True, also optimize the shell vertices.')
    parser.add_argument('--regularize_shell', type=str2bool, default=False, 
                        help='If True, also regularize the base shell vertices with a normal consistency loss.')
    parser.add_argument('-n', '--normal_consistency_factor', type=float, default=0.1, 
                        help='Factor to multiply the normal consistency loss by.')
    parser.add_argument('-g', '--gaussians_in_frosting', type=int, default=2_000_000, 
                        help='Total number of gaussians in the frosting layer.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000, 
                        help='Number of refinement iterations.')
    
    # Deprecated
    parser.add_argument('--min_frosting_factor', type=float, default=-0.5, 
                        help='(Deprecated) Min frosting factor.')
    parser.add_argument('--max_frosting_factor', type=float, default=1.5,
                        help='(Deprecated) Max frosting factor.')
    parser.add_argument('--min_frosting_range', type=float, default=0.,
                        help='(Deprecated) Minimum range for sampling points to compute initial frosting.')
    
    # For research
    parser.add_argument('--n_samples_per_vertex', type=int, default=21,
                        help='Number of samples per vertex for initializing frosting.')
    parser.add_argument('--frosting_level', type=float, default=0.01,
                        help='Isosurface level to use for initializing frosting size.')
    parser.add_argument('--smooth_initial_frosting', type=str2bool, default=True, 
                        help='If True, smooth the initial frosting.')
    parser.add_argument('--n_neighbors_for_smoothing', type=int, default=4,
                        help='Number of neighbors used for smoothing initial frosting.')
    parser.add_argument('--min_frosting_size', type=float, default=0.001,
                        help='Minimum size for the initial frosting.')
    parser.add_argument('--initial_proposal_std_range', type=float, default=3.,
                        help='Maximum range for the initial proposal interval, in terms of multiples of the closest Gaussian standard deviation.')
    parser.add_argument('--final_proposal_range', type=float, default=3.,
                        help='Maximum local range for the proposal interval, after refinement with the volumetric 3DGS. '
                        'This value is multiplied by the proposal range.')
    parser.add_argument('--final_clamping_range', type=float, default=0.1,
                        help='Minimum local size for the frosting interval, after refinement with the volumetric 3DGS. '
                        'This value is multiplied by the proposal range.')
    parser.add_argument('--use_background_sphere', type=str2bool, default=False, 
                        help='If True, optimizes a sky sphere in the background.')
    parser.add_argument('--use_background_gaussians', type=str2bool, default=True, 
                        help='If True, optimizes Gaussians in the background.')
    
    # Scene parameters
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.') 
    
    parser.add_argument('--eval', type=str2bool, default=False, help='Use eval split.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    
    # Misc
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--export_ply', type=str2bool, default=True, 
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training.')
    parser.add_argument('--export_obj', type=str2bool, default=True, 
                        help='If True, export a textured mesh as an obj file for visualization and edition in Blender.')
    parser.add_argument('--texture_square_size', type=int, default=8, 
                        help='Size of the square to use for the UV texture. Increase for higher texture resolution.')

    args = parser.parse_args()
    
    # Call function
    refined_training(args)