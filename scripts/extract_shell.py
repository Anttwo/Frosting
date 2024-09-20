import argparse
from frosting_utils.general_utils import str2bool
from frosting_extractors.coarse_shell import extract_shell_base_from_coarse_sugar

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to extract a mesh from a coarse SuGaR scene.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='path to the scene data to use.')
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    parser.add_argument('-m', '--coarse_model_path', type=str, default=None, help='')
    
    parser.add_argument('-l', '--surface_level', type=float, default=None, 
                        help='Surface level to extract the mesh at. If None, will extract levels 0.1, 0.3 and 0.5')
    parser.add_argument('-d', '--decimation_target', type=int, default=None, 
                        help='Target number of vertices to decimate the mesh to. If None, will decimate to 200_000 and 1_000_000.')
    
    parser.add_argument('--poisson_depth', type=int, default=-1, 
                        help="Depth of Poisson reconstruction. If -1, will compute automatically the depth based on the SuGaR model.")
    parser.add_argument('--cleaning_quantile', type=float, default=0.1, 
                        help='Quantile to use for cleaning the Poisson mesh.')
    parser.add_argument('--connected_components_vis_th', type=float, default=0.5, 
                        help='Threshold to use for removing non-visible connected components in the mesh.')
    parser.add_argument('--project_mesh_on_surface_points', type=str2bool, default=True, help='If True, project the mesh on the surface points for better details.')
    
    parser.add_argument('-o', '--mesh_output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')
    
    parser.add_argument('-b', '--bboxmin', type=str, default=None, help='Min coordinates to use for foreground.')
    parser.add_argument('-B', '--bboxmax', type=str, default=None, help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, help='If True, center the bounding box. Default is True.')
    
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')
    parser.add_argument('--use_centers_to_extract_mesh', type=str2bool, default=False, 
                        help='If True, just use centers of the gaussians to extract mesh.')
    parser.add_argument('--use_marching_cubes', type=str2bool, default=False, 
                        help='If True, use marching cubes to extract mesh.')
    parser.add_argument('--use_vanilla_3dgs', type=str2bool, default=False, 
                        help='If True, use vanilla 3DGS to extract mesh.')
    
    args = parser.parse_args()
    
    # Call function
    extract_shell_base_from_coarse_sugar(args)
    