import os
import argparse
import numpy as np
from frosting_utils.general_utils import str2bool


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full Frosting model.')
    
    # Data
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  
    
    # Vanilla 3DGS optimization at beginning
    parser.add_argument('--gs_output_dir', type=str, default=None,
                        help='(Optional) If None, will automatically train a vanilla Gaussian Splatting model at the beginning of the training. '
                        'Else, skips the vanilla Gaussian Splatting optimization and use the checkpoint in the provided directory.')
    
    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str,
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf", "density" or "dn_consistency". ' 
                        'We recommend using "dn_consistency" for the best mesh quality.')
    
    # Extract shell base
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000, 
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('--poisson_depth', type=int, default=-1, 
                        help="Depth of the octree for Poisson reconstruction. If -1, will compute automatically the depth based on the SuGaR model.")
    parser.add_argument('--cleaning_quantile', type=float, default=0.1, 
                        help='Quantile to use for cleaning the Poisson mesh. \
                            We recommend using 0.1 for real scenes and 0. for single-object synthetic scenes.')
    parser.add_argument('--connected_components_vis_th', type=float, default=0.001, 
                        help='Threshold to use for removing non-visible connected components in the mesh. \
                            We recommend using 0.001 for real scenes and 0.5 for single-object synthetic scenes.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    parser.add_argument('--project_mesh_on_surface_points', type=str2bool, default=True, 
                        help='If True, project the mesh on the surface points for better details.')
    
    # Parameters for Frosting
    # Render parameters
    parser.add_argument('--use_occlusion_culling', type=str2bool, default=False, 
                        help='If True, uses occlusion culling during training.')
    
    parser.add_argument('--learn_shell', type=str2bool, default=False, 
                        help='If True, also optimize the shell vertices. Should be False as this is useless in practice.')
    parser.add_argument('--regularize_shell', type=str2bool, default=False, 
                        help='If True, also regularize the base shell vertices with a normal consistency loss. Should be False as this is useless in practice.')
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
    
    # (Optional) File export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                        'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')
    parser.add_argument('--export_obj', type=str2bool, default=True, 
                        help='If True, export a textured mesh as an obj file for visualization and edition in Blender.')
    parser.add_argument('--texture_square_size', type=int, default=8, 
                        help='Size of the square allocated to each pair of triangles in the UV texture. Increase for higher texture resolution.')
    
    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False, 
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None, 
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")
      
    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=False, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')

    # Parse arguments
    args = parser.parse_args()
    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        print('Using low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        print('Using high poly config.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    if args.export_ply:
        print('Will export a ply file with the refined 3D Gaussians at the end of the training.')
    if args.export_obj:
        print('Will export a textured mesh as an obj file for visualization and edition in Blender.')
    
    # Output directory for the vanilla 3DGS checkpoint
    if args.gs_output_dir is None:
        sep = os.path.sep
        if len(args.scene_path.split(sep)[-1]) > 0:
            gs_checkpoint_dir = os.path.join("output", "vanilla_gs", args.scene_path.split(sep)[-1])
        else:
            gs_checkpoint_dir = os.path.join("output", "vanilla_gs", args.scene_path.split(sep)[-2])
        gs_checkpoint_dir = gs_checkpoint_dir + sep

        # Trains a 3DGS scene for 7k iterations
        white_background_str = '-w ' if args.white_background else ''
        # safety_command = " MKL_SERVICE_FORCE_INTEL=1"
        safety_command = ""  # TODO: Investigate why the MKL_SERVICE_FORCE_INTEL=1 flag is needed
        os.system(
            f"CUDA_VISIBLE_DEVICES={args.gpu}{safety_command} python ./gaussian_splatting/train.py \
                -s {args.scene_path} \
                -m {gs_checkpoint_dir} \
                {white_background_str}\
                --iterations 7_000"
        )
    else:
        print("A vanilla 3DGS checkpoint was provided. Skipping the vanilla 3DGS optimization.")
        gs_checkpoint_dir = args.gs_output_dir
        if gs_checkpoint_dir[-1] != os.path.sep:
            gs_checkpoint_dir += os.path.sep
    
    # Runs the train.py python script with the given arguments
    os.system(
        f"python train.py \
            -s {args.scene_path} \
            -c {gs_checkpoint_dir} \
            -r {args.regularization_type} \
            -l {args.surface_level} \
            -v {args.n_vertices_in_mesh} \
            --poisson_depth {args.poisson_depth} \
            --cleaning_quantile {args.cleaning_quantile} \
            --connected_components_vis_th {args.connected_components_vis_th} \
            --project_mesh_on_surface_points {args.project_mesh_on_surface_points} \
            --bboxmin {args.bboxmin} \
            --bboxmax {args.bboxmax} \
            --center_bbox {args.center_bbox} \
            --use_occlusion_culling {args.use_occlusion_culling} \
            --learn_shell {args.learn_shell} \
            --regularize_shell {args.regularize_shell} \
            --normal_consistency_factor {args.normal_consistency_factor} \
            --gaussians_in_frosting {args.gaussians_in_frosting} \
            --refinement_iterations {args.refinement_iterations} \
            --n_samples_per_vertex {args.n_samples_per_vertex} \
            --frosting_level {args.frosting_level} \
            --smooth_initial_frosting {args.smooth_initial_frosting} \
            --n_neighbors_for_smoothing {args.n_neighbors_for_smoothing} \
            --min_frosting_size {args.min_frosting_size} \
            --initial_proposal_std_range {args.initial_proposal_std_range} \
            --final_proposal_range {args.final_proposal_range} \
            --final_clamping_range {args.final_clamping_range} \
            --use_background_sphere {args.use_background_sphere} \
            --use_background_gaussians {args.use_background_gaussians} \
            --export_ply {args.export_ply} \
            --export_obj {args.export_obj} \
            --texture_square_size {args.texture_square_size} \
            --low_poly {args.low_poly} \
            --high_poly {args.high_poly} \
            --refinement_time {args.refinement_time} \
            --eval {args.eval} \
            --gpu {args.gpu} \
            --white_background {args.white_background}"
    )