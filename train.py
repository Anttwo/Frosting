import argparse
from frosting_utils.general_utils import str2bool
from frosting_trainers.coarse_density import coarse_training_with_density_regularization
from frosting_trainers.coarse_sdf import coarse_training_with_sdf_regularization
from frosting_trainers.coarse_density_and_dn_consistency import coarse_training_with_density_regularization_and_dn_consistency
from frosting_extractors.coarse_shell import extract_shell_base_from_coarse_sugar
from frosting_trainers.refine import refined_training


class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full Frosting model.')
    
    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        help='(Required) path to the scene data to use.')  
    parser.add_argument('-c', '--checkpoint_path',
                        type=str, 
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
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
                        help='Quantile to use for cleaning the Poisson mesh.')
    parser.add_argument('--connected_components_vis_th', type=float, default=0.001, 
                        help='Threshold to use for removing non-visible connected components in the mesh. \
                            We recommend using 0.001 for real scenes and 0.5 for single-object synthetic scenes.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None, 
                        help='Min coordinates to use for foreground.')  
    parser.add_argument('-B', '--bboxmax', type=str, default=None, 
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True, 
                        help='If True, center the bbox. Default is False.')
    parser.add_argument('--project_mesh_on_surface_points', 
                        type=str2bool, default=True, 
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
    # TODO: Add a regularization term on the inner and outer dists?
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
    
    # ----- Optimize coarse SuGaR -----
    coarse_args = AttrDict({
        'checkpoint_path': args.checkpoint_path,
        'scene_path': args.scene_path,
        'iteration_to_load': args.iteration_to_load,
        'output_dir': None,
        'eval': args.eval,
        'estimation_factor': 0.2,
        'normal_factor': 0.2,
        'gpu': args.gpu,
        'white_background': args.white_background,
    })
    if args.regularization_type == 'sdf':
        coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args)
    elif args.regularization_type == 'density':
        coarse_sugar_path = coarse_training_with_density_regularization(coarse_args)
    elif args.regularization_type == 'dn_consistency':
        coarse_sugar_path = coarse_training_with_density_regularization_and_dn_consistency(coarse_args)
    else:
        raise ValueError(f'Unknown regularization type: {args.regularization_type}')
    
    
    # ----- Extract shell base from coarse SuGaR -----
    shell_base_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'iteration_to_load': args.iteration_to_load,
        'coarse_model_path': coarse_sugar_path,
        'surface_level': args.surface_level,
        'decimation_target': args.n_vertices_in_mesh,
        'poisson_depth': args.poisson_depth,
        'cleaning_quantile': args.cleaning_quantile,
        'connected_components_vis_th': args.connected_components_vis_th,
        'project_mesh_on_surface_points': args.project_mesh_on_surface_points,
        'mesh_output_dir': None,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'center_bbox': args.center_bbox,
        'gpu': args.gpu,
        'eval': args.eval,
        'use_centers_to_extract_mesh': False,
        'use_marching_cubes': False,
        'use_vanilla_3dgs': False,
    })
    shell_base_path = extract_shell_base_from_coarse_sugar(shell_base_args)[0]
    
    
    # ----- Optimize Frosting -----
    frosting_args = AttrDict({
        'scene_path': args.scene_path,
        'checkpoint_path': args.checkpoint_path,
        'sugar_path': coarse_sugar_path,
        'mesh_path': shell_base_path,
        'output_dir': None,
        'iteration_to_load': args.iteration_to_load,
        'learn_shell': args.learn_shell,
        'use_occlusion_culling': args.use_occlusion_culling,
        'regularize_shell': args.regularize_shell,
        'normal_consistency_factor': args.normal_consistency_factor,    
        'gaussians_in_frosting': args.gaussians_in_frosting,        
        # 'n_vertices_in_fg': args.n_vertices_in_mesh,
        'refinement_iterations': args.refinement_iterations,
        'min_frosting_factor': args.min_frosting_factor,  # Deprecated
        'max_frosting_factor': args.max_frosting_factor,  # Deprecated
        'min_frosting_range': args.min_frosting_range,  # Deprecated
        'n_samples_per_vertex': args.n_samples_per_vertex,
        'frosting_level': args.frosting_level,
        'smooth_initial_frosting': args.smooth_initial_frosting,
        'n_neighbors_for_smoothing': args.n_neighbors_for_smoothing,
        'min_frosting_size': args.min_frosting_size,
        'initial_proposal_std_range': args.initial_proposal_std_range,
        'final_proposal_range': args.final_proposal_range,
        'final_clamping_range': args.final_clamping_range,
        'use_background_sphere': args.use_background_sphere,
        'use_background_gaussians': args.use_background_gaussians,
        'bboxmin': args.bboxmin,
        'bboxmax': args.bboxmax,
        'export_ply': args.export_ply,
        'export_obj': args.export_obj,
        'texture_square_size': args.texture_square_size,
        'eval': args.eval,
        'gpu': args.gpu,
        'white_background': args.white_background,
    })
    frosting_path = refined_training(frosting_args)
        