import json
import torch
import numpy as np
from frosting_scene.cameras import CamerasWrapper, GSCamera, focal2fov, fov2focal
from frosting_scene.frosting_model import Frosting, load_frosting_model, convert_frosting_into_gaussians
from pytorch3d.transforms import (
    quaternion_to_matrix, 
    quaternion_invert,
    quaternion_multiply,
    matrix_to_quaternion,
    Transform3d
)


def load_blender_package(package_path, device):
    # Load package
    package = json.load(open(package_path))
    # Convert lists into tensors
    for key, object in package.items():
        if type(object) is dict:
            for sub_key, sub_object in object.items():
                if type(sub_object) is list:
                    object[sub_key] = torch.tensor(sub_object)
        elif type(object) is list:
            for element in object:
                if element:
                    for sub_key, sub_object in element.items():
                        if type(sub_object) is list:
                            element[sub_key] = torch.tensor(sub_object)
                            
    # Process bones
    bone_to_vertices = []
    bone_to_vertex_weights = []
    for i_mesh, mesh_dict in enumerate(package['bones']):
        if mesh_dict:
            vertex_dict = mesh_dict['vertex']
            armature_dict = mesh_dict['armature']
            
            # Per vertex info
            vertex_dict['matrix_world'] = torch.Tensor(vertex_dict['matrix_world']).to(device)
            vertex_dict['tpose_points'] = torch.Tensor(vertex_dict['tpose_points']).to(device)
            # vertex_dict['groups'] = np.array(vertex_dict['groups'])
            # vertex_dict['weights'] = torch.tensor(vertex_dict['weights']).to(device)
            
            # Per bone info
            armature_dict['matrix_world'] = torch.Tensor(armature_dict['matrix_world']).to(device)
            for key, val in armature_dict['rest_bones'].items():
                armature_dict['rest_bones'][key] = torch.Tensor(val).to(device)
            for key, val in armature_dict['pose_bones'].items():
                armature_dict['pose_bones'][key] = torch.Tensor(val).to(device)
                
            # Build mapping from bone name to corresponding vertices
            vertex_groups_idx = {}
            vertex_groups_weights = {}
            
            # > For each bone of the current armature, we initialize an empty list
            for bone_name in armature_dict['rest_bones']:
                vertex_groups_idx[bone_name] = []
                vertex_groups_weights[bone_name] = []
                
            # > For each vertex, we add the vertex index to the corresponding bone lists
            for i in range(len(vertex_dict['groups'])):
                # groups_in_which_vertex_appears = vertex_dict['groups'][i]
                # weights_of_the_vertex_in_those_groups = vertex_dict['weights'][i]
                groups_in_which_vertex_appears = []
                weights_of_the_vertex_in_those_groups = []

                # We start by filtering out the groups that are not part of the current armature.
                # This is necessary for accurately normalizing the weights.
                for j_group, group in enumerate(vertex_dict['groups'][i]):
                    if group in vertex_groups_idx:
                        groups_in_which_vertex_appears.append(group)
                        weights_of_the_vertex_in_those_groups.append(vertex_dict['weights'][i][j_group])
                
                # We normalize the weights
                normalize_weights = True
                if normalize_weights:
                    sum_of_weights = np.sum(weights_of_the_vertex_in_those_groups)
                    weights_of_the_vertex_in_those_groups = [w / sum_of_weights for w in weights_of_the_vertex_in_those_groups]
                
                # We add the vertex index and the associated weight to the corresponding bone lists
                for j_group, group in enumerate(groups_in_which_vertex_appears):
                    # For safety, we check that the group belongs to the current armature, used for rendering.
                    # Indeed, for editing purposes, one might want to use multiple armatures in the Blender scene, 
                    # but only one (as expected) for the final rendering.
                    if group in vertex_groups_idx:
                        vertex_groups_idx[group].append(i)
                        vertex_groups_weights[group].append(weights_of_the_vertex_in_those_groups[j_group])

            # > Convert the lists to tensors
            for bone_name in vertex_groups_idx:
                if len(vertex_groups_idx[bone_name]) > 0:
                    vertex_groups_idx[bone_name] = torch.tensor(vertex_groups_idx[bone_name], dtype=torch.long, device=device)
                    vertex_groups_weights[bone_name] = torch.tensor(vertex_groups_weights[bone_name], device=device)

            bone_to_vertices.append(vertex_groups_idx)
            bone_to_vertex_weights.append(vertex_groups_weights)
        else:
            bone_to_vertices.append(None)
            bone_to_vertex_weights.append(None)
    package['bone_to_vertices'] = bone_to_vertices
    package['bone_to_vertex_weights'] = bone_to_vertex_weights
    
    return package


def load_frosting_models_from_blender_package(package, device):
    frosting_models = {}
    scene_paths = []

    for mesh in package['meshes']:
        scene_path = mesh['checkpoint_name']
        if not scene_path in scene_paths:
            scene_paths.append(scene_path)

    for i_scene, scene_path in enumerate(scene_paths):        
        print(f'\nLoading Gaussians to bind: {scene_path}')
        frosting = load_frosting_model(scene_path, nerfmodel=None, device=device)
        frosting_models[scene_path] = frosting
    
    return frosting_models, scene_paths


def load_cameras_from_blender_package(package, device):
    matrix_world = package['camera']['matrix_world'].to(device)
    angle = package['camera']['angle']
    znear = package['camera']['clip_start']
    zfar = package['camera']['clip_end']
    
    if not 'image_height' in package['camera']:
        print('[WARNING] Image size not found in the package. Using default value 1920 x 1080.')
        height, width = 1080, 1920
    else:
        height, width = package['camera']['image_height'], package['camera']['image_width']

    gs_cameras = []
    for i_cam in range(len(angle)):
        c2w = matrix_world[i_cam]
        c2w[:3, 1:3] *= -1  # Blender to COLMAP convention
        w2c = c2w.inverse()
        R, T = w2c[:3, :3].transpose(-1, -2), w2c[:3, 3]  # R is stored transposed due to 'glm' in CUDA code
        
        fov = angle[i_cam].item()
        
        if width > height:
            fov_x = fov
            fov_y = focal2fov(fov2focal(fov_x, width), height)
        else:
            fov_y = fov
            fov_x = focal2fov(fov2focal(fov_y, height), width)
        
        gs_camera = GSCamera(
            colmap_id=str(i_cam), 
            R=R.cpu().numpy(), 
            T=T.cpu().numpy(), 
            FoVx=fov_x, 
            FoVy=fov_y,
            image=None, 
            gt_alpha_mask=None,
            image_name=f"frame_{i_cam}", 
            uid=i_cam,
            data_device=device,
            image_height=height,
            image_width=width,
        )
        gs_cameras.append(gs_camera)
    
    return CamerasWrapper(gs_cameras)


def build_composite_scene(
    frosting_models, 
    scene_paths, 
    package, 
    use_simple_adapt=False, 
    thickness_rescaling_method='median',
    initial_thickness_rescaling_method='triangle',
    deformation_threshold=None,
):
    device = frosting_models[scene_paths[0]].device
    sh_levels = frosting_models[scene_paths[0]].sh_levels
    
    comp_shell_base_verts = torch.zeros(0, 3, dtype=torch.float32, device=device)
    comp_shell_base_faces = torch.zeros(0, 3, dtype=torch.long, device=device)
    comp_outer_dist = torch.zeros(0, dtype=torch.float32, device=device)
    comp_inner_dist = torch.zeros(0, dtype=torch.float32, device=device)
    comp_point_cell_indices = torch.zeros(0, dtype=torch.long, device=device)
    comp_bary_coords = torch.zeros(0, 6, dtype=torch.float32, device=device)
    comp_scales = torch.zeros(0, 3, dtype=torch.float32, device=device)
    comp_quaternions = torch.zeros(0, 4, dtype=torch.float32, device=device)
    comp_opacities = torch.zeros(0, 1, dtype=torch.float32, device=device)
    comp_sh_dc = torch.zeros(0, 1, 3, dtype=torch.float32, device=device)
    comp_sh_rest = torch.zeros(0, sh_levels**2-1, 3, dtype=torch.float32, device=device)
    
    comp_bg_points = torch.zeros(0, 3, dtype=torch.float32, device=device)
    comp_bg_opacities = torch.zeros(0, 1, dtype=torch.float32, device=device)
    comp_bg_scales = torch.zeros(0, 3, dtype=torch.float32, device=device)
    comp_bg_quaternions = torch.zeros(0, 4, dtype=torch.float32, device=device)
    comp_bg_sh_dc = torch.zeros(0, 1, 3, dtype=torch.float32, device=device)
    comp_bg_sh_rest = torch.zeros(0, sh_levels**2-1, 3, dtype=torch.float32, device=device)
    
    original_quaternions = torch.zeros(0, 4, dtype=torch.float32, device=device)
    original_bg_quaternions = torch.zeros(0, 4, dtype=torch.float32, device=device)
    
    bg_frosting_name = ''
    n_gaussians_in_bg_frosting = 0
    bg_idx = 0
    
    final_edition_mask = torch.zeros(0, dtype=torch.bool, device=device)

    # Build the composited surface mesh
    with torch.no_grad():
        # Background
        for i, mesh_dict in enumerate(package['meshes']):
            scene_name = mesh_dict['checkpoint_name']
            frosting = frosting_models[scene_name]
            mesh_is_animated = package['bones'][i] is not None
            
            # Full mesh from the original checkpoint        
            original_mesh = frosting.shell_base
            original_verts = original_mesh.verts_list()[0]
            original_faces = original_mesh.faces_list()[0]
            
            with torch.no_grad():
                all_faces_indices = torch.arange(0, original_mesh.faces_list()[0].shape[0], dtype=torch.long, device=device)
                filtered_mesh = original_mesh.submeshes([[all_faces_indices]])

            filtered_verts_to_verts_idx = - torch.ones(filtered_mesh.verts_list()[0].shape[0], dtype=torch.long, device=device)
            filtered_verts_to_verts_idx[filtered_mesh.faces_list()[0]] = original_mesh.faces_list()[0]
            
            # Segmented mesh parameters
            vert_idx = filtered_verts_to_verts_idx[mesh_dict['idx']]
            keep_verts_mask = torch.zeros(original_verts.shape[0], dtype=torch.bool, device=device)
            keep_verts_mask[vert_idx] = True
            keep_faces_mask = keep_verts_mask[original_faces].all(dim=1)
            keep_faces_indices = keep_faces_mask.nonzero()[..., 0]

            old_verts_to_new_verts_match = -torch.ones(original_verts.shape[0], dtype=torch.long, device=device)
            old_verts_to_new_verts_match[vert_idx] = torch.arange(vert_idx.shape[0], device=device)
            old_faces_to_new_faces_match = -torch.ones(original_faces.shape[0], dtype=torch.long, device=device)
            old_faces_to_new_faces_match[keep_faces_indices] = torch.arange(keep_faces_indices.shape[0], device=device)

            new_verts = original_verts[vert_idx]
            new_faces = old_verts_to_new_verts_match[original_faces[keep_faces_indices]]
            new_gaussians_mask = keep_faces_mask[frosting._point_cell_indices]
            
            frosting_shell_base_verts = new_verts
            frosting_shell_base_faces = new_faces
            frosting_outer_dist = frosting._outer_dist[vert_idx]
            frosting_inner_dist = frosting._inner_dist[vert_idx]
            frosting_point_cell_indices = old_faces_to_new_faces_match[frosting._point_cell_indices[new_gaussians_mask]]
            frosting_bary_coords = frosting._bary_coords[new_gaussians_mask]
            frosting_scales = frosting._scales[new_gaussians_mask]
            frosting_quaternions = frosting._quaternions[new_gaussians_mask]
            frosting_opacities = frosting._opacities[new_gaussians_mask]
            frosting_sh_dc = frosting._sh_coordinates_dc[new_gaussians_mask]
            frosting_sh_rest = frosting._sh_coordinates_rest[new_gaussians_mask]
            frosting_original_quaternions = frosting.quaternions[new_gaussians_mask]
            n_gaussians_in_this_frosting = len(frosting_bary_coords)
            
            # Adjust indices
            frosting_shell_base_faces = frosting_shell_base_faces + comp_shell_base_verts.shape[0]
            frosting_point_cell_indices = frosting_point_cell_indices + comp_shell_base_faces.shape[0]
            
            # Update composited parameters
            comp_shell_base_verts = torch.cat([comp_shell_base_verts, frosting_shell_base_verts], dim=0)
            comp_shell_base_faces = torch.cat([comp_shell_base_faces, frosting_shell_base_faces], dim=0)
            # TODO: Adjust inner and outer dists based on the new scaling of each individual mesh
            comp_outer_dist = torch.cat([comp_outer_dist, frosting_outer_dist], dim=0)
            comp_inner_dist = torch.cat([comp_inner_dist, frosting_inner_dist], dim=0)
            comp_point_cell_indices = torch.cat([comp_point_cell_indices, frosting_point_cell_indices], dim=0)
            comp_bary_coords = torch.cat([comp_bary_coords, frosting_bary_coords], dim=0)
            comp_scales = torch.cat([comp_scales, frosting_scales], dim=0)
            comp_quaternions = torch.cat([comp_quaternions, frosting_quaternions], dim=0)
            comp_opacities = torch.cat([comp_opacities, frosting_opacities], dim=0)
            comp_sh_dc = torch.cat([comp_sh_dc, frosting_sh_dc], dim=0)
            comp_sh_rest = torch.cat([comp_sh_rest, frosting_sh_rest], dim=0)
            original_quaternions = torch.cat([original_quaternions, frosting_original_quaternions], dim=0)
            
            if mesh_is_animated:
                final_edition_mask = torch.cat([final_edition_mask, torch.ones(frosting_shell_base_faces.shape[0], dtype=torch.bool, device=device)], dim=0)
            else:
                final_edition_mask = torch.cat([final_edition_mask, torch.zeros(frosting_shell_base_faces.shape[0], dtype=torch.bool, device=device)], dim=0)
            
            if frosting.use_background_gaussians:
                n_gaussians_in_this_frosting = n_gaussians_in_this_frosting + len(frosting._bg_points)
                
            if n_gaussians_in_this_frosting >= n_gaussians_in_bg_frosting:
                bg_frosting_name = scene_name
                n_gaussians_in_bg_frosting = n_gaussians_in_this_frosting
                bg_idx = i
                if frosting.use_background_gaussians:
                    comp_bg_points = frosting._bg_points
                    comp_bg_opacities = frosting._bg_opacities
                    comp_bg_scales = frosting._bg_scales
                    comp_bg_quaternions = frosting._bg_quaternions
                    comp_bg_sh_dc = frosting._bg_sh_coordinates_dc
                    comp_bg_sh_rest = frosting._bg_sh_coordinates_rest
                    original_bg_quaternions = frosting.bg_quaternions
                else:
                    comp_bg_points = torch.zeros(0, 3, dtype=torch.float32, device=device)
                    comp_bg_opacities = torch.zeros(0, 1, dtype=torch.float32, device=device)
                    comp_bg_scales = torch.zeros(0, 3, dtype=torch.float32, device=device)
                    comp_bg_quaternions = torch.zeros(0, 4, dtype=torch.float32, device=device)
                    comp_bg_sh_dc = torch.zeros(0, 1, 3, dtype=torch.float32, device=device)
                    comp_bg_sh_rest = torch.zeros(0, sh_levels**2-1, 3, dtype=torch.float32, device=device)
                    original_bg_quaternions = torch.zeros(0, 4, dtype=torch.float32, device=device)
                
    frosting_comp = frosting_models[bg_frosting_name]
    
    # Update parameters
    with torch.no_grad():
        frosting_comp._shell_base_verts = torch.nn.Parameter(comp_shell_base_verts, requires_grad=False).to(device)
        frosting_comp._shell_base_faces = torch.nn.Parameter(comp_shell_base_faces, requires_grad=False).to(device)
        frosting_comp._outer_dist = torch.nn.Parameter(comp_outer_dist, requires_grad=False).to(device)
        frosting_comp._inner_dist = torch.nn.Parameter(comp_inner_dist, requires_grad=False).to(device)
        frosting_comp._point_cell_indices = torch.nn.Parameter(comp_point_cell_indices, requires_grad=False).to(device)
        frosting_comp._bary_coords = torch.nn.Parameter(comp_bary_coords, requires_grad=False).to(device)
        frosting_comp._scales = torch.nn.Parameter(comp_scales, requires_grad=False).to(device)
        frosting_comp._quaternions = torch.nn.Parameter(comp_quaternions, requires_grad=False).to(device)
        frosting_comp._opacities = torch.nn.Parameter(comp_opacities, requires_grad=False).to(device)
        frosting_comp._sh_coordinates_dc = torch.nn.Parameter(comp_sh_dc, requires_grad=False).to(device)
        frosting_comp._sh_coordinates_rest = torch.nn.Parameter(comp_sh_rest, requires_grad=False).to(device)
        frosting_comp.original_quaternions = original_quaternions
        if frosting_comp.use_background_gaussians:
            frosting_comp._bg_points = torch.nn.Parameter(comp_bg_points, requires_grad=False).to(device)
            frosting_comp._bg_opacities = torch.nn.Parameter(comp_bg_opacities, requires_grad=False).to(device)
            frosting_comp._bg_scales = torch.nn.Parameter(comp_bg_scales, requires_grad=False).to(device)
            frosting_comp._bg_quaternions = torch.nn.Parameter(comp_bg_quaternions, requires_grad=False).to(device)
            frosting_comp._bg_sh_coordinates_dc = torch.nn.Parameter(comp_bg_sh_dc, requires_grad=False).to(device)
            frosting_comp._bg_sh_coordinates_rest = torch.nn.Parameter(comp_bg_sh_rest, requires_grad=False).to(device)
            frosting_comp.original_bg_quaternions = original_bg_quaternions
        
    # ===== Method 1 =====
    # Simply adjusting the scene with the desired thickness rescaling method.
    if False:
        frosting_comp.make_editable(
            use_simple_adapt=use_simple_adapt,
            thickness_rescaling_method=thickness_rescaling_method,
        )  # TODO: make_editable should be called before modifying the vertices? Check.

        # Update mesh vertices
        start_idx = 0
        end_idx = 0
        ref_indices = []
        for i, mesh_dict in enumerate(package['meshes']):
            ref_indices.append(start_idx)
            edited_verts = mesh_dict['xyz']
            matrix_world = mesh_dict['matrix_world'].to(device).transpose(-1, -2)
            world_transform = Transform3d(matrix=matrix_world, device=device)
            end_idx += len(edited_verts)
            
            with torch.no_grad():
                frosting_comp._shell_base_verts[start_idx:end_idx] = world_transform.transform_points(edited_verts.to(device))
            start_idx = end_idx + 0
            
            if i == bg_idx:
                bg_transform = world_transform
                
        ref_indices.append(len(frosting_comp._shell_base_verts))
        frosting_comp.ref_indices = ref_indices
        
        # Update background gaussians
        if frosting_comp.use_background_gaussians:
            with torch.no_grad():
                frosting_comp._bg_points[...] = bg_transform.transform_points(frosting_comp._bg_points.to(device))
                frosting_comp._bg_quaternions[...] = quaternion_multiply(
                    matrix_to_quaternion(
                        world_transform.get_matrix()[..., :3, :3].transpose(-1, -2)
                    ), 
                    frosting_comp.bg_quaternions
                )
    # ===== Method 2 =====
    # First apply triangle rescaling for accurate initial scaling of the scene, 
    # then apply the desired thickness rescaling method for further edition.
    else:
        frosting_comp.make_editable(
            use_simple_adapt=use_simple_adapt,
            thickness_rescaling_method=initial_thickness_rescaling_method,
        )  

        # Update mesh vertices with triangle mode and initial global transform to get accurate scaling
        start_idx = 0
        end_idx = 0
        ref_indices = []
        for i, mesh_dict in enumerate(package['meshes']):
            ref_indices.append(start_idx)
            matrix_world = mesh_dict['matrix_world'].to(device).transpose(-1, -2)
            world_transform = Transform3d(matrix=matrix_world, device=device)
            end_idx += len(mesh_dict['xyz'])
            
            with torch.no_grad():
                frosting_comp._shell_base_verts[start_idx:end_idx] = world_transform.transform_points(
                    frosting_comp._shell_base_verts[start_idx:end_idx].to(device)
                )
            start_idx = end_idx + 0
            
            if i == bg_idx:
                bg_transform = world_transform
                
        ref_indices.append(len(frosting_comp._shell_base_verts))
        frosting_comp.ref_indices = ref_indices
        
        # Update background gaussians
        if frosting_comp.use_background_gaussians:
            with torch.no_grad():
                bary_center = frosting_comp._bg_points.mean(dim=0, keepdim=True)
                old_bary_scaling = (frosting_comp._bg_points - bary_center).norm(dim=-1).median().item()
                frosting_comp._bg_points[...] = bg_transform.transform_points(frosting_comp._bg_points.to(device))
                frosting_comp._bg_quaternions[...] = quaternion_multiply(
                    matrix_to_quaternion(
                        bg_transform.get_matrix()[..., :3, :3].transpose(-1, -2)
                    ), 
                    frosting_comp.bg_quaternions
                )
                bary_center = frosting_comp._bg_points.mean(dim=0, keepdim=True)
                new_bary_scaling = (frosting_comp._bg_points - bary_center).norm(dim=-1).median().item()
                bary_scaling = new_bary_scaling / old_bary_scaling
                print(f"Rescaling background Gaussians with factor: {bary_scaling}")
                frosting_comp._bg_scales[...] = frosting_comp._bg_scales + np.log(bary_scaling)
                
        # Save updates to keep accurate initial rescaling of the thickness
        frosting_comp.update_parameters_from_edited_mesh()  
                
        # Applying finer editions with the desired thickness rescaling method 
        start_idx = 0
        end_idx = 0
        for i, mesh_dict in enumerate(package['meshes']):
            edited_verts = mesh_dict['xyz']
            matrix_world = mesh_dict['matrix_world'].to(device).transpose(-1, -2)
            world_transform = Transform3d(matrix=matrix_world, device=device)
            end_idx += len(edited_verts)
            
            with torch.no_grad():
                # Make frostings editable only for the current mesh
                edition_mask_i = (
                    (frosting_comp._shell_base_faces >= start_idx) 
                    & (frosting_comp._shell_base_faces < end_idx)
                ).all(dim=1)
                frosting_comp.make_editable(
                    edition_mask=edition_mask_i,
                    use_simple_adapt=use_simple_adapt,
                    thickness_rescaling_method=thickness_rescaling_method,
                )
                
                # Change the vertices of the current mesh
                frosting_comp._shell_base_verts[start_idx:end_idx] = world_transform.transform_points(edited_verts.to(device))
                
                # Remove parts with too much deformation
                if deformation_threshold is not None:
                    deformation_mask = torch.ones(frosting_comp._scales.shape[0], dtype=torch.bool, device=frosting_comp.device)
                    deformation_mask[frosting_comp._gaussian_edition_mask] = _get_edited_points_deformation_mask(frosting_comp, threshold=deformation_threshold)
                    frosting_comp._opacities[~deformation_mask] = -1e10
                
                # Update the parameters of the frosting model
                frosting_comp.update_parameters_from_edited_mesh()
            
            start_idx = end_idx + 0
            
    # If some animations are present, make the mesh editable only for the animated parts
    if final_edition_mask.sum() > 0:
        print('Animated parts found.')
        frosting_comp.make_editable(
            edition_mask=final_edition_mask,
            use_simple_adapt=use_simple_adapt,
            thickness_rescaling_method=thickness_rescaling_method,
        )
    else:
        print('No animated parts found.')
    
    return frosting_comp


def apply_poses_to_scene(
    frosting_comp:Frosting, 
    i_frame:int, 
    package:dict,
):
    n_frames = len(package['camera']['lens'])
    bone_to_vertices = package['bone_to_vertices']
    bone_to_vertex_weights = package['bone_to_vertex_weights']

    if frosting_comp.editable and (not frosting_comp.use_simple_adapt):
        frosting_comp.edited_cache = None
    
    with torch.no_grad():
        for i_mesh, mesh_dict in enumerate(package['bones']):
            if mesh_dict:
                start_idx, end_idx = frosting_comp.ref_indices[i_mesh], frosting_comp.ref_indices[i_mesh+1]
                vertex_groups_idx = bone_to_vertices[i_mesh]
                vertex_groups_weights = bone_to_vertex_weights[i_mesh]
                
                tpose_points = mesh_dict['vertex']['tpose_points']
                
                use_weighting = True
                # TODO: Use weight formula for vertex with multiple groups. Just add the weighted transforms, and normalize at the end.
                if use_weighting:
                    new_points = torch.zeros_like(tpose_points)
                else:
                    new_points = tpose_points.clone().to(frosting_comp.device)
                            
                for vertex_group, vertex_group_idx in vertex_groups_idx.items():
                    if len(vertex_group_idx) > 0:
                        # Build bone transform
                        bone_transform = Transform3d(matrix=mesh_dict['armature']['pose_bones'][vertex_group][i_frame % n_frames].transpose(-1, -2))
                        reset_transform = Transform3d(matrix=mesh_dict['armature']['rest_bones'][vertex_group].transpose(-1, -2)).inverse()
                        
                        # Transform points
                        if use_weighting:
                            # weights = torch.tensor(vertex_groups_weights[vertex_group], device=frosting_comp.device)
                            weights = vertex_groups_weights[vertex_group]
                            new_points[vertex_group_idx] += weights[..., None] * bone_transform.transform_points(reset_transform.transform_points(tpose_points[vertex_group_idx]))
                        else:
                            new_points[vertex_group_idx] = bone_transform.transform_points(reset_transform.transform_points(tpose_points[vertex_group_idx]))

                frosting_comp.shell_base.verts_list()[0][start_idx:end_idx] = new_points
                
                
def get_frosting_sh_rotations(frosting:Frosting):
    quaternions = frosting.quaternions
    original_quaternions = frosting.original_quaternions
    if frosting.use_background_gaussians:
        quaternions = torch.cat([quaternions, frosting.bg_quaternions], dim=0)
        original_quaternions = torch.cat([original_quaternions, frosting.original_bg_quaternions], dim=0)
    
    sh_rotations = quaternion_to_matrix(
        quaternion_multiply(
            quaternions,
            quaternion_invert(original_quaternions),
        )
    )
    if frosting.editable and (not frosting.use_simple_adapt):
        frosting.edited_cache = None
    return sh_rotations


def _get_edited_points_deformation_mask(frosting:Frosting, threshold:float=2.):
    _reference_verts = frosting._reference_verts[frosting._shell_base_faces]
    ref_dists = (_reference_verts - _reference_verts.mean(dim=-2, keepdim=True)).norm(dim=-1)
    
    _new_verts = frosting._shell_base_verts[frosting._shell_base_faces]  # n_faces, 3, 3
    new_dists = (_new_verts - _new_verts.mean(dim=-2, keepdim=True)).norm(dim=-1)
    
    ratios = (new_dists / ref_dists).max(dim=-1)[0]

    render_mask = ratios[frosting._point_cell_indices][frosting._gaussian_edition_mask] <= threshold
    
    return render_mask


def render_composited_image(
    package:dict,
    frosting:Frosting, 
    render_cameras:CamerasWrapper, 
    i_frame:int,
    sh_degree:int=None,
    deformation_threshold:float=2.,
    use_occlusion_culling:bool=False,
    return_GS_model:bool=False,
    ):
    
    use_sh = (sh_degree is None) or sh_degree > 0

    apply_poses_to_scene(frosting, i_frame, package)
    
    if frosting.editable:
        deformation_mask = torch.ones(frosting._scales.shape[0], dtype=torch.bool, device=frosting.device)
        deformation_mask[frosting._gaussian_edition_mask] = _get_edited_points_deformation_mask(frosting, threshold=deformation_threshold)
        
        splat_opacities = frosting.strengths.view(-1, 1)
        splat_opacities[~deformation_mask] = 0.
        if frosting.use_background_gaussians:
            splat_opacities = torch.cat([splat_opacities, frosting.bg_strengths.view(-1, 1)], dim=0)
    else:
        splat_opacities = None
    
    if return_GS_model:
        return convert_frosting_into_gaussians(frosting, opacities=splat_opacities)
    else:
        return frosting.render_image_gaussian_rasterizer(
            nerf_cameras=render_cameras,
            camera_indices=i_frame,
            sh_deg=sh_degree,
            compute_color_in_rasterizer=not use_sh,
            point_opacities=splat_opacities,
            sh_rotations=None if not use_sh else get_frosting_sh_rotations(frosting),
            use_occlusion_culling=use_occlusion_culling,
        )
    