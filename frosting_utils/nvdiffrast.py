import torch
import nvdiffrast.torch as dr
from frosting_scene.cameras import CamerasWrapper, P3DCameras, GSCamera
from pytorch3d.structures import Meshes
from typing import Union


def nvdiff_rasterization(
    camera:Union[P3DCameras, GSCamera],
    image_height:int, image_width:int,
    mesh=None, verts:torch.Tensor=None, faces:torch.Tensor=None,
    return_indices_only:bool=False,
    glctx=None,
):
    mesh_is_provided = mesh is not None
    verts_and_faces_are_provided = (verts is not None) and (faces is not None)
    if (not mesh_is_provided) and (not verts_and_faces_are_provided):
        raise ValueError('Either mesh or verts and faces must be provided')
    if verts is None:
        verts = mesh.verts_list()[0].float()
    if faces is None:
        faces = mesh.faces_list()[0].int()
    device = verts.device
    
    if isinstance(camera, P3DCameras):
        # TODO [WARNING]: Code below can be wrong, check it.
        # Please use GS camera for now
        
        # Get full projection matrix    
        camera_mtx = camera.get_full_projection_transform().get_matrix()[0]

        # Convert to homogeneous coordinates
        pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device)], axis=1)
        
        # Transform points to NDC/clip space
        pos = torch.matmul(pos, camera_mtx)[None, ...]
        pos[..., :3] = -pos[..., :3]
        # pos = pos / pos[..., 3:]
        img_factors = min(image_height, image_width) / torch.tensor([[[image_width, image_height]]], dtype=torch.float32, device=device)
        pos[..., :2] = pos[..., :2] * img_factors
        
    elif isinstance(camera, GSCamera):
        # Get full projection matrix
        camera_mtx = camera.full_proj_transform
        
        # Convert to homogeneous coordinates
        pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device)], axis=1)
        
        # Transform points to NDC/clip space
        pos = torch.matmul(pos, camera_mtx)[None]
    
    # Rasterize with NVDiffRast
    rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[image_height, image_width])
    bary_coords, zbuf, pix_to_face = rast_out[..., :2], rast_out[..., 2], rast_out[..., 3].int()
    
    if return_indices_only:
        return pix_to_face
    return bary_coords, zbuf, pix_to_face


def nvdiff_rasterization_with_pix_to_face(
    mesh:Meshes, 
    cameras:CamerasWrapper, 
    cam_idx:int=0,
    glctx=None,
    ):
    image_height = cameras.gs_cameras[cam_idx].image_height
    image_width = cameras.gs_cameras[cam_idx].image_width
    pix_to_face = nvdiff_rasterization(
        # cameras.p3d_cameras[cam_idx],
        cameras.gs_cameras[cam_idx],
        image_height, image_width,
        mesh=mesh,
        return_indices_only=True,
        glctx=glctx,
    )
    face_idx_to_render = pix_to_face.unique() - 1
    return face_idx_to_render
