import torch
import numpy as np
from frosting_utils.general_utils import get_expon_lr_func
from .frosting_model import Frosting


class OptimizationParams():
    def __init__(
        self, 
        iterations:int=30_000,
        position_lr_init:float=0.00016,  # 0.00016 in original 3DGS
        position_lr_final:float=0.0000016,  # 0.0000016 in original 3DGS
        position_bary_coords_lr_init:float=0.005,
        position_bary_coords_lr_final:float=0.00005,
        position_lr_delay_mult:float=0.01,
        position_lr_max_steps:int=30_000,
        feature_lr:float=0.0025,
        opacity_lr:float=0.05,
        scaling_lr:float=0.005,
        rotation_lr:float=0.001,
        ):
        
        # Basic Gaussian Splatting
        self.iterations = iterations
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_bary_coords_lr_init = position_bary_coords_lr_init
        self.position_bary_coords_lr_final = position_bary_coords_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.position_lr_max_steps = position_lr_max_steps
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr

    def __str__(self):
        return f"""OptimizationParams(
            iterations={self.iterations},
            position_lr_init={self.position_lr_init},
            position_lr_final={self.position_lr_final},
            position_bary_coords_lr_init={self.position_bary_coords_lr_init},
            position_bary_coords_lr_final={self.position_bary_coords_lr_final},
            position_lr_delay_mult={self.position_lr_delay_mult},
            position_lr_max_steps={self.position_lr_max_steps},
            feature_lr={self.feature_lr},
            opacity_lr={self.opacity_lr},
            scaling_lr={self.scaling_lr},
            rotation_lr={self.rotation_lr},
            )"""


class FrostingOptimizer():
    """Wrapper of the Adam optimizer used for Frosting optimization.
    Largely inspired by the original implementation of the 3D Gaussian Splatting paper:
    https://github.com/graphdeco-inria/gaussian-splatting
    """
    def __init__(
        self,
        model:Frosting,
        opt:OptimizationParams=None,
        spatial_lr_scale:float=None,
        ) -> None:

        self.current_iteration = 0
        self.num_iterations = opt.iterations
        
        if opt is None:
            opt = OptimizationParams()
        
        if spatial_lr_scale is None:
            spatial_lr_scale = model.get_cameras_spatial_extent()
        self.spatial_lr_scale = spatial_lr_scale
        
        l = []
        if model.learn_shell:
            # l = l + [{'params': [model._shell_base_verts], 'lr': opt.position_lr_init * spatial_lr_scale, "name": "shell_base_verts"}]
            l = l + [{'params': [model._inner_dist], 'lr': opt.position_lr_init * spatial_lr_scale, "name": "inner_dist"}]
            l = l + [{'params': [model._outer_dist], 'lr': opt.position_lr_init * spatial_lr_scale, "name": "outer_dist"}]
        l = l + [{'params': [model._bary_coords], 'lr': opt.position_bary_coords_lr_init, "name": "bary_coords"}]
        l = l + [{'params': [model._sh_coordinates_dc], 'lr': opt.feature_lr, "name": "sh_coordinates_dc"},
                 {'params': [model._sh_coordinates_rest], 'lr': opt.feature_lr / 20.0, "name": "sh_coordinates_rest"}]
        l = l + [{'params': [model._opacities], 'lr': opt.opacity_lr, "name": "opacities"}]
        l = l + [{'params': [model._scales], 'lr': opt.scaling_lr, "name": "scales"}]
        l = l + [{'params': [model._quaternions], 'lr': opt.rotation_lr, "name": "quaternions"}]
        
        if model.use_background_gaussians:
            l = l + [{'params': [model._bg_points], 'lr': opt.position_lr_init * spatial_lr_scale, "name": "bg_points"}]
            l = l + [{'params': [model._bg_opacities], 'lr': opt.opacity_lr, "name": "bg_opacities"}]
            l = l + [{'params': [model._bg_sh_coordinates_dc], 'lr': opt.feature_lr, "name": "bg_sh_coordinates_dc"}]
            l = l + [{'params': [model._bg_sh_coordinates_rest], 'lr': opt.feature_lr / 20.0, "name": "bg_sh_coordinates_rest"}]
            l = l + [{'params': [model._bg_scales], 'lr': opt.scaling_lr, "name": "bg_scales"}]
            l = l + [{'params': [model._bg_quaternions], 'lr': opt.rotation_lr, "name": "bg_quaternions"}]

        if model.use_background_sphere:
            l = l + [{'params': [model._bg_sphere_opacities], 'lr': opt.opacity_lr, "name": "bg_sphere_opacities"}]
            l = l + [{'params': [model._bg_sphere_sh_coordinates_dc], 'lr': opt.feature_lr, "name": "bg_sphere_sh_coordinates_dc"}]
            l = l + [{'params': [model._bg_sphere_sh_coordinates_rest], 'lr': opt.feature_lr / 20.0, "name": "bg_sphere_sh_coordinates_rest"}]
            l = l + [{'params': [model._bg_sphere_scales], 'lr': opt.scaling_lr, "name": "bg_sphere_scales"}]
            l = l + [{'params': [model._bg_sphere_complex], 'lr': opt.rotation_lr, "name": "bg_sphere_complex"}]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.position_sheduler_func = get_expon_lr_func(
            lr_init=opt.position_lr_init * spatial_lr_scale, 
            lr_final=opt.position_lr_final * spatial_lr_scale, 
            lr_delay_mult=opt.position_lr_delay_mult, 
            max_steps=opt.position_lr_max_steps
            )
        self.position_bary_coords_sheduler_func = get_expon_lr_func(
            lr_init=opt.position_bary_coords_lr_init, 
            lr_final=opt.position_bary_coords_lr_final, 
            lr_delay_mult=opt.position_lr_delay_mult, 
            max_steps=opt.position_lr_max_steps
            )
        
    def step(self):
        self.optimizer.step()
        self.current_iteration += 1
        
    def zero_grad(self, set_to_none:bool=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        
    def update_learning_rate(self, iteration:int=None):
        if iteration is None:
            iteration = self.current_iteration
        lr = 0.
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["shell_base_verts", "inner_dist", "outer_dist", "bg_points"]:
                lr = self.position_sheduler_func(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "bary_coords":
                lr = self.position_bary_coords_sheduler_func(iteration)
                param_group['lr'] = lr
        return lr
            
    def add_param_group(self, new_param_group):
        self.optimizer.add_param_group(new_param_group)

    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
