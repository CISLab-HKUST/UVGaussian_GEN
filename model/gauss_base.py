import os
import torch
import torch.nn as nn
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.sh_utils import eval_sh, RGB2SH
from gaussian_renderer import render

class GaussianBase(nn.Module):
    def __init__(self, sh_degree=0) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
    
    # render
    def render_to_camera(self, viewpoint_cam, pipe, background=None, scaling_modifer=1.0):
        if background == 'white':
            background = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
        else:
            background = torch.rand((3,), dtype=torch.float32, device='cuda')

        out = render(viewpoint_cam, self, pipe, background, scaling_modifer)

        # if hasattr(viewpoint_cam, 'original_image'):
        #     if hasattr(viewpoint_cam, 'gt_alpha_mask'):
        #         gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        #         gt_image = viewpoint_cam.original_image.cuda()
        #         gt_image = gt_image * gt_alpha_mask + background[:, None, None] * (1 - gt_alpha_mask)

        #         out.update({
        #             'gt_image': gt_image,
        #             'gt_alpha_mask': gt_alpha_mask,
        #         })
        #     else:
        #         gt_image = viewpoint_cam.original_image.cuda()
        #         out.update({
        #             'gt_image': gt_image,
        #         })
        

        return out
    
