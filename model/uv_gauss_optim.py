import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dreams import libcore
from dreams.bone_deformer.smplx_optim import SMPLXOptimizer
from utils.loss_utils import l1_loss, ssim, LPIPS
from utils.image_utils import psnr
from simple_knn._C import distCUDA2
from tqdm import tqdm

class UVGaussOptimizer(nn.Module):
    def __init__(self, uvgs_model, optimizer_config=None) -> None:
        super().__init__()
        self.uvgs_model = uvgs_model
        self.optimizer = None
        self.smplx_optim = None
        self.schedulers = []

        if optimizer_config is not None:
            self.setup_optimizer(optimizer_config)

        self.lpips = LPIPS(eval=False).cuda()

    def setup_optimizer(self, optimizer_config):
        self.optimizer_config = optimizer_config
        model = self.uvgs_model

        l = []
        if optimizer_config.get('optim_uv_plane', False):
            print(f'[UVGaussianAvatar] optim_uv_plane, lr={optimizer_config.optim_uv_plane.lr}')
            model.uv_plane._feat_plane = nn.Parameter(model.uv_plane._feat_plane.requires_grad_(True))
            l.append({
                'params': [model.uv_plane._feat_plane], 
                'lr': optimizer_config.optim_uv_plane.lr, 
                "name": "_feat_plane"
            })

        if optimizer_config.get('optim_offset_mlp', False):
            print(f'[UVGaussianAvatar] optim_offset_mlp, lr={optimizer_config.optim_offset_mlp.lr}')
            l.append({
                'params': model.offset_mlp_net.parameters(), 
                'lr': optimizer_config.optim_offset_mlp.lr, 
                "name": "_offset_mlp"
            })
        if optimizer_config.get('optim_rotation_mlp', False):
            print(f'[UVGaussianAvatar] optim_rotation_mlp, lr={optimizer_config.optim_rotation_mlp.lr}')
            l.append({
                'params': model.rotation_mlp_net.parameters(), 
                'lr': optimizer_config.optim_rotation_mlp.lr, 
                "name": "_rotation_mlp"
            })
        if optimizer_config.get('optim_scaling_mlp', False):
            print(f'[UVGaussianAvatar] optim_scaling_mlp, lr={optimizer_config.optim_scaling_mlp.lr}')
            l.append({
                'params': model.scaling_mlp_net.parameters(), 
                'lr': optimizer_config.optim_scaling_mlp.lr, 
                "name": "_scaling_mlp"
            })
        if optimizer_config.get('optim_color_mlp', False):
            print(f'[UVGaussianAvatar] optim_color_mlp, lr={optimizer_config.optim_color_mlp.lr}')
            l.append({
                'params': model.color_mlp_net.parameters(), 
                'lr': optimizer_config.optim_color_mlp.lr, 
                "name": "_color_mlp"
            })
        if optimizer_config.get('optim_opacity_mlp', False):
            print(f'[UVGaussianAvatar] optim_color_mlp, lr={optimizer_config.optim_opacity_mlp.lr}')
            l.append({
                'params': model.opacity_mlp_net.parameters(), 
                'lr': optimizer_config.optim_opacity_mlp.lr, 
                "name": "_opacity_mlp"
            })
        if optimizer_config.get('optim_mapper', False):
            print(f'[Mapper Based SDS] optim_mapper, lr={optimizer_config.optim_mapper.lr}')
            model.mapper.requires_grad = True
            l.append({
                'params': model.mapper, 
                'lr': optimizer_config.optim_mapper.lr, 
                "name": "_mapper"
            })

        self.optimizer = torch.optim.Adam(l, lr=5e-4, eps=1e-15)
        # model.xyz_gradient_accum = torch.zeros((model.get_xyz.shape[0], 1), device="cuda")
        # model.denom = torch.zeros((model.get_xyz.shape[0], 1), device="cuda")
        # model.percent_dense = optimizer_config.get('percent_dense', 0.01)

        # scheduler
        if optimizer_config.get('scheduler', None):
            total_iteration = optimizer_config.total_iteration
            milestones = optimizer_config.scheduler.get('milestone', 10000)
            if not isinstance(milestones, list):
                milestones = [i for i in range(1, total_iteration) if i % milestones == 0]
            decay = optimizer_config.get('decay', 0.33)

            self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=decay,
            ))

    def step(self, enable_optim=True, enable_smplx=True):
        if enable_optim:
            self.optimizer.step()

        if enable_smplx and self.smplx_optim is not None:
            self.smplx_optim.step()

        for scheduler in self.schedulers:
            scheduler.step()

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

        if self.smplx_optim is not None:
            self.smplx_optim.zero_grad()

    def smplx_state_dict(self):
        state = {
            **self.state_dict(),
            'optimizer': {**self.optimizer.state_dict()},
        }

        if self.smplx_optim is not None:
            state['smplx_optim'] = {**self.smplx_optim.state_dict()}

        return state
    
    def collect_loss(self, gt_image, image, gt_alpha_mask=None):
        Ll1 = l1_loss(image, gt_image)

        if self.optimizer_config.get('lambda_ssim', 0) > 0:
            Lssim = 1.0 - ssim(image, gt_image)
            loss = (1.0 - self.optimizer_config.lambda_ssim) * Ll1 + self.optimizer_config.lambda_ssim * Lssim
        else:
            loss = Ll1

        if self.optimizer_config.get('lambda_perceptual', 0) > 0:
            if gt_alpha_mask is not None:
                Llpips = self.lpips(gt_image * gt_alpha_mask, image * gt_alpha_mask).squeeze()
            else:
                Llpips = self.lpips(gt_image, image).squeeze()
            loss += self.optimizer_config.lambda_perceptual * Llpips

        if self.optimizer_config.get('lambda_scaling', 0) > 0:
            thresh_scaling_max = self.optimizer_config.get('thresh_scaling_max', 0.004)
            thresh_scaling_ratio = self.optimizer_config.get('thresh_scaling_ratio', 4.0)
            max_vals = self.uvgs_model.get_scaling.max(dim=-1).values
            min_vals = self.uvgs_model.get_scaling.min(dim=-1).values
            ratio = max_vals / min_vals
            thresh_idxs = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
            if thresh_idxs.sum() > 0:
                loss += self.optimizer_config.lambda_scaling * max_vals[thresh_idxs].mean()

        psnr_full = psnr(image, gt_image).mean().float().item()

        return {
            'loss': loss,
            'psnr_full': psnr_full,
        }


