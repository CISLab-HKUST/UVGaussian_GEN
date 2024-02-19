import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pytorch3d.structures.meshes as py3d_meshes
from dreams import libcore
from dreams.bone_deformer import smplx_utils
from dreams.libcore.nvdiffrast_mesh import NvdiffrastMesh
from dreams.inst_networks import network_utils
import nvdiffrast.torch as dr
from .gauss_base import GaussianBase

def to_abs_path(fn, dir):
    if not os.path.isabs(fn):
        fn = os.path.join(dir, fn)
    return fn

def to_cache_path(dir):
    cache_dir = os.path.join(dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

class UVPlane(nn.Module):
    def __init__(self, mesh, config, device='cuda') -> None:
        super().__init__()
        self.config = config
        self.device = device

        # rasterize mesh to uv
        self.nvmesh = NvdiffrastMesh(mesh)

        batch = config.get('batch', 1)
        width = height = config.resolution
        rlt = self.nvmesh.rasterizeToAtlas(width, height, with_texture=False)

        self.mask = (rlt['mask'].squeeze(-1) != 0)
        self.rast_out = rlt['rast_out']

        # create feature plane
        self.num_gauss = self.mask.sum()
        self.feat_dims = config.get('feature_dims', 48)
        self._feat_plane = 0.1 * torch.rand((batch, height, width, self.feat_dims)).float().to(device)

    # get vertex from mesh per uv pixel
    # [input] verts [b x n x 3]/[n x 3]: deformed verts on smplx mesh
    def get_xyz(self, verts=None):
        if verts is None:
            verts = self.nvmesh.vertices[0]
        
        input_dims = len(verts.shape)
        if input_dims == 2:
            verts = verts[None, ...]

        vmap, _ = dr.interpolate(verts, self.rast_out, self.nvmesh.faces)
        if input_dims == 2:
            return vmap[self.mask].squeeze(0)
        else:
            return vmap[self.mask]
        
    @property
    def get_feat(self):
        return self._feat_plane[self.mask]

class UVGaussModel(GaussianBase):
    def __init__(self, config,
                 device=torch.device('cuda'),
                 verbose=False):
        super().__init__()
        self.config = config
        self.device = device
        self.verbose = verbose

        self.register_buffer('_offset', torch.Tensor(0))
        self.register_buffer('_rotation', torch.Tensor(0))
        self.register_buffer('_scaling', torch.Tensor(0))
        self.register_buffer('_color', torch.Tensor(0))
        self.register_buffer('_opacity', torch.Tensor(0))
        self.scaling_base = 1e-3

        if config is not None:
            self.setup_config(config)

    def setup_config(self, config):
        self.config = config
        self.setup_smplx(config.smplx_model, config.yaml_dir)
        self.setup_uv_plane(config.uv_plane, config.yaml_dir)
        self.setup_mlp(config.mlp_config, config.yaml_dir)
        self.setup_mapper(config.mapper_config, config.yaml_dir)

    def setup_smplx(self, smplx_config, yaml_dir):
        if smplx_config.get('from_pt', False):
            smplx_params = smplx_utils.load_and_detach(to_abs_path(smplx_config.from_pt, yaml_dir))
        else:
            smplx_params = {
                'gender': smplx_config.get('gender', 'male-lite'),
                'num_betas': smplx_config.get('num_betas', 10),
                'num_expression_coeffs': smplx_config.get('num_expression_coeffs', 10),
                'flat_hand_mean': smplx_config.get('flat_hand_mean', True),
                'use_pca': smplx_config.get('use_pca', True),
                'num_pca_comps': smplx_config.get('num_pca_comps', 6),
                # 'use_face_contour': smplx_config.get('use_face_contour', True),
            }

        self.smplx_model = smplx_utils.create_smplx_model(**smplx_params)

        # Transfer to APose
        body_pose = torch.zeros(1, self.smplx_model.NUM_BODY_JOINTS, 3)
        body_pose[:, 15, :] = torch.tensor([0.0, 0.0, -np.pi/4])
        body_pose[:, 16, :] = torch.tensor([0.0, 0.0, +np.pi/4])
        body_pose = body_pose.clone().detach().float().requires_grad_(True)

        out = self.smplx_model(body_pose=body_pose, **smplx_params)
        self.mesh_cano = smplx_utils.convert_smplx_to_meshcpu(self.smplx_model, V=out['vertices'][0])

        if self.verbose:
            fn = os.path.join(to_cache_path(yaml_dir), 'smplx_cano.obj')
            self.mesh_cano.save_to_obj(fn)

        self.smplx_model = self.smplx_model.cuda()

    def setup_uv_plane(self, uv_config, yaml_dir):
        self.uv_plane = UVPlane(self.mesh_cano, uv_config, device=self.device)

        if self.verbose:
            uv_verts = self.uv_plane.get_xyz()
            fn = os.path.join(to_cache_path(yaml_dir), 'uv_verts_cano.ply')
            libcore.savePointsToPly(fn, uv_verts.detach().cpu())

    def setup_mlp(self, mlp_config, yaml_dir):
        # offset, quaternion, scaling, color, opacity
        # out_dims = 3 + 4 + 3 + 3 + 1
        # self.mlp_net = network_utils.get_mlp(self.uv_plane.feat_dims, 
        #                                      out_dims, mlp_config).to(self.device)
        
        self.setup_geometry_mlp(mlp_config.geometry)
        self.setup_color_mlp(mlp_config.color)
        self.color_activation = network_utils.get_activation('sigmoid')

    
    def setup_color_mlp(self, color_mlp_config):
        # opacity, color
        out_dims = 1 + 3
        self.color_mlp_net = network_utils.get_mlp(self.uv_plane.feat_dims,
                                                  out_dims, color_mlp_config).to(self.device)
        
    def setup_geometry_mlp(self, geometry_mlp_config):
        # offset, rotation, scaling
        # out_dims = 3 + 4 + 3
        self.offset_mlp_net = network_utils.get_mlp(self.uv_plane.feat_dims,
                                                      3, geometry_mlp_config).to(self.device)
        self.rotation_mlp_net = network_utils.get_mlp(self.uv_plane.feat_dims,
                                                      4, geometry_mlp_config).to(self.device)
        self.scaling_mlp_net = network_utils.get_mlp(self.uv_plane.feat_dims,
                                                      3, geometry_mlp_config).to(self.device)
    
    def setup_mapper(self, mapper_config, yaml_dir):
        self.mapper = torch.ones(tuple(mapper_config.shape), device=self.device, dtype=torch.float32)

    def update_gauss_status(self):
        x = self.uv_plane.get_feat

        offset_x = self.offset_mlp_net(x)
        rotation_x = self.rotation_mlp_net(x)
        scaling_x = self.scaling_mlp_net(x)

        color_x = self.color_mlp_net(x)

        self._offset = offset_x
        self._rotation = rotation_x
        self._scaling = scaling_x

        self._opacity = color_x[..., :1]
        self._color = color_x[..., 1:4]

    ##################################################
    @property
    def num_gauss(self):
        return self.uv_plane.num_gauss
    
    @property
    def get_offset(self):
        return self._offset

    @property
    def get_xyz(self):
        uv_verts = self.uv_plane.get_xyz()
        return uv_verts + self._offset

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) * self.scaling_base
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_colors_precomp(self, viewpoint_camera=None):
        return self.color_activation(self._color)
    
    def get_color_opacity(self, viewpoint_camera=None):
        colors_precomp = self.get_colors_precomp(viewpoint_camera=viewpoint_camera)
        opacity = self.get_opacity
        return colors_precomp, opacity
    

    


