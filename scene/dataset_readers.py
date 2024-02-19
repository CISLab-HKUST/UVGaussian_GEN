#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.camera_utils import loadCam, loadAvatarCam
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
import cv2
import random


class CameraInfoExt(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class RCameraInfoExt(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    width: int
    height: int
    FovY: np.array
    FovX: np.array
    dir: np.array
    region: str



class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if 'red' not in vertices:
        shs = np.random.random((positions.shape[0], 3)) / 255.0
        colors = SH2RGB(shs)
    else:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    if 'nx' not in vertices:
        normals=np.zeros((positions.shape[0], 3))
    else:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)



def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def get_view_direction(thetas, phis, overhead, front):
    """
    We only encoder ['front', 'side', 'back', "overhead"] and skip "bottom"
    Args:
        thetas:
        phis:
        overhead:
        front:

    Returns:

    """
    #                   phis [B,];          thetas: [B,]
    # front = 0         [-half_front, half_front)
    # side (left) = 1   [half_front, 180 - half_front)
    # back = 2          [180 - half_front, 180+half_front)
    # side (right) = 1  [180+half_front, 360-half_front)
    # top = 3           [0, overhead]
    # bottom = 4        [180-overhead, 180]

    half_front = front / 2.
    phis_abs = phis.abs()
    res = torch.ones(thetas.shape[0], dtype=torch.long)
    res[(phis_abs <= half_front)] = 0
    # res[(phis_abs > half_front) & (phis_abs < np.pi - half_front)] = 1
    res[(phis_abs > np.pi - half_front) & (phis_abs <= np.pi)] = 2
    # override by thetas
    # res[thetas <= overhead] = 3
    # res[thetas >= (np.pi - overhead)] = 4
    return res

def near_head_poses(size,
                    device,
                    shift,
                    radius_range=[0.15, 0.2],
                    theta_range=[70, 90],
                    phi_range=[-60, 60],
                    return_dirs=False,
                    angle_overhead=30,
                    angle_front=60,
                    jitter=False,
                    face_scale=1.0):
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)

    # face_center_jitter = face_center + (random.random() - 0.5) * face_scale * 0.2
    # shift = torch.as_tensor([0, face_center, 0], device=device).view(1, 3)

    radius_range = np.array(radius_range) * face_scale
    radius = torch.rand(size) * (radius_range[1] - radius_range[0]) + radius_range[0]
    thetas = torch.rand(size) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) + shift  # [B, 3]
    targets = torch.zeros_like(centers) + shift

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        angle_overhead = np.deg2rad(angle_overhead)
        angle_front = np.deg2rad(angle_front)
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius



def GenerateRandCameras(config, num_cameras):
    if config is None:
        raise Exception("Camera Config is None!")
    
    face_ratio = config.face_ratio
    device=config.device
    face_center = torch.as_tensor([0, 0.32, 0]).view(1, 3)
    body_center = torch.as_tensor([0, -0.42, 0]).view(1, 3)
    
    image_h = config.image_h
    image_w = config.image_w

    cam_infos = []

    for idx in range(num_cameras):
        
        choice = random.random()

        if choice < face_ratio:
            region = "face"
            poses, dirs, thetas, phis, radius = near_head_poses(
                size=1,
                device=device,
                radius_range=config.head_radius_range,
                phi_range=config.head_phi_range,
                theta_range=config.head_theta_range,
                angle_overhead=config.angle_overhead,
                angle_front=config.angle_front,
                jitter=config.jitter_pose,
                shift=face_center,
                face_scale=1.0,
                return_dirs=config.dir_text,
            )
        else:
            region = "body"
            poses, dirs, thetas, phis, radius = near_head_poses(
                size=1,
                device=device,
                radius_range=config.radius_range,
                phi_range=config.phi_range,
                theta_range=config.theta_range,
                angle_overhead=config.angle_overhead,
                angle_front=config.angle_front,
                jitter=config.jitter_pose,
                shift=body_center,
                face_scale=1.0,
                return_dirs=config.dir_text,
            )

        fov = random.random() * (config.fovy_range[1] - config.fovy_range[0]) + config.fovy_range[0]
        matrix = np.linalg.inv(poses.squeeze(0).cpu())
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        # matrix = poses[idx]
        # R = matrix[:3,:3]
        # T = matrix[:3, 3]
        fovy = focal2fov(fov2focal(fov, image_h), image_w)
        FovY = fovy
        FovX = fov
        cam_infos.append(RCameraInfoExt(
            uid=idx, R=R, T=T, width=image_w, height=image_h, 
            FovY=FovY, FovX=FovX, dir=dirs, region=region,
        ))
    return cam_infos

def GetRandTrainCameras(config, num_cameras):
    camera_info_list = GenerateRandCameras(config, num_cameras)
    
    scene_camera_list = []
    for camera_info in camera_info_list:
        idx = camera_info.uid
        scene_camera = loadAvatarCam(
            config,
            idx,
            camera_info,
        )
        scene_camera_list.append(scene_camera)

    return scene_camera_list


sceneLoadTypeCallbacks = {
    # "Colmap": readColmapSceneInfo,
    # "Blender" : readNerfSyntheticInfo,
    # "RandomCam" : readCircleCamInfo
}