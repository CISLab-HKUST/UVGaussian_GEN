import os
import torch
import numpy as np
from random import randint
from argparse import ArgumentParser
from arguments import PipelineParams, CamParams
from scene.dataset_readers import GetRandTrainCameras
from gaussian_renderer import network_gui
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from model.uv_gauss_model import UVGaussModel
from model.uv_gauss_optim import UVGaussOptimizer
from guidance.sd_utils import StableDiffusionGuidance
from torchvision.utils import save_image
from tools.prompt_utils import get_embeddings
from dreams import libcore
from dreams.libcore.omegaconf_utils import load_from_config



if __name__ == '__main__':
    parser = ArgumentParser(description="test tetgen")
    pp = PipelineParams(parser)
    # parser.add_argument('--prompt', type=str, required=True, help="text prompt")
    parser.add_argument('--prompt', type=str, default="An Ironman", help="text prompt")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, default="configs")
    parser.add_argument('--model_config', nargs='+', default=["uvgs_model.yaml"], help='path to config file')
    parser.add_argument('--train_config', nargs='+', default=["uvgs_train.yaml"], help='path to config file')
    parser.add_argument('--guidance_config', nargs='+', default=["guidance.yaml"], help='path to config file')
    parser.add_argument('--camera_config', nargs='+', default=["camera.yaml"], help='path to config file')
    args, extras = parser.parse_known_args()
    pipe = pp.extract(args)

    # output dir
    model_path = os.path.join(args.dat_dir, f"output-splatting/{datetime.now().strftime('@%Y%m%d-%H%M%S')}")
    os.makedirs(model_path, exist_ok=True)

    train_imgs_dir = os.path.join(model_path, "training_images")
    os.makedirs(train_imgs_dir, exist_ok=True)

    test_camera_dir = os.path.join(model_path, "test_camera")
    os.makedirs(test_camera_dir, exist_ok=True)

    prompt = args.prompt

    # load model, training and guidance config
    model_config = load_from_config(args.model_config, args.dat_dir, cli_args=extras)
    train_config = load_from_config(args.train_config, args.dat_dir, cli_args=extras)
    guidance_config = load_from_config(args.guidance_config, args.dat_dir, cli_args=extras)
    camera_config = load_from_config(args.camera_config, args.dat_dir, cli_args=extras)

    device = torch.device("cuda")

    from omegaconf import OmegaConf
    OmegaConf.save(model_config, os.path.join(model_path, 'model_config.yaml'))
    OmegaConf.save(train_config, os.path.join(model_path, 'train_config.yaml'))
    OmegaConf.save(guidance_config, os.path.join(model_path, 'guidance_config.yaml'))
    OmegaConf.save(camera_config, os.path.join(model_path, 'camera_config.yaml'))

    total_iteration = train_config.optim.total_iteration

    guidance_config.total_iter = total_iteration

    ##################################################
    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = True
    uvgs_model = UVGaussModel(model_config, verbose=True)
    uvgs_optim = UVGaussOptimizer(uvgs_model, train_config.optim)
    
    # define the mapper in uvg_model, easy to optimize


    scene_cameras = GetRandTrainCameras(config=camera_config, num_cameras=50)
    ##################################################
    network_gui.init(args.ip, args.port)
    viewpoint_stack = None
    
    out_dir = "test_camera"
    os.makedirs(out_dir, exist_ok=True)

    pbar = tqdm(range(1, total_iteration))
    for i, viewpoint_cam in enumerate(scene_cameras):
        # predict 3dgs variables from uv_plane
        uvgs_model.update_gauss_status()

        # send one image to gui (optional)
        network_gui.render_to_network(uvgs_model, pipe, args.dat_dir)

        # print(uvgs_model)

        # exit(0)
        # camera message
        direction = viewpoint_cam.dir
        region = viewpoint_cam.region
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width

        # render
        render_pkg = uvgs_model.render_to_camera(viewpoint_cam, pipe, background='white')
        image = render_pkg['render'].unsqueeze(0)
        depth = render_pkg['depth'].unsqueeze(0).repeat(1, 3, 1, 1)

        # Test Camera
        save_image(image, f'{test_camera_dir}/test_render_{region}_{direction}_{i}.png')
