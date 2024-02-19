import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from diffusers import DDIMScheduler
from torchvision.utils import save_image
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from torch.cuda.amp import custom_bwd, custom_fwd
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, DDIMScheduler
from .sd_step import *


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class StableDiffusionGuidance(object):
    def __init__(self, opt, device, mapper=None) -> None:
        self.device = device
        self.opt = opt

        self.model_path = self.opt.model_path
        self.precision_t = torch.float16 if self.opt.fp16 else torch.float32

        # load pipeline

        # TODO Use Skeleton ControlNet
        self.use_controlnet = self.opt.use_controlnet
        if self.use_controlnet: 
            self.controlnet_model_path = self.opt.controlnet_model_path
            self.controlnet_depth = ControlNetModel.from_pretrained(
                self.controlnet_model_path,
                torch_dtype=self.precision_t
            ).to(self.device)
            
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.precision_t,
        )

        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_path,
            subfolder="scheduler",
            torch_dtype=self.precision_t,
        )

        if self.opt.vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe = pipe.to(self.device)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, device=self.device)  

        # timestep scale
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.min_step = int(self.num_train_timesteps * self.opt.t_range[0])
        self.max_step = int(self.num_train_timesteps * self.opt.t_range[1])

        # fix noise
        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(self.opt.noise_seed)

        self.uncond_scale = self.opt.uncond_scale

        self.sche_func = ddim_step
        
        self.guidance_scale = self.opt.guidance_scale
        self.total_iter = self.opt.total_iter

        # Timestep annealing
        self.timestep_annealing = self.opt.timestep_annealing

        # Mapper
        if mapper is None:
            mapper = torch.ones((1, 4, 64, 64), device=self.device, dtype=self.precision_t)
            mapper.requires_grad = True
        self.mapper = mapper

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.rgb_latent_factors = torch.tensor([
                    # R       G       B
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)

        print(f'[INFO] loaded stable diffusion!')


    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)
    
    # @torch.no_grad()
    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence

    
    def get_uncond_guidance_scale(self, timestep):
        # 7.5 - 0.05 * np.exp(5 * x)
        # y = 0.001 * np.exp(10*x)
        stage = timestep / self.max_step
        cfg = 1 + 0.1 * math.exp(5 * (1 - stage))
        return cfg

    def get_anneal_ind_t(self, iteration, mode='linear'):
        if mode == 'linear':
            ind_t = int(self.max_step - (self.max_step - self.min_step) * (iteration / self.total_iter))

        elif mode == 'hifa':
            ind_t = int(self.max_step - (self.max_step - self.min_step) * math.sqrt(iteration / self.total_iter))

        return ind_t

    def train_mapper(
        self,
        null_embeddings,
        pred_rgb,
        pred_depth=None,
        iteration=None,
        resolution=(512, 512),
    ):
        if pred_rgb.shape[1] != 4:
            pred_rgb = F.interpolate(pred_rgb, resolution, mode='bilinear', align_corners=False)
            pred_depth = F.interpolate(pred_depth, resolution, mode='bilinear', align_corners=False)
            img_latents, _ = self.encode_imgs(pred_rgb)
        else:
            img_latents = pred_rgb

        with torch.no_grad():
            if self.timestep_annealing:
                ind_t = int(self.max_step - (self.max_step - self.min_step) * math.sqrt(iteration / self.total_iter))
            else:  
                ind_t = torch.randint(self.min_step, self.max_step, (img_latents.shape[0],), dtype=torch.long, device=self.device)[0]

            t = self.timesteps[ind_t]
            t = torch.tensor([t], dtype=torch.long, device=self.device)

            text_embeddings = null_embeddings

            if self.noise_temp is None:
                self.noise_temp = torch.randn(img_latents.shape, generator=self.noise_gen, dtype=self.precision_t, device=self.device)

            noise = torch.randn(img_latents.shape, generator=self.noise_gen, dtype=self.precision_t, device=self.device)
            # noise = self.noise_temp
            
            latents_noisy = self.scheduler.add_noise(img_latents, noise, t)

            down_block_res_samples = mid_block_res_sample = None
            if self.use_controlnet:
                pred_depth_input = pred_depth.repeat(latents_noisy.shape[0], 1, 1, 1).to(self.precision_t)
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latents_noisy,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )

            noise_pred = self.unet(
                latents_noisy,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample, 
            ).sample

        noise_mapped = self.mapper * noise_pred 
        # noise_mapped_scaled = noise.detach().var() / noise_mapped.var() * (noise_mapped - noise_mapped.mean()) + noise_mapped.mean()
        # loss = 0.5 * F.mse_loss(noise_mapped_scaled, noise.detach(), reduction="mean") / pred_rgb.shape[0]
        loss = 0.5 * F.mse_loss(noise_mapped, noise.detach(), reduction="mean") / pred_rgb.shape[0]

        return loss


    def train_sds(
        self,
        prompt_embeddings, # Tensor [K/2B, 66, 768]
        null_embeddings, # Tensor [B, 66, 768]
        pred_rgb, # Tensor [B, 4, 512, 512]
        pred_depth=None,
        pred_skeleton=None,
        iteration=None,
        resolution=(512, 512),
        save_folder=None,
        vis_interval=20,
    ):
        if pred_rgb.shape[1] != 4:
            pred_rgb = F.interpolate(pred_rgb, resolution, mode='bilinear', align_corners=False)
            pred_depth = F.interpolate(pred_depth, resolution, mode='bilinear', align_corners=False)
            img_latents, _ = self.encode_imgs(pred_rgb)
        else:
            img_latents = pred_rgb

        with torch.no_grad():
            if self.timestep_annealing:
                ind_t = int(self.max_step - (self.max_step - self.min_step) * math.sqrt(iteration / self.total_iter))
            else:  
                ind_t = torch.randint(self.min_step, self.max_step, (img_latents.shape[0],), dtype=torch.long, device=self.device)[0]


            t = self.timesteps[ind_t]
            t = torch.tensor([t], dtype=torch.long, device=self.device)

            text_embeddings = torch.cat([prompt_embeddings, null_embeddings], dim=0)

            if self.noise_temp is None:
                self.noise_temp = torch.randn(img_latents.shape, generator=self.noise_gen, dtype=self.precision_t, device=self.device)

            # noise = self.noise_temp
            noise = torch.randn(img_latents.shape, generator=self.noise_gen, dtype=self.precision_t, device=self.device)
            
            latents_noisy = self.scheduler.add_noise(img_latents, noise, t)

            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)

            down_block_res_samples = mid_block_res_sample = None
            if self.use_controlnet:
                pred_depth_input = pred_depth.repeat(latent_model_input.shape[0], 1, 1, 1).reshape(-1, 3, resolution[0], resolution[1]).to(self.precision_t)
                down_block_res_samples, mid_block_res_sample = self.controlnet_depth(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=pred_depth_input,
                    return_dict=False,
                )

            noise_pred = self.unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample, 
            ).sample

            noise_pred_text, noise_pred_null = noise_pred.chunk(2)

            delta_cond = self.guidance_scale * (noise_pred_text - noise_pred_null)
            
            uncond_cfg = 1.0
            if self.uncond_scale:
                uncond_cfg = self.get_uncond_guidance_scale(timestep=t.item())

            delta_uncond = uncond_cfg * (noise_pred_null - self.mapper.detach() * noise_pred_null)
            

        noise_pred = delta_uncond + delta_cond

        w = (1 - self.alphas[t.item()]).view(-1, 1, 1, 1)

        grad = w * (noise_pred)
        grad = torch.nan_to_num(grad)
        # print(f"timestep: {t.item()}, uncond guidance scale: {round(uncond_guidance_scale, 3)}, momentumn scale: {round(momentumn_scale, 3)}.")

        loss = 0.5 * F.mse_loss(img_latents, (img_latents - grad).detach(), reduction="mean") / img_latents.shape[0]
        # print(f"timestep: {t.item()}, uncond guidance scale: {round(uncond_guidance_scale, 3)}, momentumn scale: {round(momentumn_scale, 3)}.")

        if iteration % vis_interval == 0:

            lat2rgb = lambda x: torch.clip((x.permute(0,2,3,1) @ self.rgb_latent_factors.to(x.dtype)).permute(0,3,1,2), 0., 1.)
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration, t.item()))
            with torch.no_grad():
                alpha_t = self.scheduler.alphas_cumprod[t.item()]
                     
                labels = self.decode_latents((img_latents - grad).type(self.precision_t))
                
                # pred_x0_uncond = pred_x0_sp[:1, ...]

                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                latents_rgb = F.interpolate(lat2rgb(img_latents), (resolution[0], resolution[1]), mode='bilinear', align_corners=False)
                

                pred_rgb = self.decode_latents(img_latents.detach())
                viz_images = torch.cat([pred_rgb, pred_depth, latents_rgb, norm_grad,
                                        labels],dim=0) 
                save_image(viz_images, save_path_iter)

        return loss
    

    def train_step(
        self,
        prompt_embeddings,
        null_embeddings,
        pred_rgb,
        pred_depth=None,
        pred_skeleton=None,
        iteration=None,
        resolution=(512, 512),
        save_folder=None,
        vis_interval=20,
    ):

        loss_mapper = self.train_mapper(
            null_embeddings=null_embeddings,
            pred_rgb=pred_rgb.detach(),
            pred_depth=pred_depth,
            iteration=iteration,
            resolution=resolution,
        )

        loss_sds = self.train_sds(
            prompt_embeddings=prompt_embeddings,
            null_embeddings=null_embeddings,
            pred_rgb=pred_rgb,
            pred_depth=pred_depth,
            pred_skeleton=pred_skeleton,
            iteration=iteration,
            resolution=resolution,
            save_folder=save_folder,
            vis_interval=vis_interval,
        )

        return loss_sds, loss_mapper


