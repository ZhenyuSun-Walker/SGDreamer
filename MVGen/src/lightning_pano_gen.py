# import pytorch_lightning as pl
# from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
# import torch
# import os
# from PIL import Image
# from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
# import numpy as np
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from .models.pano.MVGenModel import MultiViewBaseModel
# from torchvision import models
# import torch.nn.functional as F

# class PanoGenerator(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()

#         self.lr = config['train']['lr']
#         self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
#         self.diff_timestep = config['model']['diff_timestep']
#         self.guidance_scale = config['model']['guidance_scale']

#         self.tokenizer = CLIPTokenizer.from_pretrained(
#             config['model']['model_id'], subfolder="tokenizer", torch_dtype=torch.float16)
#         self.text_encoder = CLIPTextModel.from_pretrained(
#             config['model']['model_id'], subfolder="text_encoder", torch_dtype=torch.float16)

#         self.vae, self.scheduler, unet = self.load_model(
#             config['model']['model_id'])
#         self.mv_base_model = MultiViewBaseModel(
#             unet, config['model'])
#         self.trainable_params = self.mv_base_model.trainable_parameters
        
#         self.save_hyperparameters()

#     def load_model(self, model_id):
#         vae = AutoencoderKL.from_pretrained(
#             model_id, subfolder="vae")
#         vae.eval()
#         scheduler = DDIMScheduler.from_pretrained(
#             model_id, subfolder="scheduler")
#         unet = UNet2DConditionModel.from_pretrained(
#             model_id, subfolder="unet")
#         return vae, scheduler, unet

#     @torch.no_grad()
#     def encode_text(self, text, device):
#         text_inputs = self.tokenizer(
#             text, padding="max_length", max_length=self.tokenizer.model_max_length,
#             truncation=True, return_tensors="pt"
#         ).to(device)  # Ensure on same device
#         text_input_ids = text_inputs.input_ids
#         if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
#             attention_mask = text_inputs.attention_mask.to(device)  # Ensure on same device
#         else:
#             attention_mask = None
#         prompt_embeds = self.text_encoder(
#             text_input_ids.to(device), attention_mask=attention_mask)

#         return prompt_embeds[0].float(), prompt_embeds[1]

#     @torch.no_grad()
#     def encode_image(self, x_input, vae):
#         b = x_input.shape[0]

#         x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, 2, 3, 512, 512)
#         x_input = x_input.reshape(-1,
#                                   x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
#         z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

#         z = z.sample()
#         z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
#                       z.shape[-1])  # (bs, 2, 4, 64, 64)

#         # use the scaling factor from the vae config
#         z = z * vae.config.scaling_factor
#         z = z.float()
#         return z

#     @torch.no_grad()
#     def decode_latent(self, latents, vae):
#         b, m = latents.shape[0:2]
#         latents = (1 / vae.config.scaling_factor * latents)
#         images = []
#         for j in range(m):
#             image = vae.decode(latents[:, j]).sample
#             images.append(image)
#         image = torch.stack(images, dim=1)
#         image = (image / 2 + 0.5).clamp(0, 1)
#         image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
#         image = (image * 255).round().astype('uint8')

#         return image

#     def configure_optimizers(self):
#         param_groups = []
#         for params, lr_scale in self.trainable_params:
#             param_groups.append({"params": params, "lr": self.lr * lr_scale})
#         optimizer = torch.optim.AdamW(param_groups)
#         scheduler = {
#             'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
#             'interval': 'epoch',  # update the learning rate after each epoch
#             'name': 'cosine_annealing_lr',
#         }
#         return {'optimizer': optimizer, 'lr_scheduler': scheduler}

#     def training_step(self, batch, batch_index):      
#         meta = {
#             'K': batch['K'],
#             'R': batch['R']
#         }

#         device = batch['images'].device
#         prompt_embds = []
#         for prompt in batch['prompt']:
#             prompt_embds.append(self.encode_text(
#                 prompt, device)[0])
#         latents = self.encode_image(
#             batch['images'], self.vae)
#         t = torch.randint(0, self.scheduler.num_train_timesteps,
#                         (latents.shape[0],), device=latents.device).long()
#         prompt_embds = torch.stack(prompt_embds, dim=1)

#         noise = torch.randn_like(latents)
#         noise_z = self.scheduler.add_noise(latents, noise, t)
#         t = t[:, None].repeat(1, latents.shape[1])
#         denoise = self.mv_base_model(
#             noise_z, t, prompt_embds, meta)
#         target = noise       

#         # eps mode
#         loss = F.mse_loss(denoise, target)

#         self.log('train_loss', loss)
#         return loss

#     # def training_step(self, batch):
#     #     meta = {
#     #         'K': batch['K'],
#     #         'R': batch['R']
#     #     }

#     #     device = batch['images'].device  # Get device from batch images
#     #     prompt_embds = []
#     #     for prompt in batch['prompt']:
#     #         prompt_embds.append(self.encode_text(
#     #             prompt, device)[0])
#     #     latents = self.encode_image(
#     #         batch['images'], self.vae)  # Ensure latents on same device
#     #     t = torch.randint(0, self.scheduler.num_train_timesteps, 
#     #                       (latents.shape[0],), device=latents.device).long()
#     #     prompt_embds = torch.stack(prompt_embds, dim=1) # Ensure prompt_embds on same device

#     #     noise = torch.randn_like(latents).to(device)  # Ensure noise on same device
#     #     noise_z = self.scheduler.add_noise(latents, noise, t)
#     #     t = t[:, None].repeat(1, latents.shape[1])
#     #     denoise = self.mv_base_model(noise_z, t, prompt_embds, meta)
        
#     #     # print(f"denoise shape: {denoise.shape}")
#     #     # target = noise       
#     #     # print(f"target shape: {target.shape}")

#     #     # vgg = load_vgg_model(device)  # Ensure VGG model on same device

#     #     # # Combine losses
#     #     mse_loss = F.mse_loss(denoise, target)
#     #     # denoise = denoise[:, :3, :, :, :]
#     #     # target = target[:, :3, :, :, :]
#     #     # perceptual = perceptual_loss(denoise, target, vgg)
#     #     # clip_loss = clip_score_loss(denoise, batch['prompt'], clip_model)

#     #     # Define weights for losses
#     #     # total_loss = mse_loss + 0.5 * perceptual + 0.2 * clip_loss

#     #     self.log('train_loss', mse_loss)
#     #     return mse_loss

#     def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
#         latents = torch.cat([latents]*2)
#         timestep = torch.cat([timestep]*2)
        
#         R = torch.cat([batch['R']]*2)
#         K = torch.cat([batch['K']]*2)
      
#         meta = {
#             'K': K,
#             'R': R,
#         }

#         return latents, timestep, prompt_embd, meta

#     @torch.no_grad()
#     def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
#         latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
#             latents_high_res, _timestep, prompt_embd, batch)

#         noise_pred = model(
#             latents, _timestep, _prompt_embd, meta)

#         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + self.guidance_scale * \
#             (noise_pred_text - noise_pred_uncond)

#         return noise_pred

#     @torch.no_grad()
#     def validation_step(self, batch, batch_idx):
#         images_pred = self.inference(batch)
#         images = ((batch['images']/2+0.5)
#                           * 255).cpu().numpy().astype(np.uint8)
      
#         # compute image & save
#         if self.trainer.global_rank == 0:
#             self.save_image(images_pred, images, batch['prompt'], batch_idx)

#     @torch.no_grad()
#     def inference(self, batch):
#         images = batch['images']
#         bs, m, h, w, _ = images.shape
#         device = images.device

#         latents= torch.randn(
#             bs, m, 4, h//8, w//8, device=device)

#         prompt_embds = []
#         for prompt in batch['prompt']:
#             prompt_embds.append(self.encode_text(
#                 prompt, device)[0])
#         prompt_embds = torch.stack(prompt_embds, dim=1)

#         prompt_null = self.encode_text('', device)[0]
#         prompt_embd = torch.cat(
#             [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

#         self.scheduler.set_timesteps(self.diff_timestep, device=device)
#         timesteps = self.scheduler.timesteps

#         for i, t in enumerate(timesteps):
#             _timestep = torch.cat([t[None, None]]*m, dim=1)

#             noise_pred = self.forward_cls_free(
#                 latents, _timestep, prompt_embd, batch, self.mv_base_model)

#             latents = self.scheduler.step(
#                 noise_pred, t, latents).prev_sample
#         images_pred = self.decode_latent(
#             latents, self.vae)
       
#         return images_pred
    
#     @torch.no_grad()
#     def test_step(self, batch, batch_idx):
#         images_pred = self.inference(batch)

#         images = ((batch['images']/2+0.5)
#                           * 255).cpu().numpy().astype(np.uint8)
       
       
#         scene_id = batch['image_paths'][0].split('/')[2]
#         image_id=batch['image_paths'][0].split('/')[-1].split('.')[0].split('_')[0]
        
#         output_dir = batch['resume_dir'][0] if 'resume_dir' in batch else os.path.join(self.logger.log_dir, 'images')
#         output_dir=os.path.join(output_dir, "{}_{}".format(scene_id, image_id))
        
#         os.makedirs(output_dir, exist_ok=True)
#         for i in range(images.shape[1]):
#             path = os.path.join(output_dir, f'{i}.png')
#             im = Image.fromarray(images_pred[0, i])
#             im.save(path)
#             im = Image.fromarray(images[0, i])
#             path = os.path.join(output_dir, f'{i}_natural.png')
#             im.save(path)
#         with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
#             for p in batch['prompt']:
#                 f.write(p[0]+'\n')

#     @torch.no_grad()
#     def save_image(self, images_pred, images, prompt, batch_idx):

#         img_dir = os.path.join(self.logger.log_dir, 'images')
#         os.makedirs(img_dir, exist_ok=True)

#         with open(os.path.join(img_dir, f'{self.global_step}_{batch_idx}.txt'), 'w') as f:
#             for p in prompt:
#                 f.write(p[0]+'\n')
#         if images_pred is not None:
#             for m_i in range(images_pred.shape[1]):
#                 im = Image.fromarray(images_pred[0, m_i])
#                 im.save(os.path.join(
#                     img_dir, f'{self.global_step}_{batch_idx}_{m_i}_pred.png'))
#                 if m_i < images.shape[1]:
#                     im = Image.fromarray(
#                         images[0, m_i])
#                     im.save(os.path.join(
#                         img_dir, f'{self.global_step}_{batch_idx}_{m_i}_gt.png'))

import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
import torch
import os
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from .models.pano.MVGenModel import MultiViewBaseModel
from torchvision import models
import torch.nn.functional as F
from einops import rearrange

# Load VGG model
def load_vgg_model(device):
    vgg = models.vgg16(pretrained=True).features.eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

# Set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def encode_text(text, tokenizer, device):
    text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    return text_inputs

def perceptual_loss(generated, target, vgg):
    generated = generated.reshape(1 * 4, 3, 64, 64)
    target = target.reshape(1 * 4, 3, 64, 64)

    gen_features = vgg(generated)
    target_features = vgg(target)
    return F.mse_loss(gen_features, target_features)

def clip_score_loss(generated, prompts, clip_model):
    generated = generated.reshape(1 * 4, 3, 64, 64).to(next(clip_model.parameters()).device)  # Ensure on same device
    generated_embeds = clip_model.get_image_features(generated)
    text_embeds = [encode_text(prompt, clip_model.tokenizer, generated.device) for prompt in prompts]
    text_embeds = torch.stack(text_embeds).to(generated.device)
    return -torch.mean(generated_embeds @ text_embeds.T)

class PanoGenerator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']

        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['model_id'], subfolder="tokenizer", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['model']['model_id'], subfolder="text_encoder", torch_dtype=torch.float16)

        self.vae, self.scheduler, unet = self.load_model(
            config['model']['model_id'])
        self.mv_base_model = MultiViewBaseModel(
            unet, config['model'])
        self.trainable_params = self.mv_base_model.trainable_parameters

        self.save_hyperparameters()
       
    def load_model(self, model_id):
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae")
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet")
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0].float(), prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, 2, 3, 512, 512)
        x_input = x_input.reshape(-1,
                                  x_input.shape[-3], x_input.shape[-2], x_input.shape[-1])
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])  # (bs, 2, 4, 64, 64)

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)
        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):      
        meta = {
            'K': batch['K'],
            'R': batch['R']
        }

        device = batch['images'].device  # Get device from batch images
        prompt_embds = [self.encode_text(prompt, device)[0] for prompt in batch['prompt']]
        latents = self.encode_image(batch['images'], self.vae).to(device)  # Ensure latents on same device
        t = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
        prompt_embds = torch.stack(prompt_embds, dim=1).to(device)  # Ensure prompt_embds on same device

        noise = torch.randn_like(latents).to(device)  # Ensure noise on same device
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, latents.shape[1])
        denoise = self.mv_base_model(noise_z, t, prompt_embds, meta)
        
        # print(f"denoise shape: {denoise.shape}")
        target = noise       
        # print(f"target shape: {target.shape}")

        vgg = load_vgg_model(device)  # Ensure VGG model on same device

        # Combine losses
        mse_loss = F.mse_loss(denoise, target)
        denoise = denoise[:, :3, :, :, :]
        target = target[:, :3, :, :, :]
        perceptual = perceptual_loss(denoise, target, vgg)
        print(f"denoise shape: {denoise.shape}")
        print(f"batch['prompt'] length: {len(batch['prompt'])}")
        # clip_loss = clip_score_loss(denoise, batch['prompt'], clip_model)
        

        # Define weights for losses
        # total_loss = mse_loss + 0.5 * perceptual + 0.2 * clip_loss
        total_loss = mse_loss + 0.5 * perceptual
        self.log('train_loss', total_loss)
        return total_loss

    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, batch):
        latents = torch.cat([latents]*2)
        timestep = torch.cat([timestep]*2)
        
        R = torch.cat([batch['R']]*2)
        K = torch.cat([batch['K']]*2)
      
        meta = {
            'K': K,
            'R': R,
        }

        return latents, timestep, prompt_embd, meta

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model):
        latents, _timestep, _prompt_embd, meta = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch)

        noise_pred = model(
            latents, _timestep, _prompt_embd, meta)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference(batch)
        images = ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
      
        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_pred, images, batch['prompt'], batch_idx)

    @torch.no_grad()
    def inference(self, batch):
        images = batch['images']
        bs, m, h, w, _ = images.shape
        device = images.device

        latents= torch.randn(
            bs, m, 4, h//8, w//8, device=device)

        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = self.encode_text('', device)[0]
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]]*m, dim=1)

            noise_pred = self.forward_cls_free(
                latents, _timestep, prompt_embd, batch, self.mv_base_model)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
        images_pred = self.decode_latent(
            latents, self.vae)
       
        return images_pred
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images_pred = self.inference(batch)

        images = ((batch['images']/2+0.5)
                          * 255).cpu().numpy().astype(np.uint8)
       
       
        scene_id = batch['image_paths'][0].split('/')[2]
        image_id=batch['image_paths'][0].split('/')[-1].split('.')[0].split('_')[0]
        
        output_dir = batch['resume_dir'][0] if 'resume_dir' in batch else os.path.join(self.logger.log_dir, 'images')
        output_dir=os.path.join(output_dir, "{}_{}".format(scene_id, image_id))
        
        os.makedirs(output_dir, exist_ok=True)
        for i in range(images.shape[1]):
            path = os.path.join(output_dir, f'{i}.png')
            im = Image.fromarray(images_pred[0, i])
            im.save(path)
            im = Image.fromarray(images[0, i])
            path = os.path.join(output_dir, f'{i}_natural.png')
            im.save(path)
        with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
            for p in batch['prompt']:
                f.write(p[0]+'\n')

    @torch.no_grad()
    def save_image(self, images_pred, images, prompt, batch_idx):

        img_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        with open(os.path.join(img_dir, f'{self.global_step}_{batch_idx}.txt'), 'w') as f:
            for p in prompt:
                f.write(p[0]+'\n')
        if images_pred is not None:
            for m_i in range(images_pred.shape[1]):
                im = Image.fromarray(images_pred[0, m_i])
                im.save(os.path.join(
                    img_dir, f'{self.global_step}_{batch_idx}_{m_i}_pred.png'))
                if m_i < images.shape[1]:
                    im = Image.fromarray(
                        images[0, m_i])
                    im.save(os.path.join(
                        img_dir, f'{self.global_step}_{batch_idx}_{m_i}_gt.png'))
                    

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDIMScheduler

from modules.renderers.gaussians_renderer import GaussianRenderer

from utils import sample_from_dense_cameras

import tqdm

from .gs_utils import GaussiansManeger

class GSRefinerSDSPlusPlus(nn.Module):
    def __init__(self, 
            sd_model_key='stabilityai/stable-diffusion-2-1-base',
            local_files_only=True,
            num_views=1,
            total_iterations=500,
            guidance_scale=100,
            min_step_percent=0.02, 
            max_step_percent=0.75,
            lr_scale=1,
            lr_scale_end=1,
            lrs={'xyz': 0.0001, 'features': 0.01, 'opacity': 0.05, 'scales': 0.01, 'rotations': 0.01}, 
            use_lods=True,
            lambda_latent_sds=1,
            lambda_image_sds=0.01,
            lambda_image_variation=0,
            lambda_mask_variation=0, 
            lambda_mask_saturation=0,
            use_random_background_color=True,
            grad_clip=10,
            img_size=512,
            num_densifications=5,
            text_templete='$text$',
            negative_text_templete=''
        ):
        super().__init__()

        pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_key, local_files_only=True
        )

        pipe.enable_xformers_memory_efficient_attention()
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.requires_grad_(False)
        self.vae = pipe.vae.requires_grad_(False)
        self.unet = pipe.unet.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            sd_model_key, subfolder="scheduler", local_files_only=True
        )
        
        del pipe

        self.num_views = num_views
        self.total_iterations = total_iterations
        self.guidance_scale = guidance_scale
        self.lrs = {key: value * lr_scale for key, value in lrs.items()}
        self.lr_scale = lr_scale
        self.lr_scale_end = lr_scale_end

        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)

        self.device = 'cpu'

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.set_min_max_steps(min_step_percent, max_step_percent)

        self.renderer = GaussianRenderer()

        self.text_templete = text_templete
        self.negative_text_templete = negative_text_templete

        self.use_lods = use_lods

        self.lambda_latent_sds = lambda_latent_sds
        self.lambda_image_sds = lambda_image_sds

        self.lambda_image_variation = lambda_image_variation
        self.lambda_mask_variation = lambda_mask_variation

        self.lambda_mask_saturation = lambda_mask_saturation

        self.grad_clip = grad_clip
        self.img_size = img_size

        self.use_random_background_color = use_random_background_color

        self.opacity_threshold = 0.01
        self.densification_interval = self.total_iterations // (num_densifications + 1)

    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    def to(self, device):
        self.device = device
        return super().to(device)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation_strategy='longest_first',
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(inputs.input_ids.to(next(self.text_encoder.parameters()).device))[0]
        return text_embeddings
    
    # @torch.cuda.amp.autocast(enabled=False)
    def encode_image(self, images):
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    # @torch.cuda.amp.autocast(enabled=False)
    def decode_latent(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents).sample
        return images
    
    def train_step(
        self,
        images,
        t,
        text_embeddings,
        uncond_text_embeddings,
        learnable_text_embeddings,
    ):
        latents = self.encode_image(images)

        with torch.no_grad():
            B = latents.shape[0]
            t = t.repeat(self.num_views)
            
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

        if self.use_lods:
            with torch.enable_grad():
                noise_pred_learnable = self.unet(
                    latents_noisy, 
                    t, 
                    encoder_hidden_states=learnable_text_embeddings
                ).sample

            loss_embedding = F.mse_loss(noise_pred_learnable, noise, reduction="mean")
        else:
            noise_pred_learnable = noise
            loss_embedding = 0

        with torch.no_grad():
            noise_pred = self.unet(
                torch.cat([latents_noisy, latents_noisy], 0), 
                torch.cat([t, t], 0), 
                encoder_hidden_states=torch.cat([text_embeddings, uncond_text_embeddings], 0)
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)

            alpha = self.alphas_cumprod[t].view(-1, 1, 1, 1) ** 0.5
            sigma = (1 - self.alphas_cumprod[t].view(-1, 1, 1, 1)) ** 0.5

            latents_pred = (latents_noisy - sigma * (noise_pred - noise_pred_learnable + noise)) / alpha
            images_pred = self.decode_latent(latents_pred).clamp(-1, 1)

        loss_latent_sds = (F.mse_loss(latents, latents_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / B
        loss_image_sds = (F.mse_loss(images, images_pred, reduction="none").sum([1, 2, 3]) * w * alpha / sigma).sum() / B

        return loss_latent_sds, loss_image_sds, loss_embedding

    @torch.cuda.amp.autocast(enabled=True)
    @torch.enable_grad()
    def refine_gaussians(self, gaussians, text, dense_cameras):
        
        gaussians_original = gaussians
        xyz, features, opacity, scales, rotations = gaussians

        mask = opacity[..., 0] >= self.opacity_threshold

        xyz_original = xyz[mask][None]
        features_original = features[mask][None]
        opacity_original = opacity[mask][None]
        scales_original = scales[mask][None]
        rotations_original = rotations[mask][None]

        text = self.text_templete.replace('$text$', text)

        text_embeddings = self.encode_text([text])
        uncond_text_embeddings =  self.encode_text([self.negative_text_templete.replace('$text$', text)])

        class LearnableTextEmbeddings(nn.Module):
            def __init__(self, uncond_text_embeddings):
                super().__init__()
                self.embeddings = nn.Parameter(torch.zeros_like(uncond_text_embeddings.float().detach().clone()))
                self.to(self.embeddings.device)

            def forward(self, cameras):
                B = cameras.shape[1]
                return self.embeddings.repeat(B, 1, 1)

        _learnable_text_embeddings = LearnableTextEmbeddings(uncond_text_embeddings)

        text_embeddings = text_embeddings.repeat(self.num_views, 1, 1)
        uncond_text_embeddings = uncond_text_embeddings.repeat(self.num_views, 1, 1)

        optimizer_embeddings = torch.optim.Adam(_learnable_text_embeddings.parameters(), lr=self.lrs['embeddings'])

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(new_gaussians.optimizer, gamma=(self.lr_scale_end / self.lr_scale) ** (1 / self.total_iterations))

        for i in tqdm.trange(self.total_iterations, desc='Refining...'):

            if i % self.densification_interval == 0 and i != 0:
                new_gaussians.densify_and_prune()

            with torch.cuda.amp.autocast(enabled=False):   
                cameras = sample_from_dense_cameras(dense_cameras, torch.rand(1, self.num_views).to(self.device))

                learnable_text_embeddings = _learnable_text_embeddings(cameras)

                if self.lambda_mask_variation > 0 or self.lambda_image_variation > 0:
                    with torch.no_grad():
                        images_original, _, masks_original, _, _ = self.renderer(cameras, gaussians_original, bg_color='random', h=self.img_size, w=self.img_size)

                gaussians = new_gaussians()
                images_pred, depths_pred, masks_pred, reg_losses, _ = self.renderer(cameras, gaussians, bg_color='random', h=self.img_size, w=self.img_size)
            
            t = torch.full((1,), int((i / self.total_iterations) ** (1/2) * (self.min_step - self.max_step) + self.max_step), dtype=torch.long, device=self.device)
            # t = torch.randint(self.min_step, self.max_step, (self.num_views,), dtype=torch.long, device=self.device)
            loss_latent_sds, loss_img_sds, loss_embedding = self.train_step(images_pred.squeeze(0), t, text_embeddings, uncond_text_embeddings, learnable_text_embeddings)

            loss = loss_latent_sds * self.lambda_latent_sds + loss_img_sds * self.lambda_image_sds + loss_embedding

            if self.lambda_mask_variation > 0 or self.lambda_image_variation > 0:
                loss += self.lambda_mask_variation * F.mse_loss(masks_original, masks_pred, reduction='sum') / self.num_views
                loss += self.lambda_image_variation * F.mse_loss(images_original, images_pred, reduction='sum') / self.num_views

            if self.lambda_mask_saturation > 0:
                loss += self.lambda_mask_saturation * F.mse_loss(masks_pred, torch.ones_like(masks_pred), reduction='sum') / self.num_views

            # self.lambda_scale_regularization
            if True:
                scales = torch.exp(new_gaussians._scales)
                big_points_ws = scales.max(dim=1).values > 0.1
                loss += 10 * scales[big_points_ws].sum()
                
            loss.backward()

            new_gaussians.optimizer.step()
            new_gaussians.optimizer.zero_grad()

            optimizer_embeddings.step()
            optimizer_embeddings.zero_grad()

            lr_scheduler.step()
            
            for radii, viewspace_points in zip(self.renderer.radii, self.renderer.viewspace_points):
                visibility_filter = radii > 0
                new_gaussians.is_visible[visibility_filter] = 1
                new_gaussians.max_radii2D[visibility_filter] = torch.max(new_gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                new_gaussians.add_densification_stats(viewspace_points, visibility_filter)

        gaussians = new_gaussians()
        is_visible = new_gaussians.is_visible.bool()
        gaussians = [p[:, is_visible].detach() for p in gaussians]

        del new_gaussians
        return gaussians
