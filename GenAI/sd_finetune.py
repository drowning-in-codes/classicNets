#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/30 下午4:55
#   lastModifiedTime:2024/6/30 下午4:55
#   file:sd_finetune.py
#   software: classicNets
#

# @markdown ##  Import necessary packages
# @markdown It is recommmended NOT to change codes in this cell.
import argparse
import logging
import math
import os
import random
import glob
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import AutoProcessor, AutoModel, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.training_utils import compute_snr
from diffusers.utils.torch_utils import is_compiled_module
from deepface import DeepFace
import cv2

# Do not change the following parameters, or the process may crashed due to GPU out of memory.
output_folder = os.path.join(project_dir, "logs")  # 存放model checkpoints跟validation結果的資料夾
seed = 1126  # random seed
train_batch_size = 2  # training batch size
resolution = 512  # Image size
weight_dtype = torch.bfloat16  #
snr_gamma = 5
#####

# @markdown ## Important parameters for fine-tuning Stable Diffusion
pretrained_model_name_or_path = "stablediffusionapi/cyberrealistic-41"
lora_rank = 32
lora_alpha = 16
# @markdown ### ▶️ Learning Rate
# @markdown The learning rate is the most important for your results. If you want to train slower with lots of images, or if your dim and alpha are high, move the unet to 2e-4 or lower. <p>
# @markdown The text encoder helps your Lora learn concepts slightly better. It is recommended to make it half or a fifth of the unet. If you're training a style you can even set it to 0.
learning_rate = 1e-4  # @param {type:"number"}
unet_learning_rate = learning_rate
text_encoder_learning_rate = learning_rate
lr_scheduler_name = "cosine_with_restarts"  # 設定學習率的排程
lr_warmup_steps = 100  # 設定緩慢更新的步數
# @markdown ### ▶️ Steps
# @markdown Choose your training step and the number of generated images per each validaion
max_train_steps = 200  # @param {type:"slider", min:200, max:2000, step:100}
validation_prompt = "validation_prompt.txt"
validation_prompt_path = os.path.join(prompts_folder, validation_prompt)
validation_prompt_num = 3  # @param {type:"slider", min:1, max:5, step:1}
validation_step_ratio = 1  # @param {type:"slider", min:0, max:1, step:0.1}
with open(validation_prompt_path, "r") as f:
    validation_prompt = [line.strip() for line in f.readlines()]
#####


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSRIONS = ['jpg']

train_transform = transforms.Compose(
    [transforms.Resize(resolution, interpolation=transformers.InterpolationMode.BILINEAR),
     transforms.CenterCrop(resolution),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])
     ]
)


class Text2ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_folder, captions_folder, transform, tokenizer):
        self.image_paths = []
        for ext in IMAGE_EXTENSRIONS:
            self.image_paths.extend(glob.glob(f"{image_folder}/*{ext}"))
        self.image_paths = sorted(self.image_paths)
        self.train_emb = torch.tensor([DeepFace.represent(img_path, detector_backend="ssd", model_name="GhostFaceNet",
                                                          enforce_detection=False)[0]['embedding'] for img_path in
                                       self.image_paths])
        caption_paths = sorted(glob.glob(f"{captions_folder}/*txt"))
        captions = []
        for p in caption_paths:
            with open(p, "r") as f:
                captions.append(f.readline())
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                           return_tensors="pt")
        self.input_ids = inputs.input_ids
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        input_id = self.input_ids[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image)
        except Exception as e:
            print(f"Could not load image path:{img_path},error:{e}")
            return None
        return tensor, input_id

    def __len__(self):
        return len(self.image_paths)


def prepare_lora_model(pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", model_path=None):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer"
    )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     pretrained_model_name_or_path,
    #     subfolder="tokenizer"
    # )
    text_encoder = torch.load(os.path.join(model_path, "text_encoder.pt"))
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        subfolder="unet"
    )
    text_encoder = torch.load(os.path.join(model_path, "text_encoder.pt"))
    unet = torch.load(os.path.join(model_path, "unet.pt"))
    vae.requires_grad_(False)
    for name, param in unet.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    for name, parm in text_encoder.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    unet.to(DEVICE, dtype=weight_dtype)
    vae.to(DEVICE, dtype=weight_dtype)
    text_encoder.to(DEVICE, dtype=weight_dtype)
    return tokenizer, noise_scheduler, unet, vae, text_encoder


def prepare_optimizer(unet, text_encoder, unet_leraning_rate=5e-4, text_encoder_learning_rate=1e-4):
    unet_lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    text_encoder_lora_layers = list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    trainable_params = [
        {"param": unet_lora_layers, "lr": unet_leraning_rate},
        {"param": text_encoder_lora_layers, "lr": text_encoder_learning_rate},
    ]
    optimizer = torch.option.AdamW(
        trainable_params,
        lr=unet_learning_rate
    )
    return optimizer


def evaluate(pretrained_model_name_or_path, weight_type, seed, unet_path, text_encoder_path, validation_prompt,
             output_folder, train_emb):
    pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        safety_checker=None
    )
    pipeline.unet = torch.load(unet_path)
    pipeline.text_encoder = torch.load(text_encoder_path)
    pipeline = pipeline.to(DEVICE)
    clip_model_name = "openai/clip-vit-base-patch32"
    clip_model = AutoModel.from_pretrained(clip_model_name)
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)
    with torch.no_grad():
        generator = torch.Generator(device=DEVICE)
        generator = generator.manual_seed(seed)
        face_score = 0
        clop_score = 0
        mis = 0
        print("Generating validation pictures")
        images = []
        for i in range(0, len(validation_prompt), 4):
            images.extend(pipeline(validation_prompt[i:min(i + 4, len(validation_prompt))], num_inference_steps=30,
                                   generator=generator).images)
            print("Calculating validaion score ......")
            valid_emb = []
            for i, image in enumerate(tqdm(images)):
                save_file = f"{output_folder}/valid_image_{i}.png"
                image.save(save_file)
                opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                emb = DeepFace.represent(
                    opencvImage,
                    detector_backend="ssd",
                    model_name="GhostFaceNet",
                    enforce_detection=False,
                )
                if emb == [] or emb[0]['face_confidence'] == 0:
                    mis += 1
                    continue
                emb = emb[0]
                inputs = clip_processor(text=validation_prompt[i], images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                sim = outputs.logits_per_image
                clip_score += sim.item()
                valid_emb.append(emb['embedding'])
            if len(valid_emb) == 0:
                return 0, 0, mis
            valid_emb = torch.tensor(valid_emb)
            valid_emb = (valid_emb / torch.norm(valid_emb, p=2, dim=-1)[:, None]).cuda()
            train_emb = (train_emb / torch.norm(train_emb, p=2, dim=-1)[:, None]).cuda()
            face_score = torch.cdist(valid_emb, train_emb, p=2).mean().item()
            # face_score = torch.min(face_score, 1)[0].mean()
            clip_score /= len(validation_prompt) - mis
        return face_score, clip_score, mis


tokenizer, noise_scheduler, unet, vae, text_encoder = prepare_lora_model(pretrained_model_name_or_path, model_path)
optimizer = prepare_optimizer(unet, text_encoder, unet_learning_rate, text_encoder_learning_rate)
lr_scheduler = get_scheduler(
    lr_scheduler_name,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps,
    num_training_steps=max_train_steps,
    num_cycles=3
)

dataset = Text2ImageDataset(
    images_folder=images_folder,
    captions_folder=captions_folder,
    transform=train_transform,
    tokenizer=tokenizer,
)


def collate_fn(examples):
    pixel_values = []
    input_ids = []
    for tensor, input_id in examples:
        pixel_values.append(tensor)
        input_ids.append(input_id)
    pixel_values = torch.stack(pixel_values, dim=0).float()
    input_ids = torch.stack(input_ids, dim=0)
    return {"pixel_values": pixel_values, "input_ids": input_ids}


train_dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=train_batch_size,
    num_workers=8,
)
print("Preparation Finished!")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
progress_bar = tqdm(
    range(0, max_train_steps),
    initial=0,
    desc="Steps",
)
global_step = 0
num_epochs = math.ceil(max_train_steps / len(train_dataloader))
validation_step = int(max_train_steps * validation_step_ratio)
best_face_score = float("inf")
for epoch in range(num_epochs):
    unet.train()
    text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        if global_step >= max_train_steps:
            break
        latents = vae.encode(batch["pixel_values"].to(DEVICE, dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"].to(latents.device), return_dict=False)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        if not snr_gamma:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            snr = compute_snr(noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            if noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        global_step += 1

        if global_step % validation_step == 0 or global_step == max_train_steps:
            save_path = os.path.join(output_folder, f"checkpoint-last")
            unet_path = os.path.join(save_path, "unet.pt")
            text_encoder_path = os.path.join(save_path, "text_encoder.pt")
            print(f"Saving Checkpoint to {save_path} ......")
            os.makedirs(save_path, exist_ok=True)
            torch.save(unet, unet_path)
            torch.save(text_encoder, text_encoder_path)
            save_path = os.path.join(output_folder, f"checkpoint-{global_step + 1000}")
            os.makedirs(save_path, exist_ok=True)
            face_score, clip_score, mis = evaluate(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                weight_dtype=weight_dtype,
                seed=seed,
                unet_path=unet_path,
                text_encoder_path=text_encoder_path,
                validation_prompt=validation_prompt[:validation_prompt_num],
                output_folder=save_path,
                train_emb=dataset.train_emb
            )
            print("Step:", global_step, "Face Similarity Score:", face_score, "CLIP Score:", clip_score,
                  "Faceless Images:", mis)
            if face_score < best_face_score:
                best_face_score = face_score
                save_path = os.path.join(output_folder, f"checkpoint-best")
                unet_path = os.path.join(save_path, "unet.pt")
                text_encoder_path = os.path.join(save_path, "text_encoder.pt")
                os.makedirs(save_path, exist_ok=True)
                torch.save(unet, unet_path)
                torch.save(text_encoder, text_encoder_path)
print("Fine-tuning Finished!!!")
checkpoint_path = os.path.join(output_folder, f"checkpoint-last")  # 設定使用哪個checkpoint inference
unet_path = os.path.join(checkpoint_path, "unet.pt")
text_encoder_path = os.path.join(checkpoint_path, "text_encoder.pt")
inference_path = os.path.join(project_dir, "inference")
os.makedirs(inference_path, exist_ok=True)
train_image_paths = []
for ext in IMAGE_EXTENSIONS:
    train_image_paths.extend(glob.glob(f"{images_folder}/*{ext}"))
train_image_paths = sorted(train_image_paths)
train_emb = torch.tensor([DeepFace.represent(img_path, detector_backend="ssd", model_name="GhostFaceNet",
                                             enforce_detection=False)[0]['embedding'] for img_path in
                          train_image_paths])

face_score, clip_score, mis = evaluate(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    weight_dtype=weight_dtype,
    seed=seed,
    unet_path=unet_path,
    text_encoder_path=text_encoder_path,
    validation_prompt=validation_prompt,
    output_folder=inference_path,
    train_emb=train_emb,
)
print("Face Similarity Score:", face_score, "CLIP Score:", clip_score, "Faceless Images:", mis)
