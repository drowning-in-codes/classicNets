#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/22 下午9:08
#   lastModifiedTime:2024/6/22 下午9:07
#   file:train.py
#   software: classicNets
#

import argparse
import glob
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from constants import Configure, WeightedType
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from dataloader import GarbageData, get_train_transforms
from torch.utils.data import Dataset, DataLoader
from loss import dice_loss, dice_loss_multi_class
from models import LightUNet, UNet, NestedUNet
import re
from torch.utils.tensorboard import SummaryWriter
from metrics import Evaluator
import numpy as np
import segmentation_models_pytorch as smp
from loss import CombinedLoss

# 训练数据路径
train_path_dir = "/home/data/"
# 设置保存路径
save_base_dir = "/project/train/models/"
# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.getLogger().setLevel(logging.INFO)


def save_model(model, epoch=0):
    state_dict = model.state_dict()

    logging.info("save models to" + save_base_dir)
    save_path = save_base_dir + "model_{}.pth".format(epoch)
    # 保存模型
    torch.save(state_dict, save_path)


def load_checkpoint_model(model_path_base, instance_model, must_load_checkpoint=False):
    start_epoch = 0
    model_files = glob.glob(f"{model_path_base}/model_*.pth")
    if must_load_checkpoint:
        assert len(model_files) >= 1, "没有找到匹配的模型"
    else:
        if len(model_files) == 0:
            logging.info("------------------train from scratch------------------")
            instance_model.train()
            instance_model.to(device=device)
            return instance_model, start_epoch
    latest_model_file = max(model_files, key=lambda f: int(re.search(r'\d+', f).group()))
    start_epoch = int(re.search(r'\d+', latest_model_file).group())
    checkpoint = torch.load(os.path.join(model_path_base, latest_model_file))

    instance_model.load_state_dict(checkpoint)
    instance_model.train()
    instance_model.to(device=device)
    logging.info("------------------train at epoch {}------------------".format(start_epoch))
    return instance_model, start_epoch


def load_model(model_name="UNet"):
    if not Configure.THIRD_PARTY_MODEL:
        assert model_name in ["UNet", "LightUNet", "NestedUNet", "PAN"], "其他模型暂不支持"
        # 创建模型实例
        instance_model = eval(model_name)()
        model, start_epoch = load_checkpoint_model(save_base_dir, instance_model=instance_model,
                                                   must_load_checkpoint=False)
        return model, start_epoch
    third_party_models_name = ["Unet", "Unet++", "EfficientUNet++", "ResUnet", "ResUnet++", "MANet", "Linknet", "FPN",
                               "PSPNet", "PAN",
                               "DeepLabV3", "DeepLabV3+"]
    assert model_name in third_party_models_name, "segmentation-models-pytorch库目前支持这些模型"
    third_party_model = [smp.Unet, smp.UnetPlusPlus, smp.EfficientUnetPlusPlus, smp.ResUnet, smp.ResUnetPlusPlus,
                         smp.MAnet, smp.Linknet, smp.FPN, smp.PSPNet, smp.PAN, smp.DeepLabV3, smp.DeepLabV3Plus]
    model_mappings = {
        name: model
        for name, model in zip(third_party_models_name, third_party_model)
    }
    model = model_mappings[model_name](
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=Configure.NUM_CLASSES,  # model output channels (number of classes in your dataset)
    )
    model, start_epoch = load_checkpoint_model(save_base_dir, instance_model=model,
                                               must_load_checkpoint=False)
    return model, start_epoch


def train_model(amp_enabled=True, gradient_clipping=None, tensorboard_writer_enabled=False, evaluater_enabled=False):
    train_metrics = None
    grad_scaler = None
    writer = None
    model_name = "UNet"
    model, start_epoch = load_model(model_name)
    if tensorboard_writer_enabled:
        writer = SummaryWriter()
    # 创建权重
    weights = None
    if Configure.WEIGHTED:
        pixel_percentage = {
            "background": 51.148658,
            "algae": 0.064494,
            "dead_twigs_leaves": 0.012825,
            "garbage": 2.345101,
            "water": 46.428922
        }
        assert len(pixel_percentage.keys()) == Configure.NUM_CLASSES, "类别不对应"
        # 计算初始权重
        initial_weights = torch.tensor([1 / pixel_percentage[key] for key in pixel_percentage], dtype=torch.float32)
        if Configure.WEIGHTED_TYPE == WeightedType.class_type:
            # 标准化权重，使其和等于类别数
            weights = initial_weights / initial_weights.sum() * len(initial_weights)
        else:
            weights = initial_weights.softmax(dim=-1)
        weights = weights.to(device)
    if Configure.PLAIN_LOSS:
        criterion = nn.CrossEntropyLoss(
            weight=weights) if Configure.NUM_CLASSES > 1 else nn.BCEWithLogitsLoss()
    else:
        criterion = CombinedLoss()
    # 创建数据集实例
    transforms = get_train_transforms()
    dataset = GarbageData(train_path_dir, transforms)
    # 实例化一个GradScaler对象
    if amp_enabled:
        grad_scaler = torch.cuda.amp.GradScaler()
    # 创建数据加载器
    train_dataloader = DataLoader(dataset, batch_size=Configure.BATCH_SIZE, shuffle=True,
                                  num_workers=Configure.NUM_WORKERS, pin_memory=True)
    # lr_scheduler
    last_epoch = start_epoch if start_epoch > 0 else -1
    # optimizer = optim.Adam(model.parameters(), lr=Configure.LEARNING_RATE)
    if last_epoch == -1:
        optimizer = optim.AdamW(model.parameters(),
                                lr=Configure.LEARNING_RATE)
    else:
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': Configure.LEARNING_RATE}],
                                lr=Configure.LEARNING_RATE)
    # 需要 val_loss
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
    #                                           max_lr=Configure.LEARNING_RATE * 20,  # 最大学习率
    #                                           steps_per_epoch=len(train_dataloader),  # 每个epoch的迭代次数
    #                                           epochs=Configure.NUM_EPOCHES,  # 总共的epoch数
    #                                           pct_start=0.2)  # warm-up阶段所占比例
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Configure.STEP_SIZE, gamma=Configure.GAMMA,last_epoch=start_epoch if start_epoch > 0 else -1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=Configure.GAMMA,
                                                     last_epoch=last_epoch)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Configure.NUM_EPOCHES, eta_min=1e-6,
    #                                                        last_epoch=last_epoch)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=Configure.LEARNING_RATE / 5,
    #                                               max_lr=Configure.LEARNING_RATE * 20, cycle_momentum=False,
    #                                               last_epoch=last_epoch)
    model.to(device)
    logging.info("------------------------------start training {}-----------------------".format(model_name))
    # 查看指标
    if evaluater_enabled:
        train_metrics = Evaluator(Configure.NUM_CLASSES)
    for epoch in range(start_epoch, start_epoch + Configure.NUM_EPOCHES):
        if evaluater_enabled:
            train_metrics.reset()
        running_loss = 0.0
        pbar = tqdm(train_dataloader)
        for step, (images, masks) in enumerate(pbar):
            # clear cache
            torch.cuda.empty_cache()
            images = images.to(device)
            masks = masks.to(device)
            # 使用自动混合精度
            if amp_enabled:
                with (torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled)):
                    # 前向传播
                    outputs = model(images)  # [1,NUM_CLASS,H,W]
                    # 计算损失
                    if Configure.PLAIN_LOSS:
                        if Configure.NUM_CLASSES == 1:
                            loss = criterion(outputs.squeeze(1), masks.float())
                            loss += dice_loss_multi_class(F.sigmoid(outputs.squeeze(1)), masks.float(),
                                                          multiclass=False)
                        else:
                            loss = criterion(outputs, masks)
                            loss += dice_loss_multi_class(
                                F.softmax(outputs, dim=1).float(),
                                F.one_hot(masks, Configure.NUM_CLASSES).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    else:
                        loss = criterion(outputs, masks)

            else:
                # 前向传播
                outputs = model(images)
                # 计算损失
                if Configure.PLAIN_LOSS:
                    if Configure.NUM_CLASSES == 1:
                        loss = criterion(outputs.squeeze(1), masks.float())
                        loss += dice_loss_multi_class(F.sigmoid(outputs.squeeze(1)), masks.float(),
                                                      multiclass=False)
                    else:
                        loss = criterion(outputs, masks)
                        loss += dice_loss_multi_class(
                            F.softmax(outputs, dim=1).float(),
                            F.one_hot(masks, Configure.NUM_CLASSES).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                else:
                    loss = criterion(outputs, masks)

            # 计算metrics
            if evaluater_enabled:
                pred = outputs.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                train_metrics.add_batch(masks.cpu().numpy(), pred)
            # 清零梯度
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                # 缩放损失的梯度
                grad_scaler.scale(loss).backward()
            if amp_enabled and gradient_clipping is not None:
                # Unscales the gradients of optimizer's assigned params in-place
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            if amp_enabled:
                # 缩放梯度值
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                grad_scaler.step(optimizer)
                # Updates the scale for next iteration.
                grad_scaler.update()
            else:
                # 反向传播和优化
                loss.backward()
                optimizer.step()
            pbar.set_description_str(
                f"Epoch {epoch + 1}/{start_epoch + Configure.NUM_EPOCHES},step:{step + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            scheduler.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataloader)

        logging.info(f"Epoch {epoch + 1}/{Configure.NUM_EPOCHES}, Loss: {epoch_loss:.4f}")

        if (epoch + 1) % Configure.INTERVAL == 0:
            if evaluater_enabled:
                miou = train_metrics.Mean_Intersection_over_Union()[1]
                pix_acc = train_metrics.Pixel_Accuracy()
            if tensorboard_writer_enabled:
                writer.add_scalar('Loss/train', running_loss, epoch + 1)
                writer.add_scalar('IOU/train', miou, epoch + 1)
                writer.add_scalar('Pixel_Accuracy/train', pix_acc, epoch + 1)
                logging.info(
                    f"Epoch {epoch + 1}/{start_epoch + Configure.NUM_EPOCHES}, miou: {miou:.4f}, pix acc:{pix_acc:4f}")
            save_model(model, epoch + 1)

    save_model(model, Configure.NUM_EPOCHES)
    if tensorboard_writer_enabled:
        writer.close()


if __name__ == '__main__':
    train_model()
