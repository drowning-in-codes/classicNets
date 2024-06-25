import logging
import glob
import json
import torch
from PIL import Image
from models import UNet
import cv2
import json
import os
import re
import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose
from constants import Configure

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
mappings = {1: "algae", 2: "dead_twigs_leaves", 3: "garbage", 4: "water"}
mappings_num_obj = {1: 2, 2: 2, 3: 5, 4: 1}


def detect(pred_mask, detect_obj_index=3):
    # pred_mask [H,W,1]
    binary_mask = np.where(pred_mask == detect_obj_index, 1, 0).astype(np.uint8) * 255
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    # 遍历连通域分析结果，获取每个区域的掩码
    # 没有连通的当作新的物体
    masks = []
    # 获取连通组件的面积
    areas = stats[:, cv2.CC_STAT_AREA]
    # 根据面积大小进行排序
    sorted_indices = np.argsort(areas)[::-1]  # 降序排列的索引
    min_num_labels = min(num_labels, mappings_num_obj[detect_obj_index])
    index = 1
    for label in sorted_indices:
        if label == 0:
            """
            表示背景,跳过
            """
            continue
        # 获得每个不连通的物体
        # region_mask = np.uint8(labels == label) * 255  # 获取当前标签对应的区域掩码
        region_stats = stats[label]  # 获取当前标签对应的区域统计信息
        # 将面积较小的东西排除掉
        width, height = region_stats[2:4]
        if width * height < Configure.EXCLUDE_AREA:
            break
        masks.append(region_stats)
        if index >= min_num_labels:
            break
        index += 1

    object_infos = []
    for region_idx, region_stats in enumerate(masks):
        x, y = region_stats[:2]
        width, height = region_stats[2:4]
        area_ratio = float(region_stats[cv2.CC_STAT_AREA] / (width * height))
        object_info = {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "name": mappings[detect_obj_index],
            "area_ratio": area_ratio
        }
        object_infos.append(object_info)

    return object_infos


def generate_all_objs(pred_mask):
    objects_infos = []
    garbage_data = []
    for idx, obj_type in mappings.items():
        obj_info = detect(pred_mask, idx)
        if idx == 3:
            garbage_data = obj_info
        objects_infos.extend(obj_info)

    return garbage_data, objects_infos


def load_model(model_path_base, instance_model, must_load_checkpoint=False):
    start_epoch = 0
    model_files = glob.glob(f"{model_path_base}/model_*.pth")

    if must_load_checkpoint:
        assert len(model_files) >= 1, "没有找到匹配模型"
    else:
        if len(model_files) == 0:
            logging.info("train from scratch")
            instance_model.train()
            instance_model.to(device=device)
            return instance_model, start_epoch

    latest_model_file = max(model_files, key=lambda f: int(re.search(r'\d+', f).group()))
    start_epoch = int(re.search(r'\d+', latest_model_file).group())
    checkpoint = torch.load(os.path.join(model_path_base, latest_model_file))

    instance_model.load_state_dict(checkpoint)
    instance_model.train()
    instance_model.to(device=device)
    logging.info("train from epoch {}".format(start_epoch))

    return instance_model, start_epoch


def load_test_model(model_path_base, instance_model, enable_HALF=True, must_load_checkpoint=False):
    start_epoch = 0
    model_files = glob.glob(f"{model_path_base}/model_*.pth")

    if must_load_checkpoint:
        assert len(model_files) >= 1, "没有找到匹配模型"
    else:
        if len(model_files) == 0:
            logging.info("------------------inference from scratch------------------")
            if enable_HALF:
                instance_model.half()
            instance_model.eval()
            instance_model.to(device=device)
            return instance_model, start_epoch

    latest_model_file = max(model_files, key=lambda f: int(re.search(r'\d+', f).group()))
    start_epoch = int(re.search(r'\d+', latest_model_file).group())
    checkpoint = torch.load(os.path.join(model_path_base, latest_model_file))

    instance_model.load_state_dict(checkpoint)
    if enable_HALF:
        instance_model.half()
    instance_model.eval()
    instance_model.to(device=device)
    logging.info("------------------inference at epoch {}------------------".format(start_epoch))

    return instance_model, start_epoch


def stitch_masks(masks, original_shape, crop_size=Configure.GRID_SIZE, num_classes=Configure.NUM_CLASSES):
    height, width = original_shape
    full_mask = np.zeros((1, num_classes, height, width), dtype=np.uint8)
    index = 0
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            mask = masks[index].cpu().numpy()
            h = min(crop_size, height - y)
            w = min(crop_size, width - x)
            full_mask[:, :, y:y + h, x:x + w] = mask[:, :, :h, :w]
            index += 1
    return full_mask


def crop_image(image, crop_size=Configure.GRID_SIZE):
    # image [H,W,3]
    height, width, channels = image.shape
    crops = []
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            h = min(crop_size, height - y)
            w = min(crop_size, width - x)
            crop = image[y:y + h, x:x + w, :]
            pad_h = crop_size - h
            pad_w = crop_size - w
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            crops.append(crop)
    return crops
