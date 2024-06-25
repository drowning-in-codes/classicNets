import json
import torch
from PIL import Image
from models import NestedUNet, LightUNet, UNet
import cv2
import json
import os
import glob
import logging
import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose
from constants import Configure
from utils import generate_all_objs, load_test_model, crop_image, stitch_masks
from dataloader import get_test_transforms

model_dir = "/project/train/models/"
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

HALF = Configure.ENABLE_HALF

logging.getLogger().setLevel(logging.INFO)


@torch.no_grad()
def init():
    """Initialize model
    Returns: model
    """
    model_name = "NestedUNet"
    assert model_name in ["UNet", "LightUNet", "NestedUNet", "PAN"], "其他模型暂不支持"
    model = eval(model_name)()
    logging.info("------------------------------start inference {}-----------------------".format(model_name))
    model, _ = load_test_model(model_dir, model, HALF)
    return model


def process_image(handle=None, input_image=None, args=None, **kwargs):
    """Do inference to analysis input_image and get output
    Attributes:
            handle: algorithm handle returned by init()
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args: string in JSON format, format: {
            "mask_output_path": "/path/to/output/mask.png"
        }
    Returns: process result
    """
    args = json.loads(args)
    mask_output_path = args.get("mask_output_path")
    # Process image here
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.uint8)  # [h,w,3]
    crops = crop_image(input_image)
    trans = get_test_transforms()
    masks = []
    with torch.no_grad():
        for crop in crops:
            if HALF:
                crop_result, _ = trans(crop, None)
                crop_result = crop_result.type(torch.HalfTensor).to(device)
            else:
                crop_result, _ = trans(crop, None)
                crop_result = crop_result.to(device)  # [3,H,W]
            if crop.ndim == 3:
                crop_result = crop_result.unsqueeze(0)  # [1,3,H,W]
            seg_mask = handle(crop_result)  # [1,5,H,W]
            masks.append(seg_mask)
    full_mask = stitch_masks(masks, input_image.shape[:2])
    pred_mask = np.argmax(full_mask, axis=1)  # [1,H,W]
    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    garbage_obj, all_objects = generate_all_objs(pred_mask)  # 检测垃圾
    target_count = len(garbage_obj)
    is_alert = True if target_count > 0 else False
    res = {
        "algorithm_data": {
            "is_alert": is_alert,
            "target_count": target_count,
            "target_info": garbage_obj
        },
        "model_data": {
            "objects": all_objects,
        },
    }
    if mask_output_path is not None and mask_output_path != "":
        # generate mask pic
        mask_data = pred_mask.astype(np.uint8)
        pred_mask_per_frame = Image.fromarray(mask_data[..., 0], mode='L')
        pred_mask_per_frame.save(mask_output_path)
        res["model_data"]["mask"] = mask_output_path

    # 删除不再需要的数据
    del crops
    del full_mask
    del masks
    del pred_mask
    del input_image
    torch.cuda.empty_cache()
    return json.dumps(res, indent=4)


if __name__ == '__main__':
    path_base = "/home/data/1704"
    img_files = glob.glob(os.path.join(path_base, "*.jpg"))
    mask_files = glob.glob(os.path.join(path_base, "*.png"))
    idx = 3
    image_data = cv2.imread(img_files[idx])
    mask_data = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
    print(mask_data.shape)

    res = process_image(init(), image_data, '{"mask_output_path":"./mask.png"}')
    print(res)
    mask_result = cv2.imread("./mask.png")
    print(mask_result.shape)
    garbage_obj, all_objects = generate_all_objs(mask_data)
    target_count = len(garbage_obj)
    is_alert = True if target_count > 0 else False
    gt_res = {
        "algorithm_data": {
            "is_alert": is_alert,
            "target_count": target_count,
            "target_info": garbage_obj
        },
        "model_data": {
            "objects": all_objects,
        },
    }
    print(gt_res)
