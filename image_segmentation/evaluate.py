import logging

from tqdm import tqdm
import torch
import torch.nn as nn
from dataloader import GarbageData, get_test_transforms
from torch.utils.data import DataLoader
import numpy as np
from constants import Configure
from metrics import Evaluator, iou

# 训练数据路径
train_path_dir = "/home/data/1704"
# 设置保存路径
save_base_dir = "/project/train/models/"
logging.getLogger().setLevel(logging.INFO)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models import NestedUNet, UNet, LightUNet


def evaluate(model, criterion):
    model.eval()
    trans = get_test_transforms()
    dataset = GarbageData(train_path_dir, trans)
    val_dataloader = DataLoader(dataset, batch_size=1)
    validation_loss = 0
    print(len(val_dataloader))
    segmetric_val = Evaluator(Configure.NUM_CLASSES)
    with torch.no_grad():
        pbar = tqdm(val_dataloader)
        for image, label in pbar:
            torch.cuda.empty_cache()
            image = image.type(torch.HalfTensor).to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)

            pred = output.data.cpu().numpy()
            label = label.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            validation_loss += loss.item()
            segmetric_val.add_batch(label, pred)
            miou = np.mean(iou(pred, label, 1, 5))
            logging.info("miou:{:3f}".format(float(miou)))
    logging.info(
        "Val Pix Acc: {:.3f}".format(segmetric_val.Pixel_Accuracy()),
        "Val MIoU: {:.3f}".format(segmetric_val.Mean_Intersection_over_Union()[1]),
        "Val Loss: {:.3f}".format(validation_loss / len(val_dataloader)))


if __name__ == '__main__':
    model = NestedUNet().to(device)
    model.half()
    model.load_state_dict(torch.load(save_base_dir + "model_130.pth"), strict=True)
    evaluate(model, nn.CrossEntropyLoss())
