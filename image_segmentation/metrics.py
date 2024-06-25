import torch
import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    # 计算像素准确率
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    # 计算每一类IoU和MIoU
    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        return IoU, MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    # 加入数据
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    # 重置
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def pix_acc(outputs, targets, batch_size):
    """Pixel accuracy

    Args:
        outputs (torch.nn.Tensor): prediction outputs
        targets (torch.nn.Tensor): prediction targets
        batch_size (int): size of minibatch
    """
    acc = 0.
    for idx in range(batch_size):
        output = outputs[idx]
        target = targets[idx]
        correct = torch.sum(torch.eq(output, target).long())
        acc += correct / np.prod(np.array(output.shape)) / batch_size
    return acc.item()


def iou(outputs, targets, batch_size, n_classes):
    """Intersection over union

    Args:
        outputs (torch.nn.Tensor): prediction outputs
        targets (torch.nn.Tensor): prediction targets
        batch_size (int): size of minibatch
        n_classes (int): number of segmentation classes
    """
    eps = 1e-6
    class_iou = np.zeros(n_classes)
    for idx in range(batch_size):
        outputs_cpu = outputs[idx].cpu()
        targets_cpu = targets[idx].cpu()

        for c in range(n_classes):
            i_outputs = np.where(outputs_cpu == c)  # indices of 'c' in output
            i_targets = np.where(targets_cpu == c)  # indices of 'c' in target
            intersection = np.intersect1d(i_outputs, i_targets).size
            union = np.union1d(i_outputs, i_targets).size
            class_iou[c] += (intersection + eps) / (union + eps)

    class_iou /= batch_size

    return class_iou
