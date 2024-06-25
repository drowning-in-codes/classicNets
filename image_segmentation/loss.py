from torch.nn import functional as F
import torch
from torch import Tensor
from segmentation_models_pytorch import losses
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, mode="multiclass", focal_loss_enable=True, jaccard_loss_enabled=True):
        super().__init__()
        self.dice_criterion = losses.DiceLoss(mode)
        self.lovasz_criterion = losses.LovaszLoss(mode)
        self.sce_criterion = losses.SoftCrossEntropyLoss()
        self.focal_loss_enable = focal_loss_enable
        self.jaccard_loss_enabled = jaccard_loss_enabled
        if focal_loss_enable:
            self.focal_criterion = losses.FocalLoss(mode)
        if jaccard_loss_enabled:
            self.jaccard_criterion = losses.JaccardLoss(mode)

    def forward(self, pred, target):
        dice_l = self.dice_criterion(pred, target)
        lovasz_l = self.lovasz_criterion(pred, target)
        sce_l = self.sce_criterion(pred, target)
        loss = dice_l + lovasz_l + sce_l
        if self.focal_loss_enable:
            focal_l = self.focal_criterion(pred, target)
            loss += focal_l
        if self.jaccard_loss_enabled:
            jaccard_l = self.jaccard_criterion(pred, target)
            loss += jaccard_l
        return loss


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss_multi_class(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def dice_loss(pred, target, smooth=1.):
    """
    Dice loss for single class
    """
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class Weighted_Cross_Entropy_Loss(torch.nn.Module):
    """Cross entropy loss that uses weight maps."""

    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target, weights):
        n, c, H, W = pred.shape
        # # Calculate log probabilities
        logp = F.log_softmax(pred, dim=1)

        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(n, 1, H, W))

        # Multiply with weights
        weighted_logp = (logp * weights).view(n, -1)

        # Rescale so that loss is in approx. same interval
        weighted_loss = weighted_logp.sum(1) / weights.view(n, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_loss.mean()

        return weighted_loss
