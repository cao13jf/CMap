import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


#==================================================
#   different loss functions
#==================================================

#  MSE loss
def mse_loss(output, target, *args):
    # bk_sum = (target == 0).sum([1, 2, 3]).float()
    # ft_sum = (target != 0).sum([1, 2, 3]).float()

    # #  get weights
    # if weight_type == "square":
    #     instance_weights = (bk_sum / (ft_sum + 1e-5))**2
    # elif weight_type == "identity":
    #     instance_weights = (bk_sum / (ft_sum + 1e-5))
    # elif weight_type == "sqrt":
    #     instance_weights = torch.sqrt((bk_sum / (ft_sum + 1e-5)))
    # else:
    #     raise ValueError("Unsupport weight type '{}' for generalized loss".format(weight_type))
    # weights = []
    # for i in range(target.shape[0]):
    #     weight0 = (target[i, ...] != 0).float()
    #     weight = weight0 * instance_weights[i]
    #     weight[weight0 == 0] = 1
    #     weights.append(weight)
    weights = (0.2 * (target - target.min()) + (target - target.min()).mean())
    loss = 0.5 * (weights * (target - output)**2).mean()

    return loss

#  Focal loss
def focal_loss(output, target, alpha=0.25, gamma=2.0):
    #  change data into [N, C]
    target = target.float()
    if output.dim() > 2:
        output = output.view(output.size[0], output.size[1], -1)
        output = output.transpose(1, 2)
        output = output.contiguous().view(-1, output.size(2))  # change into contiguous array
    if output.dim() == 5:
        target = target.contiguous().view(target.size[0],target.size[1], -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if output.dim() == 4:
        target.target.view(-1)

    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    return loss.mean()

#  dice loss
def dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

#  generalized dice
def generalized_dice_loss(output, target, eps=1e-5, weight_type="square"):
    if target.shape[1] == 1:  # multiple class are combined in on volume
        target = target.squeeze(dim=1)
        n_class = output.shape[1]
        target = expand_target(target, n_class, "softmax")

    output = flatten(output)
    target = flatten(target)
    target_sum = target.sum(-1)
    if weight_type == "square":
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == "identity":
        class_weights = 1. / (target_sum + eps)
    elif weight_type == "sqrt":
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError("Unsupport weight type '{}' for generalized loss".format(weight_type))

    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    return 1 - 2 * intersect_sum / denominator_sum

#  attention loss.
def attention_MSE_loss(output, target, mask, eps=1e-5, weight_type=None):
    mask = mask.float()
    n_class = output.shape[1]
    if len(target.shape) == 4:  # multiple class are combined in on volume
        #  get weight for soft MSE
        weight_dis = torch.abs(target.unsqueeze(1) - 0)
        for i in range(1, n_class):
            weight_dis = torch.cat((weight_dis, torch.abs(target.unsqueeze(1) - i)), dim=1)
        weight_dis = (weight_dis.float() + n_class)/ (n_class * 2)
        target = one_hot_encode(target, n_class).float()
        mask = mask.unsqueeze(1).repeat(1, n_class, 1, 1, 1)
    output, target, weight_dis, mask = flatten([output, target, weight_dis, mask])
    # target_sum = (target * mask).sum(-1)
    # class_weights = get_weight(target_sum, weight_type)
    # class_wegits = torch.tensor([1, 1, 1, 2, 2, 5]).float().to(output.device)

    return ((torch.abs(output - target) * weight_dis * mask)).sum() / (mask.sum() + eps)

#   attention dice loss
def attention_dice_loss(output, target, mask, eps=1e-5, weight_type="square"):
    mask = mask.float()
    if len(target.shape) == 4:  # multiple class are combined in on volume
        n_class = output.shape[1]
        target = one_hot_encode(target, n_class).float()
        mask = mask.unsqueeze(1).repeat(1, n_class, 1, 1, 1)

    output, target, mask = flatten([output, target, mask])
    target_sum = (target * mask).sum(-1)
    class_weights = get_weight(target_sum, weight_type)

    #  calculate generalized Dice
    intersect = (output * target * mask).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = ((output + target)* mask).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    return 1 - 2 * intersect_sum / denominator_sum

#===================================================
#  function used to calculate the prediction scaore
#===================================================
#   dice loss without softmax
def dice_score(predict, target, eps=1e-8):
    num = 2 * (predict * target).sum() + eps
    den = predict.sum() + target.sum() + eps
    return num / den

#  dice score after softmax
def softmax_dice_score(predict, target, eps=1e-8):
    pred = predict > 0
    targ = target > 0
    return dice_score(pred, targ)


#===================================================
#   functions used in loss
#===================================================
def expand_target(condensed_data, n_class, mode="softmax"):
    target_size = list(condensed_data.size())
    target_size.insert(1, n_class)
    target_shape = tuple(target_size)
    target_expand = torch.zeros(target_shape)
    if mode.lower() == "softmax":
        target_expand[:, 0, :, :, :] = (condensed_data == 0).float()
        target_expand[:, 1, :, :, :] = (condensed_data == 1).float()
    if mode.lower() == "sigmoid":
        target_expand[:, 1, :, :, :] = (condensed_data == 1).float()
        target_expand[:, 1, :, :, :] = (condensed_data == 2).float()
    return target_expand.to(condensed_data.device)

#  one hot encode
def one_hot_encode(condensed_data, n_class):
    target_size = list(condensed_data.size())
    target_size.insert(1, n_class)
    target_shape = tuple(target_size)
    target_expand = torch.zeros(target_shape)
    target_expand.scatter_(1, condensed_data.cpu().unsqueeze(1), 1)

    return target_expand.to(condensed_data.device)

def flatten(input):
    # has C channel
    assert input.dim() > 3, "Only support volume data flatten"
    if input.dim() == 5:
        C = input.size(1)
        axis_order = (1, 0) + tuple(range(2, input.dim()))
        input = input.permute(axis_order)
    else:
        C = 1
    return input.reshape(C, -1)

def get_weight(target_sum, weight_type):
    """
    Get weights for classes
    :param target_sum:  [C x N]  sum of pixels in each class
    :param weight_type:
    :return:
    """
    if weight_type == "square":
        class_weights = 1. / (target_sum * target_sum + 1e-5)
    elif weight_type == "identity":
        class_weights = 1. / (target_sum + 1e-5)
    elif weight_type == "sqrt":
        class_weights = 1. / (torch.sqrt(target_sum) + 1e-5)
    else:
        raise ValueError("Unsupport weight type '{}' for generalized loss".format(weight_type))

    return class_weights
