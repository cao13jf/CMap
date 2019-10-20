import torch
import torch.nn as nn
import torch.nn.functional as F


#==================================================
#   different loss functions
#==================================================
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
    if len(target.shape) == 4:  # multiple class are combined in on volume
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
def AttentionMSELoss(output, target, eps=1e-5):
    target = target.float()
    MSE_wise = (output - target)**2
    attention_mask = (target >= 0).float()

    return (attention_mask * MSE_wise).sum() / (attention_mask.sum() + eps)

#   attention dice loss
#  generalized dice
def attention_dice_loss(output, target, mask, eps=1e-5, weight_type="square"):
    mask = mask.float()
    if len(target.shape) == 4:  # multiple class are combined in on volume
        n_class = output.shape[1]
        target = one_hot_encode(target, n_class).float()
        mask = mask.unsqueeze(1).repeat(1, n_class, 1, 1, 1)

    output = flatten(output)
    target = flatten(target)
    mask = flatten(mask)
    target_sum = (target * mask).sum(-1)
    if weight_type == "square":
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == "identity":
        class_weights = 1. / (target_sum + eps)
    elif weight_type == "sqrt":
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError("Unsupport weight type '{}' for generalized loss".format(weight_type))

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
