import torch

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output)
    num = target.size(0)
    output = output.view(num, -1)
    target = target.view(num, -1)
    intersection = (output * target)
    dice = (2. * intersection.sum(1) + smooth) / (output.sum(1) + target.sum(1) + smooth)
    dice_tensor = dice.sum() / num
    dice = dice_tensor.item()
    iou = dice / (2.-dice)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
