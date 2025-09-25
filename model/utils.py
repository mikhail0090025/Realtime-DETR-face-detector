import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms, utils

import numpy as np
from PIL import Image
import os
import math
import torchvision.models as models
from torchvision.ops import generalized_box_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_xmin = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_ymin = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_xmax = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_ymax = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)
    union_area = area1[:, None] + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou

def bbox_iou(box1, box2, eps=1e-7):
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    iou = inter / union
    return iou, area1, area2, union

def ciou_loss_xyxy(pred_xyxy, target_xyxy, eps=1e-7):
    """
    pred_xyxy, target_xyxy: (N,4) in xyxy format (can be normalized [0,1]).
    returns scalar loss = mean(1 - CIoU)
    """
    iou, _, _, _ = bbox_iou(pred_xyxy, target_xyxy, eps)

    # pred widths/heights
    p_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=eps)
    p_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=eps)
    g_w = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=eps)
    g_h = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=eps)

    # centers
    p_x = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    p_y = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    g_x = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
    g_y = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2

    # distance between centers
    center_dist = (p_x - g_x) ** 2 + (p_y - g_y) ** 2

    # smallest enclosing box
    cw = (torch.max(pred_xyxy[:, 2], target_xyxy[:, 2]) - torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])).clamp(min=eps)
    ch = (torch.max(pred_xyxy[:, 3], target_xyxy[:, 3]) - torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])).clamp(min=eps)
    c_diag = cw ** 2 + ch ** 2 + eps

    u = center_dist / c_diag

    # aspect ratio term
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(g_w / g_h) - torch.atan(p_w / p_h), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (u + alpha * v)
    loss = (1 - ciou).mean()
    return loss

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

def get_objects_on_image(image, model, device):
    classifier_output, mlp_output = model.forward_lite(image.to(device), 6, 4)
    classifier_output = classifier_output.squeeze(0).softmax(dim=-1)
    mlp_output = mlp_output.squeeze(0)
    threshold = 0.8
    mask = classifier_output[:, 1] > threshold
    classifier_output, mlp_output = classifier_output[mask], mlp_output[mask]

    _,  _, img_w, img_h = image.size()
    boxes = []
    for box in mlp_output.cpu().detach().numpy():
        cx, cy, w, h = box
        x0 = (cx - w/2) * img_w
        y0 = (cy - h/2) * img_h
        x1 = (cx + w/2) * img_w
        y1 = (cy + h/2) * img_h
        boxes.append([x0, y0, x1-x0, y1-y0])  # matplotlib patch: [x, y, width, height]

    return boxes, classifier_output[:, 1].cpu().detach().numpy()  # boxes, scores