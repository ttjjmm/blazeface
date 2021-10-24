import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.loss import SSDLoss
# from icecream import ic


class BlazeHead(nn.Module):
    """
    Head block for Face detection network

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        anchor_generator(object): instance of anchor genertor method.
        kernel_size (int): kernel size of Conv2D in FaceHead.
        padding (int): padding of Conv2D in FaceHead.
        conv_decay (float): norm_decay (float): weight decay for conv layer weights.
        loss (object): loss of face detection model.
    """

    def __init__(self,
                 cfg_loss,
                 num_priors,
                 num_classes=1,
                 in_channels=(96, 96),
                 kernel_size=3,
                 padding=1):
        super(BlazeHead, self).__init__()
        # add background class
        self.num_classes = num_classes + 1
        self.in_channels = in_channels
        self.loss = SSDLoss(**cfg_loss)

        # if isinstance(anchor_generator, dict):
        #     self.anchor_generator = AnchorGeneratorSSD(**anchor_generator)
        self.num_priors = num_priors
        # print(self.num_priors)
        self.boxes = nn.ModuleList()
        self.scores = nn.ModuleList()
        # print(padding, in_channels)
        for i, num_prior in enumerate(self.num_priors):
            box_conv = nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=num_prior * 4,
                kernel_size=(kernel_size, kernel_size),
                padding=(padding, padding))
            self.boxes.append(box_conv)

            score_conv = nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=num_prior * self.num_classes,
                kernel_size=(kernel_size, kernel_size),
                padding=(padding, padding))
            self.scores.append(score_conv)

    def forward(self, feats, gt_bboxes=None, prior_boxes=None):
        box_preds = []
        cls_scores = []
        # prior_boxes = []
        for feat, box_conv, score_conv in zip(feats, self.boxes, self.scores):
            bs = feat.shape[0]
            box_pred = box_conv(feat)
            # box_pred -> [b, 2 * 4, w, h]
            box_pred = box_pred.permute(0, 2, 3, 1).contiguous()
            box_preds.append(box_pred.view(bs, -1, 4))
            cls_score = score_conv(feat)
            cls_score = cls_score.permute(0, 2, 3, 1).contiguous()
            cls_scores.append(cls_score.view(bs, -1, self.num_classes))
        box_preds = torch.cat(box_preds, dim=1)
        cls_scores = torch.cat(cls_scores, dim=1)

        # if not torch.onnx.is_in_onnx_export():
        #     prior_boxes = self.anchor_generator()
        #     device = box_preds.device
        #     prior_boxes = prior_boxes.to(device)
        # else:
        #     prior_boxes = None

        # for train
        if self.training:
            return self.get_loss((box_preds, cls_scores), gt_bboxes, prior_boxes)
        # for onnx export
        # elif torch.onnx.is_in_onnx_export():
        #     return box_preds, F.softmax(cls_scores, dim=-1)
        # # for inference
        # else:
        return box_preds, F.softmax(cls_scores, dim=-1)


    def get_loss(self, preds, targets, prior_boxes):
        return self.loss(preds, targets, prior_boxes)



