import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import ceil
import six
from itertools import product as product
from utils.loss import SSDLoss

from icecream import ic


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
                 cfg_anchor=None,
                 num_classes=1,
                 in_channels=(96, 96),
                 kernel_size=3,
                 padding=1,
                 loss='SSDLoss', **kwargs):
        super(BlazeHead, self).__init__()
        # add background class
        self.num_classes = num_classes + 1
        self.in_channels = in_channels
        self.anchor_generator = AnchorGeneratorSSD(**cfg_anchor)

        if loss == 'SSDLoss':
            self.loss = SSDLoss()

        # if isinstance(anchor_generator, dict):
        #     self.anchor_generator = AnchorGeneratorSSD(**anchor_generator)

        self.num_priors = self.anchor_generator.num_priors
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

    # @classmethod
    # def from_config(cls, cfg, input_shape):
    #     return {'in_channels': [i.channels for i in input_shape], }

    def forward(self, feats, gt_bbox=None, gt_class=None):
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
        if not torch.onnx.is_in_onnx_export():
            prior_boxes = self.anchor_generator()

        if self.training:
            return self.get_loss(box_preds, cls_scores, gt_bbox, gt_class, prior_boxes)
        elif torch.onnx.is_in_onnx_export():
            return box_preds, F.softmax(cls_scores, dim=-1)
        else:
            return (box_preds, F.softmax(cls_scores, dim=-1)), prior_boxes
    #
    # def get_loss(self, boxes, scores, gt_bbox, gt_class, prior_boxes):
    #     return self.loss(boxes, scores, gt_bbox, gt_class, prior_boxes)



def prior_box(min_sizes, steps, clip, image_size, offset):
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    anchors = []
    for k, f in enumerate(feature_maps):
        min_size = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            # print(i, j)
            for min_size_n in min_size:
                s_kx = min_size_n / image_size[1]
                s_ky = min_size_n / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + offset]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + offset]]
                for cy, cx in product(dense_cy, dense_cx):
                    # ic(cy, cx)
                    anchors += [cx, cy, s_kx, s_ky]
    # back to torch land
    output = torch.Tensor(anchors).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output


class AnchorGeneratorSSD(object):
    def __init__(self,
                 steps=[8, 16],
                 aspect_ratios=[[1.], [1.]],
                 min_sizes=[[16, 24], [32, 48, 64, 80, 96, 128]],
                 offset=0.5,
                 flip=True,
                 clip=False):
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_sizes = min_sizes
        self.offset = offset
        self.flip = flip
        self.clip = clip
        # print(self.min_sizes)


        self.num_priors = []
        for item in self.min_sizes:
            self.num_priors.append(len(item))
        # for aspect_ratio, min_size, max_size in zip(aspect_ratios, self.min_sizes, self.max_sizes):
        #     if isinstance(min_size, (list, tuple)):
        #         self.num_priors.append(len(_to_list(min_size)) + len(_to_list(max_size)))
        #     else:
        #         self.num_priors.append((len(aspect_ratio) * 2 + 1) * len(_to_list(min_size)) + len(_to_list(max_size)))


    def __call__(self):
        boxes = prior_box(self.min_sizes,self.steps, clip=False, image_size=(640, 640), offset=0.5)

        # for input, min_size, max_size, aspect_ratio, step in zip(
        #         inputs, self.min_sizes, self.max_sizes, self.aspect_ratios,
        #         self.steps):
        #     box, _ = ops.prior_box(
        #         input=input,
        #         image=image,
        #         min_sizes=_to_list(min_size),
        #         max_sizes=_to_list(max_size),
        #         aspect_ratios=aspect_ratio,
        #         flip=self.flip,
        #         clip=self.clip,
        #         steps=[step, step],
        #         offset=self.offset,
        #         min_max_aspect_ratios_order=self.min_max_aspect_ratios_order)
        #     boxes.append(paddle.reshape(box, [-1, 4]))
        return boxes


if __name__ == '__main__':
    anchor = AnchorGeneratorSSD(
        steps= [8, 16],
        aspect_ratios= [[1.], [1.]],
        min_sizes= [[16, 24], [32, 48, 64, 80, 96, 128]], # 1:8 2:16
        offset= 0.5,
        flip=False)
    print(anchor().shape)

    # cfg = {'steps': [8, 16],
    #     'aspect_ratios': [[1.], [1.]],
    #     'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]],
    #     'offset': 0.5,
    #     'flip':False}
    #
    # m = BlazeHead(cfg)
    # print(m)
