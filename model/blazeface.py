import torch
import torch.nn as nn

from model.blazenet import BlazeNet
from model.neck import BlazeNeck
from model.head import BlazeHead
from model.post_process import SSDBox
from model.loss import AnchorGeneratorSSD
from utils.nms import multiclass_nms

from icecream import ic


class BlazeFace(nn.Module):
    """
    BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs,
               see https://arxiv.org/abs/1907.05047
    """
    def __init__(self, cfg_backbone, cfg_neck, cfg_head, cfg_post, cfg_anchor, img_size):
        super(BlazeFace, self).__init__()
        self.backbone = BlazeNet(**cfg_backbone)
        self.is_neck = cfg_neck is not None
        if self.is_neck:
            self.neck = BlazeNeck(**cfg_neck)
        self.anchors_gen = AnchorGeneratorSSD(**cfg_anchor)
        cfg_head['num_priors'] = self.anchors_gen.num_priors
        self.priors = self.anchors_gen(img_size)
        self.blaze_head = BlazeHead(**cfg_head)
        self.post_process = SSDBox(**cfg_post)

        # self.load_weights('./weights/blazeface_1000e.pt')
        self.load_weights('./weights/blazeface_fpn_ssh_1000e.pt')

    def load_weights(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt, strict=True)
        print('=> loaded pretrained weights from path: {}.'.format(path))
        del ckpt

    def forward(self, inputs, targets=None):
        # Backbone
        feats = self.backbone(inputs)
        # neck
        if self.is_neck:
            feats = self.neck(feats)
        # head_feats = self.blaze_head(neck_feats)
        # return head_feats
        # blaze Head
        if self.training:
            return self.blaze_head(feats,
                                   targets,
                                   self.anchors.to(feats[0].device))
        else:
            return self.blaze_head(feats) # preds => [box, cls]
            # return preds, anchors

    def inference(self, inputs, anchors=None, img_size=None):
        """infer one image -> preds[0].shape: torch.Size([1, 22400, 4])
                              preds[1].shape: torch.Size([1, 22400, 2]) """
        if anchors is None:
            anchors = self.priors.to(inputs.device)
        preds = self(inputs)
        dets = self.post_process(preds, anchors, img_size)
        return dets
