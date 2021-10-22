import torch
import torch.nn as nn

from model.blazenet import BlazeNet
from model.neck import BlazeNeck
from model.head import BlazeHead
from utils.tools import SSDBox
from utils.nms import multiclass_nms

# from icecream import ic


class BlazeFace(nn.Module):
    """
    BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs,
               see https://arxiv.org/abs/1907.05047
    """
    def __init__(self, cfg_backbone, cfg_neck, cfg_head, cfg_post_process):
        super(BlazeFace, self).__init__()
        self.backbone = BlazeNet(**cfg_backbone)
        self.is_neck = cfg_neck is not None
        if self.is_neck:
            self.neck = BlazeNeck(**cfg_neck)

        self.blaze_head = BlazeHead(**cfg_head)
        self.post_process = SSDBox()

        # self.load_weights('./weights/blazeface_1000e.pt')
        self.load_weights('./weights/blazeface_fpn_ssh_1000e.pt')

    def load_weights(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt, strict=True)
        print('=> loaded pretrained weights from path: {}.'.format(path))
        del ckpt

    def forward(self, inputs):
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
                                   inputs['gt_bbox'],
                                   inputs['gt_class'])
        else:
            preds, anchors = self.blaze_head(feats) # preds => [box, cls]
            return preds, anchors

    def inference(self, inputs):
        preds, anchors = self(inputs)
        # ic(anchors.shape)
        pred_boxes = decode(preds[0].squeeze(0), anchors, [0.1, 0.2])
        # ic(pred_boxes.shape)
        pred_scores = preds[1].squeeze(0)
        # ic(pred_scores.shape)

        det_bboxes, det_labels = multiclass_nms(
            pred_boxes,
            pred_scores,
            score_thr=0.1,
            nms_cfg=dict(type='nms', iou_threshold=0.4),
            max_num=1000)
        # ic(det_bboxes.shape, det_labels.shape)
        dets = det_bboxes * torch.tensor([640, 640, 640, 640, 1], device=inputs.device)

        return dets


    # def get_loss(self, ):
    #     return {"loss": self._forward()}
    #
    # def get_pred(self):
    #     bbox_pred, bbox_num = self._forward()
    #     output = {
    #         "bbox": bbox_pred,
    #         "bbox_num": bbox_num,
    #     }
    #     return output



# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """
    Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes