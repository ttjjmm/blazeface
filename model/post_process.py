import torch
import torch.nn.functional as F
from utils.nms import multiclass_nms


class SSDBox(object):
    def __init__(self, iou_thr=0.01, nms_thr=0.4, variance=None):
        if variance is None:
            variance = [0.1, 0.2]
        self.iou_thr = iou_thr
        self.nms_thr = nms_thr
        self.variance = variance

    @staticmethod
    def decode(loc, priors, variances):
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def __call__(self,
                 preds,
                 prior_boxes,
                 im_shape=None):
        if im_shape is None:
            im_shape = [640, 640]
        pred_boxes = self.decode(preds[0].squeeze(0), prior_boxes, self.variance)
        pred_scores = preds[1].squeeze(0)
        det_bboxes, det_labels = multiclass_nms(
            pred_boxes,
            pred_scores,
            score_thr=self.iou_thr,
            nms_cfg=dict(type='nms', iou_threshold=self.iou_thr),
            max_num=500)
        dets = det_bboxes * torch.tensor([im_shape[0], im_shape[1], im_shape[0], im_shape[1], 1],
                                         device=preds[0].device)
        return dets