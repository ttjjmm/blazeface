import torch
import torch.nn.functional as F
from utils.nms import multiclass_nms


class SSDBox(object):
    def __init__(self, iou_thr=0.01, nms_thr=0.4, is_normalized=True):
        self.iou_thr = iou_thr
        self.nms_thr = nms_thr

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
                 im_shape,
                 scale_factor):
        pred_boxes = self.decode(preds[0].squeeze(0), prior_boxes, [0.1, 0.2])
        pred_scores = preds[1].squeeze(0)


        # return boxes, scores