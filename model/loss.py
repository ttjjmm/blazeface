import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil
import six
from itertools import product as product
from icecream import ic

class AnchorGeneratorSSD(object):
    def __init__(self,
                 steps=None,
                 aspect_ratios=None,
                 min_sizes=None,
                 offset=0.5,
                 flip=True,
                 clip=False):
        if min_sizes is None:
            min_sizes = [[16, 24], [32, 48, 64, 80, 96, 128]]
        if aspect_ratios is None:
            aspect_ratios = [[1.], [1.]]
        if steps is None:
            steps = [8, 16]
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.min_sizes = min_sizes
        self.offset = offset
        self.flip = flip
        self.clip = clip

        self.num_priors = list()
        for item in self.min_sizes:
            self.num_priors.append(len(item))
        # for aspect_ratio, min_size, max_size in zip(aspect_ratios, self.min_sizes, self.max_sizes):
        #     if isinstance(min_size, (list, tuple)):
        #         self.num_priors.append(len(_to_list(min_size)) + len(_to_list(max_size)))
        #     else:
        #         self.num_priors.append((len(aspect_ratio) * 2 + 1) * len(_to_list(min_size)) + len(_to_list(max_size)))

    def __call__(self, image_size=(640, 640)):
        boxes = self.prior_box(self.min_sizes, self.steps, clip=False, image_size=image_size, offset=0.5)
        return boxes

    @staticmethod
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


class SSDLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self,
                 num_classes=2,
                 overlap_thresh=0.35,
                 prior_for_matching=True,
                 bkg_label=0,
                 neg_mining=True,
                 neg_pos=7,
                 neg_overlap=0.35,
                 encode_target=False):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes # num_classes default is 2: [face, bkg]
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]


    def priors_match_targets(self, targets, priors):
        """
        targets: torch.size([A, 5])
        priors: torch.size([B, 4])

        """
        loc = targets[:, :4]
        conf = targets[:, 4]

        overlaps = jaccard(
            loc,
            xywh2xyxy(priors)
        )

        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=False)
        # ignore hard gt
        valid_gt_idx = best_prior_overlap >= 0.2
        best_prior_idx_filter = best_prior_idx[valid_gt_idx]

        if best_prior_idx_filter.size(0) == 0:
            # loc_t[idx] = 0
            # conf_t[idx] = 0
            return None, None

        # [1,num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=False)
        # ic(best_truth_overlap.shape, best_truth_idx.shape)
        best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior

        # TODO refactor: index best_prior_idx with long tensor
        # ensure every gt matches with its prior of max overlap
        for j in range(best_prior_idx.size(0)):  # 判别此anchor是预测哪一个boxes
            best_truth_idx[best_prior_idx[j]] = j
        matches = loc[best_truth_idx]  # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
        conf = conf[best_truth_idx]    # Shape: [num_priors] 此处为每一个anchor对应的label取出来
        conf[best_truth_overlap < self.threshold] = 0  # label as background   overlap<0.35的全部作为负样本
        loc = encode(matches, priors, self.variance)

        return loc, conf


    def forward(self, predictions, targets, priors):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds
            priors (tensor): Prior Boxes, priors shape: torch.size(num_priors,4)
            targets (list): Ground truth boxes and labels for a batch, each batch's shape: list(torch.size(num_ogjs, 5), ...)
        """
        loc_data, conf_data = predictions
        device = loc_data.device
        batch_size = loc_data.size(0)
        prior_size = self.priors.size(0)

        # match priors (default boxes) and ground truth boxes
        # loc_t = torch.Tensor(batch_size, prior_size, 4)
        loc_t = torch.zeros((batch_size, prior_size, 4), device=device, dtype=torch.float32)
        # conf_t = torch.LongTensor(batch_size, prior_size)
        conf_t = torch.zeros((batch_size, prior_size), device=device, dtype=torch.int64)

        # iterate each batch
        for batch_idx in range(batch_size):
            # truths = targets[idx][:, :4]
            # labels = targets[idx][:, -1]
            # landms = targets[idx][:, 4:14].data
            loc, conf = self.priors_match_targets(targets[batch_idx], priors)
            loc_t[batch_idx] = loc
            conf_t[batch_idx] = conf

        pos_ind = conf_t != 0
        conf_t[pos_ind] = 1

        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        # pos1 = conf_t > zeros
        # num_pos_landm = pos1.long().sum(1, keepdim=True)
        # N1 = max(num_pos_landm.data.sum().float(), 1)
        # pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        # landm_p = landm_data[pos_idx1].view(-1, 10)
        # landm_t = landm_t[pos_idx1].view(-1, 10)
        # loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos_ind.unsqueeze(pos_ind.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos_ind.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos_ind.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos_ind.size(1) - 1)
        neg_ind = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos_ind.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg_ind.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos_ind + neg_ind).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        # loss_landm /= N1
        return loss_l, loss_c  #, loss_landm
#

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def xywh2xyxy(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,     # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]
    union = area_a + area_b - inter
    return inter / union  # [A, B]


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]



# if __name__ == '__main__':
#
#
#     loss = SSDLoss(device='cuda:0')
#
#
#     bbox = [
#         [0.5576172, 0.29589844, 0.5800781, 0.3251953, 1],
#         [0.46875, 0.29882812, 0.4951172, 0.3330078, 1],
#         [0.43359375, 0.2919922, 0.4560547, 0.32421875, 1],
#         [0.4189453, 0.2685547, 0.44140625, 0.296875, 1],
#         [0.39160156, 0.3310547, 0.41796875, 0.36523438, 1],
#         [0.3408203, 0.30371094, 0.36621094, 0.3330078, 1],
#         [0.3154297, 0.30664062, 0.34277344, 0.3359375, 1],
#     ]
#
#     pred = [torch.randn((1, 22400, 4), device='cuda:0'), torch.randn((1, 22400, 2), device='cuda:0')]
#     gt_box = torch.from_numpy(np.array(bbox))
#     gt_box = torch.unsqueeze(gt_box, dim=0)
#     loss(pred, gt_box.to('cuda:0'))


    # gt_label = torch.randn((4, 16, 1))
    #
    # loss = SSDLoss()
    # loss.bipartite_match_for_batch(gt_box, gt_label, prior)