import torch
import torch.nn as nn
import torch.nn.functional as F





# class SSDLoss(nn.Module):
#     """
#     SSDLoss
#
#     Args:
#         overlap_threshold (float32, optional): IoU threshold for negative bboxes
#             and positive bboxes, 0.5 by default.
#         neg_pos_ratio (float): The ratio of negative samples / positive samples.
#         loc_loss_weight (float): The weight of loc_loss.
#         conf_loss_weight (float): The weight of conf_loss.
#         prior_box_var (list): Variances corresponding to prior box coord, [0.1,
#             0.1, 0.2, 0.2] by default.
#     """
#
#     def __init__(self,
#                  overlap_threshold=0.5,
#                  neg_pos_ratio=3.0,
#                  loc_loss_weight=1.0,
#                  conf_loss_weight=1.0,
#                  prior_box_var=[0.1, 0.1, 0.2, 0.2]):
#         super(SSDLoss, self).__init__()
#         self.overlap_threshold = overlap_threshold
#         self.neg_pos_ratio = neg_pos_ratio
#         self.loc_loss_weight = loc_loss_weight
#         self.conf_loss_weight = conf_loss_weight
#         self.prior_box_var = [1. / a for a in prior_box_var]
#
#     def _bipartite_match_for_batch(self, gt_bbox, gt_label, prior_boxes, bg_index):
#         """
#         Args:
#             gt_bbox (Tensor): [B, N, 4]
#             gt_label (Tensor): [B, N, 1]
#             prior_boxes (Tensor): [A, 4]
#             bg_index (int): Background class index
#         """
#         batch_size, num_priors = gt_bbox.shape[0], prior_boxes.shape[0]
#         ious = iou_similarity(gt_bbox.reshape((-1, 4)), prior_boxes).reshape(
#             (batch_size, -1, num_priors))
#
#         # Calculate the number of object per sample.
#         num_object = (ious.sum(axis=-1) > 0).astype('int64').sum(axis=-1)
#
#         # For each prior box, get the max IoU of all GTs.
#         prior_max_iou, prior_argmax_iou = ious.max(axis=1), ious.argmax(axis=1)
#         # For each GT, get the max IoU of all prior boxes.
#         gt_max_iou, gt_argmax_iou = ious.max(axis=2), ious.argmax(axis=2)
#
#         # Gather target bbox and label according to 'prior_argmax_iou' index.
#         batch_ind = torch.arange(
#             0, batch_size, dtype=torch.int64).unsqueeze(-1).tile([1, num_priors])
#         prior_argmax_iou = torch.stack([batch_ind, prior_argmax_iou], dim=-1)
#         targets_bbox = torch.gather(gt_bbox, prior_argmax_iou)
#         targets_label = torch.gather(gt_label, prior_argmax_iou)
#         # Assign negative
#         bg_index_tensor = torch.full([batch_size, num_priors, 1], bg_index,
#                                       'int64')
#         targets_label = torch.where(
#             prior_max_iou.unsqueeze(-1) < self.overlap_threshold,
#             bg_index_tensor, targets_label)
#
#         # Ensure each GT can match the max IoU prior box.
#         for i in range(batch_size):
#             if num_object[i] > 0:
#                 targets_bbox[i] = torch.scatter(
#                     targets_bbox[i], gt_argmax_iou[i, :int(num_object[i])],
#                     gt_bbox[i, :int(num_object[i])])
#                 targets_label[i] = torch.scatter(
#                     targets_label[i], gt_argmax_iou[i, :int(num_object[i])],
#                     gt_label[i, :int(num_object[i])])
#
#         # Encode box
#         prior_boxes = prior_boxes.unsqueeze(0).tile([batch_size, 1, 1])
#         targets_bbox = bbox2delta(
#             prior_boxes.reshape([-1, 4]),
#             targets_bbox.reshape([-1, 4]), self.prior_box_var)
#         targets_bbox = targets_bbox.reshape([batch_size, -1, 4])
#
#         return targets_bbox, targets_label
#
#     def _mine_hard_example(self, conf_loss, targets_label, bg_index):
#         pos = (targets_label != bg_index).astype(conf_loss.dtype)
#         num_pos = pos.sum(axis=1, keepdim=True)
#         neg = (targets_label == bg_index).astype(conf_loss.dtype)
#
#         conf_loss = conf_loss.clone() * neg
#         loss_idx = conf_loss.argsort(axis=1, descending=True)
#         idx_rank = loss_idx.argsort(axis=1)
#         num_negs = []
#         for i in range(conf_loss.shape[0]):
#             cur_num_pos = num_pos[i]
#             num_neg = torch.clip(
#                 cur_num_pos * self.neg_pos_ratio, max=pos.shape[1])
#             num_negs.append(num_neg)
#         num_neg = torch.stack(num_negs).expand_as(idx_rank)
#         neg_mask = (idx_rank < num_neg).astype(conf_loss.dtype)
#
#         return (neg_mask + pos).astype('bool')
#
#     def forward(self, boxes, scores, gt_bbox, gt_label, prior_boxes):
#         boxes = torch.cat(boxes, dim=1)
#         scores = torch.cat(scores, dim=1)
#         gt_label = gt_label.unsqueeze(-1).astype('int64')
#         prior_boxes = torch.cat(prior_boxes, dim=0)
#         bg_index = scores.shape[-1] - 1
#
#         # Match bbox and get targets.
#         targets_bbox, targets_label = \
#             self._bipartite_match_for_batch(gt_bbox, gt_label, prior_boxes, bg_index)
#         targets_bbox.stop_gradient = True
#         targets_label.stop_gradient = True
#
#         # Compute regression loss.
#         # Select positive samples.
#         bbox_mask = (targets_label != bg_index).astype(boxes.dtype)
#         loc_loss = bbox_mask * F.smooth_l1_loss(
#             boxes, targets_bbox, reduction='none')
#         loc_loss = loc_loss.sum() * self.loc_loss_weight
#         # F.binary_cross_entropy_with_logits()
#         # Compute confidence loss.
#         conf_loss = F.softmax_with_cross_entropy(scores, targets_label)
#         # Mining hard examples.
#         label_mask = self._mine_hard_example(
#             conf_loss.squeeze(-1), targets_label.squeeze(-1), bg_index)
#         conf_loss = conf_loss * label_mask.unsqueeze(-1).astype(conf_loss.dtype)
#         conf_loss = conf_loss.sum() * self.conf_loss_weight
#
#         # Compute overall weighted loss.
#         normalizer = (targets_label != bg_index).astype('float32').sum().clip(
#             min=1)
#         loss = (conf_loss + loc_loss) / (normalizer + 1e-9)
#
#         return loss


# def bbox2delta(src_boxes, tgt_boxes, weights):
#     src_w = src_boxes[:, 2] - src_boxes[:, 0]
#     src_h = src_boxes[:, 3] - src_boxes[:, 1]
#     src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
#     src_ctr_y = src_boxes[:, 1] + 0.5 * src_h
#
#     tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
#     tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
#     tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
#     tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h
#
#     wx, wy, ww, wh = weights
#     dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
#     dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
#     dw = ww * torch.log(tgt_w / src_w)
#     dh = wh * torch.log(tgt_h / src_h)
#
#     deltas = torch.stack((dx, dy, dw, dh), dim=1)
#     return deltas


class MultiBoxLoss(nn.Module):
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

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = priors.size(0)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        # landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            # landms = targets[idx][:, 4:14].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            # landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        # pos1 = conf_t > zeros
        # num_pos_landm = pos1.long().sum(1, keepdim=True)
        # N1 = max(num_pos_landm.data.sum().float(), 1)
        # pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        # landm_p = landm_data[pos_idx1].view(-1, 10)
        # landm_t = landm_t[pos_idx1].view(-1, 10)
        # loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pos = conf_t != zeros
        conf_t[pos] = 1
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!


        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        # loss_landm /= N1

        return loss_l, loss_c  #, loss_landm
#

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        # landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        # landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]

    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior

    # TODO refactor: index best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):     # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    conf = labels[best_truth_idx]               # Shape: [num_priors, ] 此处为每一个anchor对应的label取出来
    conf[best_truth_overlap < threshold] = 0    # label as background   overlap<0.35的全部作为负样本
    loc = encode(matches, priors, variances)

    # matches_landm = landms[best_truth_idx]
    # landm = encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    # landm_t[idx] = landm


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def point_form(boxes):
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
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
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
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]
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
