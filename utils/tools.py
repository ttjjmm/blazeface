import numpy as np
import torch
import torch.nn.functional as F
from utils.flops_counter import get_model_complexity_info
from icecream import ic

# class BBoxPostProcess(object):
#     __shared__ = ['num_classes']
#     __inject__ = ['decode', 'nms']
#
#     def __init__(self, num_classes=80, decode=None, nms=None):
#         super(BBoxPostProcess, self).__init__()
#         self.num_classes = num_classes
#         self.decode = decode
#         self.nms = nms
#
#     def __call__(self, head_out, rois, im_shape, scale_factor):
#         """
#         Decode the bbox and do NMS if needed.
#
#         Args:
#             head_out (tuple): bbox_pred and cls_prob of bbox_head output.
#             rois (tuple): roi and rois_num of rpn_head output.
#             im_shape (Tensor): The shape of the input image.
#             scale_factor (Tensor): The scale factor of the input image.
#         Returns:
#             bbox_pred (Tensor): The output prediction with shape [N, 6], including
#                 labels, scores and bboxes. The size of bboxes are corresponding
#                 to the input image, the bboxes may be used in other branch.
#             bbox_num (Tensor): The number of prediction boxes of each batch with
#                 shape [1], and is N.
#         """
#         if self.nms is not None:
#             bboxes, score = self.decode(head_out, rois, im_shape, scale_factor)
#             bbox_pred, bbox_num, _ = self.nms(bboxes, score, self.num_classes)
#         else:
#             bbox_pred, bbox_num = self.decode(head_out, rois, im_shape,
#                                               scale_factor)
#         return bbox_pred, bbox_num
#
#     def get_pred(self, bboxes, bbox_num, im_shape, scale_factor):
#         """
#         Rescale, clip and filter the bbox from the output of NMS to
#         get final prediction.
#
#         Notes:
#         Currently only support bs = 1.
#
#         Args:
#             bboxes (Tensor): The output bboxes with shape [N, 6] after decode
#                 and NMS, including labels, scores and bboxes.
#             bbox_num (Tensor): The number of prediction boxes of each batch with
#                 shape [1], and is N.
#             im_shape (Tensor): The shape of the input image.
#             scale_factor (Tensor): The scale factor of the input image.
#         Returns:
#             pred_result (Tensor): The final prediction results with shape [N, 6]
#                 including labels, scores and bboxes.
#         """
#
#         if bboxes.shape[0] == 0:
#             bboxes = torch.tensor(
#                 np.array(
#                     [[-1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32'))
#             bbox_num = torch.tensor(np.array([1], dtype='int32'))
#
#         origin_shape = torch.floor(im_shape / scale_factor + 0.5)
#
#         origin_shape_list = []
#         scale_factor_list = []
#         # scale_factor: scale_y, scale_x
#         for i in range(bbox_num.shape[0]):
#             expand_shape = paddle.expand(origin_shape[i:i + 1, :],
#                                          [bbox_num[i], 2])
#             scale_y, scale_x = scale_factor[i][0], scale_factor[i][1]
#             scale = paddle.concat([scale_x, scale_y, scale_x, scale_y])
#             expand_scale = paddle.expand(scale, [bbox_num[i], 4])
#             origin_shape_list.append(expand_shape)
#             scale_factor_list.append(expand_scale)
#
#         self.origin_shape_list = paddle.concat(origin_shape_list)
#         scale_factor_list = paddle.concat(scale_factor_list)
#
#         # bboxes: [N, 6], label, score, bbox
#         pred_label = bboxes[:, 0:1]
#         pred_score = bboxes[:, 1:2]
#         pred_bbox = bboxes[:, 2:]
#         # rescale bbox to original image
#         scaled_bbox = pred_bbox / scale_factor_list
#         origin_h = self.origin_shape_list[:, 0]
#         origin_w = self.origin_shape_list[:, 1]
#         zeros = paddle.zeros_like(origin_h)
#         # clip bbox to [0, original_size]
#         x1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 0], origin_w), zeros)
#         y1 = paddle.maximum(paddle.minimum(scaled_bbox[:, 1], origin_h), zeros)
#         x2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 2], origin_w), zeros)
#         y2 = paddle.maximum(paddle.minimum(scaled_bbox[:, 3], origin_h), zeros)
#         pred_bbox = paddle.stack([x1, y1, x2, y2], axis=-1)
#         # filter empty bbox
#         keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
#         keep_mask = paddle.unsqueeze(keep_mask, [1])
#         pred_label = paddle.where(keep_mask, pred_label,
#                                   paddle.ones_like(pred_label) * -1)
#         pred_result = paddle.concat([pred_label, pred_score, pred_bbox], axis=1)
#         return pred_result
#
#     def get_origin_shape(self, ):
#         return self.origin_shape_list


class SSDBox(object):
    def __init__(self, is_normalized=True):
        self.is_normalized = is_normalized
        self.norm_delta = float(not self.is_normalized)

    def __call__(self,
                 preds,
                 prior_boxes,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes, scores = preds
        outputs = []
        ic(prior_boxes.shape)
        ic(scores.shape)

        ic(boxes.shape)
        prior_box = prior_boxes
        for box, score in zip(boxes, scores):
            print(prior_box.shape)
            pb_w = prior_box[:, 2] - prior_box[:, 0] + self.norm_delta
            pb_h = prior_box[:, 3] - prior_box[:, 1] + self.norm_delta
            pb_x = prior_box[:, 0] + pb_w * 0.5
            pb_y = prior_box[:, 1] + pb_h * 0.5
            out_x = pb_x + box[:, :, 0] * pb_w * 0.1
            out_y = pb_y + box[:, :, 1] * pb_h * 0.1
            out_w = torch.exp(box[:, :, 2] * 0.2) * pb_w
            out_h = torch.exp(box[:, :, 3] * 0.2) * pb_h

            if self.is_normalized:
                h = torch.unsqueeze(
                    im_shape[:, 0] / scale_factor[:, 0], dim=-1)
                w = torch.unsqueeze(
                    im_shape[:, 1] / scale_factor[:, 1], dim=-1)
                output = torch.stack([(out_x - out_w / 2.) * w,
                                      (out_y - out_h / 2.) * h,
                                      (out_x + out_w / 2.) * w,
                                      (out_y + out_h / 2.) * h], dim=-1)
            else:
                output = torch.stack([out_x - out_w / 2., out_y - out_h / 2., out_x + out_w / 2. - 1., out_y + out_h / 2. - 1.], dim=-1)
            outputs.append(output)
        boxes = torch.cat(outputs, dim=1)

        scores = F.softmax(torch.cat(scores, dim=-1))
        scores = torch.transpose(scores, 1, 2)

        return boxes, scores


def flops_info(model, input_shape=(3, 320, 320)):
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')