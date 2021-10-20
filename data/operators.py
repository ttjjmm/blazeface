try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number, Integral
import cv2
import numpy as np
from icecream import ic

import matplotlib.pyplot as plt



__all__ = ['Pipeline', 'Resize', 'RandomDistort']


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass



class Pipeline(object):
    def __init__(self, operators: dict=None):
        assert isinstance(operators, dict), 'Wrong Data Augmentation Pipeline!'
        self.ops = [eval(k)(**v) for k, v in operators.items()]


    def __call__(self, data):
        """
        data format: {'image': np.array, 'gt_bbox': np.array}
        """
        for op in self.ops:
            data = op(data)
        return data


class Resize(object):
    def __init__(self, target_size, keep_ratio=True, keep_size=True, pad_value=(128.5, 128.5, 128.5), interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        self.keep_ratio = keep_ratio
        self.keep_size = keep_size
        self.pad_value = pad_value
        self.interp = interp
        self.pad_size = None
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale, pad_size):
        im_scale_x, im_scale_y = scale
        image = cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        # plt.imshow(image)
        # plt.show()
        if self.keep_size:
            h, w = image.shape[:2]
            ic(image.shape)
            ic(pad_size)
            canvas = np.ones((self.target_size[1], self.target_size[0], 3), dtype=np.float32)
            canvas *= np.array(self.pad_value, dtype=np.float32)
            canvas[pad_size[1]: pad_size[1] + h, pad_size[0]: pad_size[0] + w, :] = image
            return canvas
        else:
            return image


    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y

        if self.keep_size:
            bbox[:, 0::2] += self.pad_size[0]
            bbox[:, 1::2] += self.pad_size[1]

        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    # @staticmethod
    # def apply_keypoint(keypoints, scale, size):
    #     im_scale_x, im_scale_y = scale
    #     resize_w, resize_h = size


    def __call__(self, sample):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            pad_h = int((self.target_size[1] - resize_h) // 2)
            pad_w = int((self.target_size[0] - resize_w) // 2)
            self.pad_size = (pad_w, pad_h)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]
            pad_h = 0
            pad_w = 0
            self.pad_size = (pad_w, pad_h)
        # resize
        im = self.apply_image(sample['image'], (im_scale_x, im_scale_y), self.pad_size)

        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])
        return sample



# class RandomCrop(object):
#     """Random crop image and bboxes.
#     Args:
#         aspect_ratio (list): aspect ratio of cropped region.
#             in [min, max] format.
#         thresholds (list): iou thresholds for decide a valid bbox crop.
#         scaling (list): ratio between a cropped region and the original image.
#              in [min, max] format.
#         num_attempts (int): number of tries before giving up.
#         allow_no_crop (bool): allow return without actually cropping them.
#         cover_all_box (bool): ensure all bboxes are covered in the final crop.
#         is_mask_crop(bool): whether crop the segmentation.
#     """
#
#     def __init__(self,
#                  aspect_ratio=[.5, 2.],
#                  thresholds=[.0, .1, .3, .5, .7, .9],
#                  scaling=[.3, 1.],
#                  num_attempts=50,
#                  allow_no_crop=True,
#                  cover_all_box=False,
#                  is_mask_crop=False):
#         super(RandomCrop, self).__init__()
#         self.aspect_ratio = aspect_ratio
#         self.thresholds = thresholds
#         self.scaling = scaling
#         self.num_attempts = num_attempts
#         self.allow_no_crop = allow_no_crop
#         self.cover_all_box = cover_all_box
#         self.is_mask_crop = is_mask_crop
#
#
#
#     def apply(self, sample):
#         if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
#             return sample
#
#         h, w = sample['image'].shape[:2]
#         gt_bbox = sample['gt_bbox']
#
#         # NOTE Original method attempts to generate one candidate for each
#         # threshold then randomly sample one from the resulting list.
#         # Here a short circuit approach is taken, i.e., randomly choose a
#         # threshold and attempt to find a valid crop, and simply return the
#         # first one found.
#         # The probability is not exactly the same, kinda resembling the
#         # "Monty Hall" problem. Actually carrying out the attempts will affect
#         # observability (just like opening doors in the "Monty Hall" game).
#         thresholds = list(self.thresholds)
#         if self.allow_no_crop:
#             thresholds.append('no_crop')
#         np.random.shuffle(thresholds)
#
#         for thresh in thresholds:
#             if thresh == 'no_crop':
#                 return sample
#
#             for i in range(self.num_attempts):
#                 scale = np.random.uniform(*self.scaling)
#                 if self.aspect_ratio is not None:
#                     min_ar, max_ar = self.aspect_ratio
#                     aspect_ratio = np.random.uniform(
#                         max(min_ar, scale**2), min(max_ar, scale**-2))
#                     h_scale = scale / np.sqrt(aspect_ratio)
#                     w_scale = scale * np.sqrt(aspect_ratio)
#                 else:
#                     h_scale = np.random.uniform(*self.scaling)
#                     w_scale = np.random.uniform(*self.scaling)
#                 crop_h = h * h_scale
#                 crop_w = w * w_scale
#                 if self.aspect_ratio is None:
#                     if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
#                         continue
#
#                 crop_h = int(crop_h)
#                 crop_w = int(crop_w)
#                 crop_y = np.random.randint(0, h - crop_h)
#                 crop_x = np.random.randint(0, w - crop_w)
#                 crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
#                 iou = self._iou_matrix(
#                     gt_bbox, np.array(
#                         [crop_box], dtype=np.float32))
#                 if iou.max() < thresh:
#                     continue
#
#                 if self.cover_all_box and iou.min() < thresh:
#                     continue
#
#                 cropped_box, valid_ids = self._crop_box_with_center_constraint(
#                     gt_bbox, np.array(
#                         crop_box, dtype=np.float32))
#                 if valid_ids.size > 0:
#                     found = True
#                     break
#
#         return sample
#
#     def _iou_matrix(self, a, b):
#         tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
#         br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
#
#         area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
#         area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
#         area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
#         area_o = (area_a[:, np.newaxis] + area_b - area_i)
#         return area_i / (area_o + 1e-10)
#
#     def _crop_box_with_center_constraint(self, box, crop):
#         cropped_box = box.copy()
#
#         cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
#         cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
#         cropped_box[:, :2] -= crop[:2]
#         cropped_box[:, 2:] -= crop[:2]
#
#         centers = (box[:, :2] + box[:, 2:]) / 2
#         valid = np.logical_and(crop[:2] <= centers,
#                                centers < crop[2:]).all(axis=1)
#         valid = np.logical_and(
#             valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))
#
#         return cropped_box, np.where(valid)[0]
#
#     def _crop_image(self, img, crop):
#         x1, y1, x2, y2 = crop
#         return img[y1:y2, x1:x2, :]


# class RandomCrop(object):
#     """Random crop the image & bboxes & masks.
#
#     The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
#     then the cropped results are generated.
#
#     Args:
#         crop_size (tuple): The relative ratio or absolute pixels of
#             height and width.
#         crop_type (str, optional): one of "relative_range", "relative",
#             "absolute", "absolute_range". "relative" randomly crops
#             (h * crop_size[0], w * crop_size[1]) part from an input of size
#             (h, w). "relative_range" uniformly samples relative crop size from
#             range [crop_size[0], 1] and [crop_size[1], 1] for height and width
#             respectively. "absolute" crops from an input with absolute size
#             (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
#             crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
#             in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
#         allow_negative_crop (bool, optional): Whether to allow a crop that does
#             not contain any bbox area. Default False.
#         bbox_clip_border (bool, optional): Whether clip the objects outside
#             the border of the image. Defaults to True.
#
#     Note:
#         - If the image is smaller than the absolute crop size, return the
#             original image.
#         - The keys for bboxes, labels and masks must be aligned. That is,
#           `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
#           `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
#           `gt_masks_ignore`.
#         - If the crop does not contain any gt-bbox region and
#           `allow_negative_crop` is set to False, skip this image.
#     """
#
#     def __init__(self,
#                  crop_size,
#                  crop_type='absolute',
#                  allow_negative_crop=False,
#                  bbox_clip_border=True):
#         if crop_type not in [
#                 'relative_range', 'relative', 'absolute', 'absolute_range'
#         ]:
#             raise ValueError(f'Invalid crop_type {crop_type}.')
#         if crop_type in ['absolute', 'absolute_range']:
#             assert crop_size[0] > 0 and crop_size[1] > 0
#             assert isinstance(crop_size[0], int) and isinstance(
#                 crop_size[1], int)
#         else:
#             assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
#         self.crop_size = crop_size
#         self.crop_type = crop_type
#         self.allow_negative_crop = allow_negative_crop
#         self.bbox_clip_border = bbox_clip_border
#         # The key correspondence from bboxes to labels and masks.
#         self.bbox2label = {
#             'gt_bboxes': 'gt_labels',
#             'gt_bboxes_ignore': 'gt_labels_ignore'
#         }
#         self.bbox2mask = {
#             'gt_bboxes': 'gt_masks',
#             'gt_bboxes_ignore': 'gt_masks_ignore'
#         }
#
#     def _crop_data(self, results, crop_size, allow_negative_crop):
#         """Function to randomly crop images, bounding boxes, masks, semantic
#         segmentation maps.
#
#         Args:
#             results (dict): Result dict from loading pipeline.
#             crop_size (tuple): Expected absolute size after cropping, (h, w).
#             allow_negative_crop (bool): Whether to allow a crop that does not
#                 contain any bbox area. Default to False.
#
#         Returns:
#             dict: Randomly cropped results, 'img_shape' key in result dict is
#                 updated according to crop size.
#         """
#         assert crop_size[0] > 0 and crop_size[1] > 0
#         for key in results.get('img_fields', ['img']):
#             img = results[key]
#             margin_h = max(img.shape[0] - crop_size[0], 0)
#             margin_w = max(img.shape[1] - crop_size[1], 0)
#             offset_h = np.random.randint(0, margin_h + 1)
#             offset_w = np.random.randint(0, margin_w + 1)
#             crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
#             crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
#
#             # crop the image
#             img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
#             img_shape = img.shape
#             results[key] = img
#         results['img_shape'] = img_shape
#
#         # crop bboxes accordingly and clip to the image boundary
#         for key in results.get('bbox_fields', []):
#             # e.g. gt_bboxes and gt_bboxes_ignore
#             bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
#                                    dtype=np.float32)
#             bboxes = results[key] - bbox_offset
#             if self.bbox_clip_border:
#                 bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
#                 bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
#             valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
#                 bboxes[:, 3] > bboxes[:, 1])
#             # If the crop does not contain any gt-bbox area and
#             # allow_negative_crop is False, skip this image.
#             if (key == 'gt_bboxes' and not valid_inds.any()
#                     and not allow_negative_crop):
#                 return None
#             results[key] = bboxes[valid_inds, :]
#             # label fields. e.g. gt_labels and gt_labels_ignore
#             label_key = self.bbox2label.get(key)
#             if label_key in results:
#                 results[label_key] = results[label_key][valid_inds]
#
#             # mask fields, e.g. gt_masks and gt_masks_ignore
#             mask_key = self.bbox2mask.get(key)
#             if mask_key in results:
#                 results[mask_key] = results[mask_key][
#                     valid_inds.nonzero()[0]].crop(
#                         np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
#
#         # crop semantic seg
#         for key in results.get('seg_fields', []):
#             results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]
#
#         return results
#
#     def _get_crop_size(self, image_size):
#         """Randomly generates the absolute crop size based on `crop_type` and
#         `image_size`.
#
#         Args:
#             image_size (tuple): (h, w).
#
#         Returns:
#             crop_size (tuple): (crop_h, crop_w) in absolute pixels.
#         """
#         h, w = image_size
#         if self.crop_type == 'absolute':
#             return (min(self.crop_size[0], h), min(self.crop_size[1], w))
#         elif self.crop_type == 'absolute_range':
#             assert self.crop_size[0] <= self.crop_size[1]
#             crop_h = np.random.randint(
#                 min(h, self.crop_size[0]),
#                 min(h, self.crop_size[1]) + 1)
#             crop_w = np.random.randint(
#                 min(w, self.crop_size[0]),
#                 min(w, self.crop_size[1]) + 1)
#             return crop_h, crop_w
#         elif self.crop_type == 'relative':
#             crop_h, crop_w = self.crop_size
#             return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
#         elif self.crop_type == 'relative_range':
#             crop_size = np.asarray(self.crop_size, dtype=np.float32)
#             crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
#             return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
#
#     def __call__(self, results):
#         """Call function to randomly crop images, bounding boxes, masks,
#         semantic segmentation maps.
#
#         Args:
#             results (dict): Result dict from loading pipeline.
#
#         Returns:
#             dict: Randomly cropped results, 'img_shape' key in result dict is
#                 updated according to crop size.
#         """
#         image_size = results['img'].shape[:2]
#         crop_size = self._get_crop_size(image_size)
#         results = self._crop_data(results, crop_size, self.allow_negative_crop)
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(crop_size={self.crop_size}, '
#         repr_str += f'crop_type={self.crop_type}, '
#         repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
#         repr_str += f'bbox_clip_border={self.bbox_clip_border})'
#         return repr_str


class RandomFlip(object):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    # def apply_keypoint(self, gt_keypoint, width):
    #     for i in range(gt_keypoint.shape[1]):
    #         if i % 2 == 0:
    #             old_x = gt_keypoint[:, i].copy()
    #             gt_keypoint[:, i] = width - old_x
    #     return gt_keypoint
    @staticmethod
    def apply_image(image):
        return image[:, ::-1, :]

    @staticmethod
    def apply_bbox(bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def __call__(self, sample):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)

            # if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
            #     sample['gt_keypoint'] = self.apply_keypoint(
            #         sample['gt_keypoint'], width)

            sample['flipped'] = True
            sample['image'] = im

        return sample



class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 bbox_clip_border=True):
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]

        return results


    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return min(self.crop_size[0], h), min(self.crop_size[1], w)
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)

        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results




class RandomDistort(object):
    """Random color distortion.
    Args:
        hue (list): hue settings. in [lower, upper, probability] format.
        saturation (list): saturation settings. in [lower, upper, probability] format.
        contrast (list): contrast settings. in [lower, upper, probability] format.
        brightness (list): brightness settings. in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        count (int): the number of doing distrot
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, sample):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = self.apply_contrast(img)

        img = self.apply_saturation(img)
        img = self.apply_hue(img)

        if not mode:
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img = cv2.imread('/home/ubuntu/Documents/pycharm/blazeface/samples/test.jpg')

    kyw = {
        'Resize': {'target_size': (640, 640), 'keep_ratio': True},
        'RandomFlip': {'prob': 0.5},
    }
    p = Pipeline(operators=kyw)
    data = {'image': img}
    data = p(data)
    # r = Resize([640, 640])
    # p = Pad([640, 640])
    # data = r(data)
    # data = p(data)
    plt.imshow(data['image'].astype(np.uint8))
    plt.show()
    print(data['image'].shape)


# if __name__ == '__main__':

#     p = Pipeline(operators=kyw)


