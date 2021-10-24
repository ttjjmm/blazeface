import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from icecream import ic


from data.operators import Pipeline

from tqdm import tqdm

#TODO
# 1. make cache for annotation
# 2. handle data with mask face
# 3. handle Two categories such as mask, no_mask


class WiderFaceDataset(Dataset):
    """
    only for eval and train
    """

    CLASSES = ('FG',)

    def __init__(self, data_path, pipeline, img_size=(640, 640), mode='train', min_size=None, with_kp=True, cache_label=False):
        super(WiderFaceDataset, self).__init__()
        assert mode in ['train', 'val'], 'dataset mode implement error!'

        self.img_path = os.path.join(data_path, mode, 'images')
        self.label_path = os.path.join(data_path, mode, 'labelv2.txt')
        self.img_size = img_size
        self.with_kp = with_kp
        self.mode = mode
        self.min_size = min_size if mode != 'val' else None  # rule out small faces
        self.NK = 5
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        # self.test_mode = mode == 'test'
        self.cache_path = os.path.join(data_path, 'label.cache')

        self.data_infos = self.load_annotations(self.label_path)

        # print(self.anno_info)
        # print(self.label_path, self.img_path)
        # self.load_annotations(self.label_path)

        # kyw = {
        #     'Resize': {'target_size': (640, 640), 'keep_ratio': True},
        #     'RandomFlip': {'prob': 0.5},
        # }
        self.aug_pipeline = Pipeline(pipeline)


    def _cache_labels(self):
        # check path
        if os.path.exists(self.label_path):
            # load cache directly
            pass
        else:
            # cache labels
            pass

    def _parse_ann_line(self, line):
        """
        Parse one line for one face
        Args:
            line:

        Returns:

        """
        values = [float(x) for x in line.strip().split()]
        bbox = np.array(values[0:4], dtype=np.float32)

        ignore = False
        if self.min_size is not None:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.min_size or h < self.min_size:
                ignore = True
        if self.with_kp:
        # for facial keypoints
            kps = np.zeros((self.NK, 3), dtype=np.float32)
            if len(values) > 4:
                if len(values) > 5:
                    # print(values)
                    kps = np.array(values[4:19], dtype=np.float32).reshape((self.NK, 3))
                    for li in range(kps.shape[0]):
                        if (kps[li, :] == -1).all():
                            # assert kps[li][2]==-1
                            kps[li][2] = 0.0  # weight = 0, ignore
                        else:
                            assert kps[li][2] >= 0
                            kps[li][2] = 1.0  # weight
                            # if li==0:
                            #  landmark_num+=1
                            # if kps[li][2]==0.0:#visible
                            #  kps[li][2] = 1.0
                            # else:
                            #  kps[li][2] = 0.0
                else:  # len(values)==5
                    if not ignore:
                        ignore = (values[4] == 1)
        else:
            kps = None
        return dict(bbox=bbox, kps=kps, ignore=ignore, cat='FG')


    def load_annotations(self, ann_file):
        name = None
        bbox_map = {}
        # read annotation file, eg. label.txt
        with open(ann_file, 'r') as fr:
            all_lines = fr.readlines()
            fr.close()
        # parse each line which is in format of widerface annotation
        for line in all_lines:
            line = line.strip()
            if line.startswith('#'):
                value = line[1:].strip().split()
                name = value[0]
                width = int(value[1])
                height = int(value[2])
                bbox_map[name] = dict(width=width, height=height, objs=[])
                continue
            assert name is not None
            assert name in bbox_map
            bbox_map[name]['objs'].append(line)

        print('origin image size', len(bbox_map))

        data_infos = []
        for name in bbox_map:
            item = bbox_map[name]
            width = item['width']
            height = item['height']
            vals = item['objs']
            objs = []
            for line in vals:
                data = self._parse_ann_line(line)
                if data is None:
                    continue
                objs.append(data) #data is (bbox, kps, cat)
            # if len(objs) == 0 and not self.test_mode: # test_mode: [image dir, height, width, no bboxes and keypoints]
            #     continue
            data_infos.append(dict(filename=name, width=width, height=height, objs=objs))
        return data_infos


    def get_ann_info(self, idx):
        data_info = self.data_infos[idx]
        bboxes = []
        keypoints = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in data_info['objs']:
            label = self.cat2label[obj['cat']]
            bbox = obj['bbox']
            kps = obj['kps']
            ignore = obj['ignore']
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
                keypoints.append(kps)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
            keypoints = np.zeros((0, self.NK, 3)) if self.with_kp else None
        else:
            # bboxes = np.array(bboxes, ndmin=2) - 1
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
            keypoints = np.array(keypoints, ndmin=3) if self.with_kp else None

        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            # bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            keypoints=keypoints.astype(np.float32) if keypoints is not None else None,
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann


    def __len__(self):
        return len(self.data_infos)


    def __getitem__(self, idx):
        img_info = self.data_infos[idx]

        img_name = img_info['filename']

        img_dir = os.path.join(self.img_path, img_name)

        img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
        assert img is not None, img_dir
        h, w = img.shape[:2]
        annos = self.get_ann_info(idx)

        # gt_bbox = np.concatenate([annos['bboxes'], np.expand_dims(annos['labels'], axis=-1)], axis=1)
        # ic(annos)
        data = {
            'image': img, 'gt_bbox': annos['bboxes']
        }
        data = self.aug_pipeline(data)
        gt_bbox = np.concatenate([data['gt_bbox'], np.expand_dims(annos['labels'], axis=-1)], axis=1)
        return {'image': data['image'],
                'gt_bbox': torch.from_numpy(gt_bbox),
                'img_info': img_name,
                'org_size': np.array((w, h)),
                'scale_factor': data['scale_factor']}
        # exit(11)

        # img = data['image'].astype(np.uint8)
        #
        # bbox = data['gt_bbox'].astype(np.int32)
        # # keypoint = annos['keypoints'].astype(np.int32)
        # for i in bbox:
        #     cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), cv2.LINE_4)
        #
        # # for i in keypoint[0]:
        # #     cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255))
        # plt.imshow(img)
        # plt.show()
        # ic(annos)

    @staticmethod
    def collate(batch_samples):
        batch_dt = {key: list()  for key in batch_samples[0].keys()}
        for idx in range(len(batch_samples)):
            for k, v in batch_samples[idx].items():
                batch_dt[k].append(v)
        if 'image' in batch_dt:
            batch_dt['image'] = torch.stack(batch_dt['image'], dim=0)
        if 'org_size' in batch_dt:
            batch_dt['org_size'] = np.stack(batch_dt['org_size'], axis=0)
        if 'scale_factor' in batch_dt:
            batch_dt['scale_factor'] = np.stack(batch_dt['scale_factor'], axis=0)
        return batch_dt
        # return {'images': torch.stack(img_ls, dim=0), 'gt_bboxes': bbox_ls, 'img_infos': img_info}




if __name__ == '__main__':
    data_p = '/home/tjm/Documents/python/pycharmProjects/blazeface/data/widerface'
    kyw = {
            'Resize': {'target_size': (640, 640), 'keep_ratio': True},
            'RandomFlip': {'prob': 0.5},
        }
    data = WiderFaceDataset(data_p, mode='train', min_size=20, with_kp=False, pipeline=kyw)
    x = data[12]
    # print(x)


