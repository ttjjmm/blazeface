import math
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import build_dataloader
from model import build_model
from model.post_process import SSDBox
from model.loss import AnchorGeneratorSSD
from utils import load_config
from utils.evaluate import evaluation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/blazeface_fpn_ssh.yaml',
                        help='model configuration file path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--score_thr', type=float, default=0.2,
                        help='iou threashold')
    parser.add_argument('--nms_thr', type=float, default=0.45,
                        help='iou threashold')
    parser.add_argument('--size', default=(640, 640), type=tuple,
                        help='detect input image size')
    # parser.add_argument('--seed', type=int, default=None,
    #                     help='random seed')
    args = parser.parse_args()
    return args


class Evaluator(object):
    def __init__(self, model, val_loader, priors, img_size=None, score_thr=0.2, nms_thr=0.4, device='cuda:0'):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.post_process = SSDBox(iou_thr=score_thr, nms_thr=nms_thr)
        self.anchors = priors.to(device)
        self.device = device
        self.img_size = img_size

    def eval(self):
        self.model.eval()
        all_result_dict = dict()
        for data in tqdm(self.val_loader, desc='Evalution Stage'):
            imgs = data['image'].to(self.device)
            imgs_info = data['img_info']
            batch_szie = imgs.size(0)
            scale_factor = data['scale_factor']
            pad_size = self.padsize(self.img_size, data['org_size'], scale_factor)

            with torch.no_grad():
                box_pred, score_pred = self.model(imgs)

            for idx in range(batch_szie):
                det_bboxes = self.post_process((box_pred[idx].unsqueeze(0),
                                                score_pred[idx].unsqueeze(0)),
                                               self.anchors).cpu().numpy()
                det_bboxes[:, [0, 1]] = (det_bboxes[:, [0, 1]] - pad_size[idx]) / scale_factor[idx]
                det_bboxes[:, [2, 3]] = det_bboxes[:, [2, 3]] - pad_size[idx] / scale_factor[idx]

                key, filename = imgs_info[idx].split('/')

                if key not in all_result_dict:
                    all_result_dict[key] = dict()
                det_bboxes = np.array(det_bboxes).astype(np.float64)
                det_bboxes[:, 2:4] = det_bboxes[:, 2:4] - det_bboxes[:, :2]
                all_result_dict[key][filename] = det_bboxes

        aps = evaluation.eval_map(all_result_dict, all=False)
        print('Easy:{:.4f}, Medium:{:.4f}, Hard:{:.4f}'.format(*aps))

    @staticmethod
    def padsize(target_size, org_size, scale_factor):
        if isinstance(target_size, (list, tuple)):
            target_size = np.array(target_size)
        elif isinstance(target_size, (int, float)):
            target_size = np.array((target_size, target_size))
        else:
            raise RuntimeError('target size value error!')
        pad_size = (target_size[np.newaxis, :] - org_size * scale_factor).astype(int) // 2
        # pad_size = (target_size[np.newaxis, :] / scale_factor)
        return pad_size


def evaluate():
    args = parse_args()
    cfgs = load_config(args.cfg)
    data_cfg = cfgs['data'].copy()
    anchor_cfg = cfgs['model']['AnchorGeneratorSSD']

    img_size = data_cfg['val']['dataset']['img_size']
    device = args.device
    anchor_gen = AnchorGeneratorSSD(**anchor_cfg)
    priors = anchor_gen(image_size=img_size)
    model = build_model(cfgs['model'].copy(), img_size)
    val_loader = build_dataloader(data_cfg['val'], mode='val')

    # define evaluator
    evaluator = Evaluator(model, val_loader, priors, img_size, args.score_thr, args.nms_thr, device)
    evaluator.eval()


if __name__ == '__main__':
    evaluate()










