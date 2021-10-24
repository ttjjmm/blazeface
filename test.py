import math
import argparse
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import build_dataloader
from model import build_model
from model.post_process import SSDBox
from model.loss import AnchorGeneratorSSD
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/blazeface_fpn_ssh.yaml',
                        help='model configuration file path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--score_thr', type=float, default=0.01,
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
        for data in tqdm(self.val_loader, desc='Evalution Stage'):
            dets = list()
            imgs = data['image'].to(self.device)
            imgs_info = data['img_info']
            batch_szie = imgs.size(0)

            with torch.no_grad():
                box_pred, score_pred = self.model(imgs)

            for idx in range(batch_szie):
                det_bboxes = self.post_process((box_pred[0].unsqueeze(0), score_pred[0].unsqueeze(0)), self.anchors)
                print(det_bboxes.shape)



                key, filename = imgs_info[idx].split('/')
                print(key, filename)

            break


    def scale2orgsize(self, org_size, target_size):
        pass



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










