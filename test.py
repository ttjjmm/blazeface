import math
import argparse
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import build_dataloader
from model import build_model
from model.post_process import SSDBox
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
    def __init__(self, model, val_loader, iou_thr, nms_thr, device):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.post_process = SSDBox(iou_thr=iou_thr, nms_thr=nms_thr)
        self.device = device

    def eval(self):
        self.model.eval()
        for idx, data in enumerate(self.val_loader):

            imgs = data['images'].to(self.device)
            with torch.no_grad():
                preds = self.model(imgs)
                print(preds)
            break

    def __call__(self, *args, **kwargs):
        pass




def evaluate():
    args = parse_args()
    cfgs = load_config(args.cfg)
    data_cfg = cfgs['data'].copy()
    device = args.device
    model = build_model(cfgs['model'].copy())
    val_loader = build_dataloader(data_cfg['val'], mode='val')

    # define evaluator
    evaluator = Evaluator(model, val_loader, device)
    evaluator.eval()


if __name__ == '__main__':
    evaluate()










