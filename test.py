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
    def __init__(self, model, val_loader, priors, score_thr=0.01, nms_thr=0.4, device='cuda:0'):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.post_process = SSDBox(iou_thr=score_thr, nms_thr=nms_thr)
        self.anchors = priors
        self.device = device

    def eval(self):
        self.model.eval()
        for idx, data in enumerate(self.val_loader):

            imgs = data['images'].to(self.device)
            with torch.no_grad():
                preds = self.model(imgs)
            break



    def __call__(self, *args, **kwargs):
        pass




def evaluate():
    args = parse_args()
    cfgs = load_config(args.cfg)
    data_cfg = cfgs['data'].copy()
    img_size = data_cfg['val']['img_size']
    device = args.device
    prior = AnchorGeneratorSSD()
    model = build_model(cfgs['model'].copy(), img_size)
    val_loader = build_dataloader(data_cfg['val'], mode='val')

    # define evaluator
    evaluator = Evaluator(model, val_loader, args.score_thr, args.nms_thr, device)
    evaluator.eval()


if __name__ == '__main__':
    evaluate()










