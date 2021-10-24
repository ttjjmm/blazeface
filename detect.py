import os
import torch
import argparse
import cv2
import numpy as np

from model import build_model
from model.loss import AnchorGeneratorSSD
from data.operators import Resize
from utils import load_config

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/blazeface_fpn_ssh.yaml',
                        help='model configuration file path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img_path', type=str, default='./samples/12_Group_Group_12_Group_Group_12_728.jpg',
                        help='detect sample image path')
    parser.add_argument('--size', default=(640, 640), type=tuple,
                        help='detect input image size')
    parser.add_argument('--org_size', nargs='?', const=True, default=True,
                        help='resume most recent training')
    # parser.add_argument('--seed', type=int, default=None,
    #                     help='random seed')
    args = parser.parse_args()
    return args


class FaceDetector(object):
    def __init__(self, args):
        cfg = load_config(args.cfg)
        self.device = args.device
        cfg_model = cfg['model'].copy()
        self.model = build_model(cfg_model, args.size).to(self.device)
        if args.org_size or args.size is None:
            self.anchor = AnchorGeneratorSSD(**cfg_model['AnchorGeneratorSSD'])
            self.size = None
        else:
            self.anchor = None
            self.size = args.size

        self.img_path = args.img_path

    def preprocess(self):
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        priors = None
        if self.size is None:
            self.size = [img.shape[1], img.shape[0]]
            priors = self.anchor(self.size[::-1]).to(self.device)
        data = {'image': img, 'im_shape': self.size}
        if self.size is not None:
            resize = Resize(self.size)
            data = resize(data)
        img = data['image']
        raw_img = img.astype(np.uint8)
        data['raw_img'] = raw_img
        img = (img - [123, 117, 104]) / [127.502231, 127.502231, 127.502231]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(np.expand_dims(img, axis=0)).to(torch.float32)

        data['image'] = img
        return data, priors

    @staticmethod
    def visualize(dets, raw_img):
        for b in dets:
            # print(b[4])
            text = "{:.3f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(raw_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(raw_img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
        plt.imshow(raw_img)
        plt.show()

    def detect(self):
        image_info, priors = self.preprocess()
        img_size = self.size if priors is not None else None
        image = image_info['image'].to(self.device)
        self.model.eval()
        with torch.no_grad():
            dets = self.model.inference(image, priors, img_size)
        self.visualize(dets, image_info['raw_img'])
        return 0



if __name__ == '__main__':
    args_ = parse_args()
    print(args_)
    FD = FaceDetector(args_)
    data_info = FD.detect()

    # plt.imshow(data_info['raw_img'])
    # plt.show()
