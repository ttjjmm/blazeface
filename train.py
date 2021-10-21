import os
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm

from model import build_model
from data import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/blazeface.yaml',
                        help='model configuration file path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img_path', type=str, default='./samples/test.jpg',
                        help='detect sample image path')
    parser.add_argument('--size', default=(640, 640), type=tuple,
                        help='detect input image size')
    # parser.add_argument('--seed', type=int, default=None,
    #                     help='random seed')
    args = parser.parse_args()
    return args



def load_config(cfg_path):
    with open(cfg_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)

    # config = yaml.load(open(cfg_path, 'rb'), Loader=yaml.Loader)
    return file_cfg



class Trainer(object):
    def __init__(self, args):
        self.device = args.device
        cfgs = load_config(args.cfg)
        model_cfg = cfgs['model'].copy()
        data_cfg = cfgs['data'].copy()
        self.model = build_model(model_cfg).to(self.device)
        print(data_cfg['train'])
        # exit(11)
        self.train_loader = build_dataloader(data_cfg['train'], mode='train')
        # self.val_loader = build_dataloader(data_cfg['val'], mode='val')




    def train_epoch(self):
        pass


    def save_weights(self):
        pass


    def load_weights(self):
        pass





if __name__ == '__main__':
    arg = parse_args()

    t = Trainer(arg)









