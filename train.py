import os
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm

from model import build_model
from data import build_dataloader
from utils.tools import create_workspace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/blazeface.yaml',
                        help='model configuration file path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--resume', nargs='?', const=True, default=False,
                        help='resume most recent training')
    parser.add_argument('--seed', type=int, default=718,
                        help='random seed')
    args = parser.parse_args()
    return args


def init_seeds(seed=0):
    """
    manually set a random seed for numpy, torch and cuda
    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(cfg_path):
    with open(cfg_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)

    # config = yaml.load(open(cfg_path, 'rb'), Loader=yaml.Loader)
    return file_cfg


# def


class Trainer(object):
    def __init__(self, args):

        cfgs = load_config(args.cfg)
        model_cfg = cfgs['model'].copy()
        data_cfg = cfgs['data'].copy()
        self.resume = args.resume
        self.device = args.device
        self.model = build_model(model_cfg).to(self.device)

        self.train_loader = build_dataloader(data_cfg['train'], mode='train')
        # self.val_loader = build_dataloader(data_cfg['val'], mode='val')

        # workspace steup


        # matric

        # logger
    def initial_setup(self):
        pass


    def train(self):
        for idx, batch in enumerate(tqdm(self.train_loader)):
            print(batch[0].shape)
            # pass

    def train_epoch(self):
        pass


    def save_weights(self):
        pass


    def load_weights(self):
        pass



def main():
    args = parse_args()

    # t = Trainer(arg)
    # t.train()
    pass



if __name__ == '__main__':
    main()








