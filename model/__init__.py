import matplotlib.pyplot as plt
from model.blazeface import BlazeFace
import yaml
import torch


# def load_config(cfg_path):
#     config = yaml.load(open(cfg_path, 'rb'), Loader=yaml.Loader)
#     return config


def build_model(config):
    # config = load_config(cfg_path)
    arch = config.pop('architecture')
    if arch in config:
        arch_cfg = config[arch]
        backbone = arch_cfg['backbone']
        neck = arch_cfg['neck'] if 'neck' in arch_cfg else None
        head = arch_cfg['head']
        postprocess = arch_cfg['post_process']
    else:
        raise AttributeError("object has no attribute '{}'".format(arch))

    cfg_backbone = config[backbone]
    cfg_neck = config[neck] if neck else None
    cfg_head = config[head]
    cfg_postprocess = config[postprocess]
    cfg_head['cfg_anchor'] = {'steps': [8, 16],
                              'aspect_ratios': [[1.], [1.]],
                              'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]],
                              'offset': 0.5,
                              'flip':False}

    model = BlazeFace(cfg_backbone, cfg_neck, cfg_head, cfg_postprocess)
    return model
