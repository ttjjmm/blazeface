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
        loss = arch_cfg['loss']
        anchor = arch_cfg['anchor']
    else:
        raise AttributeError("object has no attribute '{}'".format(arch))

    cfg_backbone = config[backbone]
    cfg_neck = config[neck] if neck else None
    cfg_head = config[head]
    cfg_postprocess = config[postprocess]
    cfg_head['cfg_loss'] = config[loss]
    cfg_head['cfg_anchor'] = config[anchor]

    model = BlazeFace(cfg_backbone, cfg_neck, cfg_head, cfg_postprocess)
    return model
