import matplotlib.pyplot as plt
from model.blazeface import BlazeFace
import yaml
import torch

def build_model(config, img_size):
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
    cfg_anchor = config[anchor]
    cfg_head['cfg_loss'] = config[loss]

    model = BlazeFace(cfg_backbone,
                      cfg_neck,
                      cfg_head,
                      cfg_postprocess,
                      cfg_anchor,
                      img_size)
    return model
