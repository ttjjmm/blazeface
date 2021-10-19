"""
The implementation of some base CNN pruning methods
Author: Jimmy Tao
Date: 2021-10-12
"""


import numpy as np
import torch
import torch.nn as nn
import os

from abc import ABCMeta, abstractmethod
from model.blazenet import BlazeNet



class BasePruner(metaclass=ABCMeta):
    """
    :param model: nn.Module

    """
    def __init__(self, model, prune_rate):
        self.model = model
        self.prune_rate = prune_rate
        self.param_dt = dict()
        self.mat = dict()



    def get_codebook(self):
        pass


    @abstractmethod
    def get_mask(self):
        pass



class SoftFilterPruner(BasePruner):
    def __init__(self, model: nn.Module, prune_rate=0.9):
        super(SoftFilterPruner, self).__init__(model, prune_rate)
        self.param_ls = dict()


    def get_mask(self):
        pass


    def get_param_dt(self):
        for k, v in self.model.named_parameters():
            print(k, v.shape)


    # def get_zero_idx_ls(self, filter_weight, compress_rate, inch=True):
    #     if len(filter_weight.size()) == 4:
    #         if inch:
    #             in_chs = filter_weight.size()[1]
    #             idx_ls = [0] * in_chs
    #             filter_prune_num = int(in_chs * (1 - compress_rate))
    #             filter_idx = self.L2Norm_distance()
    #     else:
    #         pass


    @staticmethod
    def L2Norm_distance(filter_weight, compress_rate, inch=True):
        idx_ls = list()
        if len(filter_weight.size()) == 4:
            chs = filter_weight.size()[0] if not inch else filter_weight.size()[1]
            idx_ls = [1] * chs

            filter_prune_num = int(chs * (1 - compress_rate))
            weight_vector = filter_weight.view(chs, -1) if not inch else filter_weight.permute(1, 0, 2, 3).view(chs, -1) # shape:[outch, n]
            norm = torch.norm(weight_vector, p=2, dim=1)  # L2-Norm
            # norm = norm.cpu().numpy()
            filter_idx = torch.argsort(norm)[:filter_prune_num]
            for idx in filter_idx:
                idx_ls[idx] = 0
        else:
            pass
        return idx_ls


    def __call__(self, *args, **kwargs):
        self.get_param_dt()



    def update(self):
        pass


    def is_zero(self):
        pass



if __name__ == '__main__':
    # m = BlazeNet(act='hard_swish', lite_edition=False)
    # pruner = SoftFilterPruner(m)
    # inp = torch.randn((24, 16, 3, 3))
    # x = pruner.get_L2Norm_distance(inp, 0.8)
    # print(x)

    x1 = (torch.randn((4, 4, 3, 3)) * torch.tensor([1, 0, 1, 0])[np.newaxis, :, np.newaxis, np.newaxis]).permute(0,2,1,3)
    print(x1.shape)
    # print(torch.tensor([1, 0, 1, 0])[np.newaxis, :, np.newaxis, np.newaxis].shape)








