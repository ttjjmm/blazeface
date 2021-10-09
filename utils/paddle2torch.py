import os
import shutil
import tempfile
import paddle.fluid as fluid

import torch
from collections import OrderedDict


def _load_state(path):
    """
    记载paddlepaddle的参数
    :param path:
    :return:
    """
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)
    return state



if __name__ == '__main__':
    x =  _load_state('/home/ubuntu/Documents/pycharm/blazeface/weights/blazeface_1000e.pdparams')
    new_dict = OrderedDict()
    for k, v in x.items():
        if not isinstance(v, dict):
            print(k, v.shape)
            new_dict[k] = torch.FloatTensor(v)
    torch.save(new_dict, '/home/ubuntu/Documents/pycharm/blazeface/weights/blazeface_1000e.pth')
