import torch
from model import build_model

def convert():
    m = build_model('/config/blazeface_fpn_ssh.yaml')
    ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/blazeface/weights/blazeface_fpn_ssh_1000e.pth')
    from collections import OrderedDict
    new_stat = OrderedDict()
    for k, v in m.state_dict().items():
        key_points = k.split('.')
        if key_points[-1] == 'num_batches_tracked':
            continue
        if key_points[-1] == 'running_mean':
            tail = '_mean'
        elif key_points[-1] == 'running_var':
            tail = '_variance'
        else:
            tail = key_points[-1]

        if key_points[1] == 'blaze_':
            if key_points[3] == 'conv_dw':
                new_key = '{}.{}.{}.{}.{}'.format(key_points[0],
                                                  key_points[1] + key_points[2],
                                                  'blaze_{}1_dw'.format(int(key_points[2])),
                                                  key_points[5],
                                                  tail)
            elif key_points[3] == '_shortcut':
                new_key = '{}.{}.{}.{}.{}'.format(key_points[0],
                                                  key_points[1] + key_points[2],
                                                  'blaze_{}_shortcut_conv'.format(key_points[2]), key_points[5], tail)
            else:
                new_key = '{}.{}.{}.{}.{}'.format(key_points[0],
                                                  key_points[1] + key_points[2],
                                                  key_points[3], key_points[4], tail)

        elif key_points[1].split('_')[0] == 'double':
            if key_points[3] == '_shortcut':
                new_key = '{}.{}.{}.{}.{}'.format(key_points[0],
                                                  key_points[1] + key_points[2],
                                                  'double_blaze_{}_shortcut_conv'.format(key_points[2]),
                                                  key_points[5],
                                                  tail)
            elif key_points[3].split('_')[1][0:2] == 'dw':
                if key_points[3].split('_')[1] == 'dw2':
                    new_key = '{}.{}.{}.{}.{}'.format(key_points[0],
                                                      key_points[1] + key_points[2],
                                                      'double_blaze_{}2_dw'.format(key_points[2]), key_points[5], tail)
                else:
                    new_key = '{}.{}.{}.{}.{}'.format(key_points[0],
                                                      key_points[1] + key_points[2],
                                                      'double_blaze_{}1_dw'.format(key_points[2]), key_points[5], tail)
            else:
                new_key = '{}.{}.{}.{}.{}'.format(key_points[0], key_points[1] + key_points[2], key_points[3],
                                                  key_points[4], tail)
        elif key_points[0] == 'blaze_head':
            new_key = '{}.{}.{}'.format(key_points[0], key_points[1] + key_points[2], tail)
        else:
            new_key = ''
            for i in key_points[:-1]:
                new_key += i + '.'
            new_key += tail
        if new_key not in ckpt.keys():
            print(new_key)
            raise RuntimeError
        if v.shape != ckpt[new_key].shape:
            print(new_key)
            raise RuntimeError
        # print(k, v.shape)
        new_stat[k] = ckpt[new_key]

    m.load_state_dict(new_stat)
    torch.save(m.state_dict(), './blazeface_fpn_ssh_1000e.pt')