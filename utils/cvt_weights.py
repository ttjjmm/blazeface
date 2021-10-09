import torch
from model import build_model
from collections import OrderedDict


def convert():
    m = build_model('../config/blazeface.yaml')
    print(m)
    ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/blazeface/weights/blazeface_1000e.pt')

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

    m.load_state_dict(new_stat, strict=True)
    torch.save(m.state_dict(), './blazeface_1000e.pt')


def convert2():

    cache = {
        'blaze_head.boxes.1.weight': 'blaze_head.boxes1.weight',
        'blaze_head.boxes.1.bias': 'blaze_head.boxes1.bias',
        'blaze_head.scores.0.weight': 'blaze_head.scores0.weight',
        'blaze_head.scores.0.bias': 'blaze_head.scores0.bias'
    }



    new_dict = OrderedDict()
    m = build_model('../config/blazeface.yaml')
    ckpt = torch.load('/home/ubuntu/Documents/pycharm/blazeface/weights/blazeface_1000e.pth')
    key_list = list()
    for k, v in m.state_dict().items():
        if k.split('.')[-1] != 'num_batches_tracked':
            key_list.append(k)
        # print(k, v.shape)
    print(len(key_list))
    print(len(ckpt))
    for idx, (k, v) in enumerate(ckpt.items()):
        print(k, v.shape)
        key = key_list[idx]
        if key in cache:
            print('----------->', cache[key])
            new_dict[key] = ckpt[cache[key]]
        else:
            new_dict[key] = v
        # assert v.shape == m.state_dict()[key].shape, (key, k)
    m.load_state_dict(new_dict, strict=True)
    torch.save(m.state_dict(), '/home/ubuntu/Documents/pycharm/blazeface/weights/blazeface_1000e.pt')


if __name__ == '__main__':
    convert2()