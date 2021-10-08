import torch
import torchvision
import os
import struct
from model import build_model






# from data import cfg_mnet, cfg_re50, cfg_shuffle, cfg_mbdet
# from models.retinaface import RetinaFace
#
#
#
#
#
# def check_keys(model, pretrained_state_dict):
#     ckpt_keys = set(pretrained_state_dict.keys())
#     model_keys = set(model.state_dict().keys())
#     used_pretrained_keys = model_keys & ckpt_keys
#     unused_pretrained_keys = ckpt_keys - model_keys
#     missing_keys = model_keys - ckpt_keys
#     print('Missing keys:{}'.format(len(missing_keys)))
#     print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
#     print('Used keys:{}'.format(len(used_pretrained_keys)))
#     assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
#     return True
#
#
# def remove_prefix(state_dict, prefix):
#     """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
#     print('remove prefix \'{}\''.format(prefix))
#     f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
#     return {f(key): value for key, value in state_dict.items()}
#
#
# def load_model(model, pretrained_path, load_to_cpu):
#     print('Loading pretrained model from {}'.format(pretrained_path))
#     if load_to_cpu:
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
#     else:
#         device = torch.cuda.current_device()
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
#     if "state_dict" in pretrained_dict.keys():
#         pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
#     else:
#         pretrained_dict = remove_prefix(pretrained_dict, 'module.')
#     check_keys(model, pretrained_dict)
#     model.load_state_dict(pretrained_dict, strict=False)
#     del pretrained_dict
#     return model


def main():
    print('cuda device count: ', torch.cuda.device_count())
    device = 'cuda:0'


    net = m = build_model('../config/blazeface_fpn_ssh.yaml')

    checkpoint = torch.load('../weights/blazeface_fpn_ssh_1000e.pt', map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint)

    # net = torch.load('/home/ubuntu/Documents/pycharm/Pytorch_Retinaface/weights/mobilenet0.25_Final.pth')
    net = net.to(device)
    net.eval()
    print('model: ', net)
    #print('state dict: ', net.state_dict().keys())
    tmp = torch.ones(1, 3, 640, 640).to(device)
    print('input: ', tmp)
    out = net(tmp)
    print('output:', out)

    if os.path.exists('blazeface_fpn_ssh.wts'):
        return
    f = open("retinaface.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()