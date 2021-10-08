import matplotlib.pyplot as plt

from model.blazeface import BlazeFace
import yaml
import torch

def load_config(cfg_path):
    config = yaml.load(open(cfg_path, 'rb'), Loader=yaml.Loader)
    return config


def build_model(cfg_path):
    config = load_config(cfg_path)
    arch = config.pop('architecture')
    if arch in config:
        backbone = config[arch]['backbone']
        neck = config[arch]['neck']
        head = config[arch]['head']
        postprocess = config[arch]['post_process']
    else:
        raise AttributeError("object has no attribute '{}'".format(arch))

    cfg_backbone = config[backbone]
    cfg_neck = config[neck]
    cfg_head = config[head]
    cfg_postprocess = config[postprocess]
    cfg_head['cfg_anchor'] = {'steps': [8, 16],
                              'aspect_ratios': [[1.], [1.]],
                              'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]],
                              'offset': 0.5,
                              'flip':False}

    model = BlazeFace(cfg_backbone, cfg_neck, cfg_head, cfg_postprocess)
    return model


if __name__ == '__main__':
    import cv2
    import numpy as np
    from data.operators import Resize, Pad
    from utils.tools import flops_info
    m = build_model('../config/blazeface.yaml').eval()
    # flops_info(m)
    # exit(11)

    # inp = torch.randn((1, 3, 640, 640))

    # for i in m(inp):
    #     print(i.shape)
    #
    # exit(11)

    img = cv2.imread('../samples/test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = {'image': img, 'im_shape': [640, 640]}
    resize = Resize([640, 640])
    pad = Pad([640, 640])
    data = resize(data)
    data = pad(data)

    img = data['image']

    raw_img = img.astype(np.uint8)



    img = (img - [123, 117, 104]) / [127.502231, 127.502231, 127.502231]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(np.expand_dims(img, axis=0)).to(torch.float32)

    data['image'] = img

    with torch.no_grad():
        x = m.inference(img)

    for b in x:
        # print(b[4])
        text = "{:.3f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(raw_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(raw_img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))


    plt.imshow(raw_img)
    plt.show()