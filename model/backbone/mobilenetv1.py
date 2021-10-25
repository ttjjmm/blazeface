import torch
import torch.nn as nn


def conv_bn(inp, oup, stride=1, leaky=0.0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, (3, 3), (stride, stride), (1, 1), bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, (3, 3), stride, (1, 1), groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, (1, 1), (1, 1), (0, 0), bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )


class MobileNetV1(nn.Module):
    """
    The Impelement of MobileNetv1 x0.25
    """
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        # self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        out = {}
        for i in range(1, 4):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            out[i - 1] = x

        return out


if __name__ == '__main__':
    m = MobileNetV1()
    in_tensor = torch.randn((4, 3, 640, 640))
    for k, v in m(in_tensor).items():
        print(v.shape)