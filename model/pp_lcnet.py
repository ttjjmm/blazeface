import torch
import torch.nn as nn

# MODEL_URLS = {
#     "PPLCNet_x0_25":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
#     "PPLCNet_x0_35":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams",
#     "PPLCNet_x0_5":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams",
#     "PPLCNet_x0_75":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
#     "PPLCNet_x1_0":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams",
#     "PPLCNet_x1_5":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams",
#     "PPLCNet_x2_0":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams",
#     "PPLCNet_x2_5":
#     "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams"
# }

# __all__ = list(MODEL_URLS.keys())

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
                #k, in_c, out_c, s, use_se
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)

        self.bn = nn.BatchNorm2d(num_filters)
        self.hardswish = nn.Hardswish()
        self._init_weights()

    def _init_weights(self):
        # original implementation is unknown
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)
                # nn.init.constant_(module.bias.data, 0)
            # elif isinstance(module, nn.BatchNorm2d):
            #     nn.init.constant_(module.weight.data, 1)
            #     nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0))
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return x * identity


class PPLCNet(nn.Module):
    def __init__(self, scale=1.0):
        super(PPLCNet, self).__init__()
        self.scale = scale
        # self.class_expand = class_expand
        self.out_idx = [3, 4, 5, 6]
        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])
        out_channels = [make_divisible(NET_CONFIG[k][-1][2] * scale) for k in NET_CONFIG] #self.stage_out_channels[:-1]
        self.out_channels = [out_channels[idx - 2] for idx in self.out_idx]
        self.load_pretrained_weights()

    def load_pretrained_weights(self):
        ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/centerdet/samples/PPLCNet_x1_0_pretrained.pt')
        self.load_state_dict(ckpt, strict=True)
        del ckpt

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        for i in range(2, 7):
            stage = getattr(self, 'blocks{}'.format(i))
            x = stage(x)
            if i in self.out_idx:
               outs.append(x)
        return outs


if __name__ == '__main__':
    inp = torch.randn((4, 3, 320, 320))
    m = PPLCNet(scale=1.0)
    # ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/centerdet/samples/PPLCNet_x1_0_pretrained.pt')
    # m.load_state_dict(ckpt, strict=True)
    # for k, v in m.state_dict().items():
    #     print(k, v.shape)

    # from tools.utils import flops_info
    #
    # flops_info(m)
    for i in  m(inp):
        print(i.shape)






