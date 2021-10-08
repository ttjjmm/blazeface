import torch
import torch.nn as nn
import torch.nn.functional as F

# def hard_swish(x):
#     return x * F.relu6(x + 3) / 6.


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 num_groups=1,
                 act='relu'):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias=False)

        self._batch_norm = nn.BatchNorm2d(out_channels)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        elif act == 'leaky':
            self.act = nn.LeakyReLU(inplace=True)
        elif act == 'hard_swish':
            self.act = nn.Hardswish(inplace=True)
        elif act is None:
            self.act = nn.Identity()
        else:
            raise NotImplementedError


        self._init_weights()

    def _init_weights(self):
        # original implementation is unknown
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)
                # nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self.act(x)
        return x


class BlazeBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels1,
                 out_channels2,
                 double_channels=None,
                 stride=1,
                 use_5x5kernel=True,
                 act='relu'):
        super(BlazeBlock, self).__init__()
        assert stride in [1, 2]
        self.use_pool = not stride == 1
        self.use_double_block = double_channels is not None
        self.conv_dw = nn.ModuleList()
        if use_5x5kernel:
            self.conv_dw.append(
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=out_channels1,
                    kernel_size=5,
                    stride=stride,
                    padding=2,
                    num_groups=out_channels1))
        else:
            self.conv_dw.append(
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=out_channels1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    num_groups=out_channels1))
            self.conv_dw.append(
                ConvBNLayer(
                    in_channels=out_channels1,
                    out_channels=out_channels1,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    num_groups=out_channels1))
        act = act if self.use_double_block else None
        self.conv_pw = ConvBNLayer(
            in_channels=out_channels1,
            out_channels=out_channels2,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act)
        if self.use_double_block:
            self.conv_dw2 = nn.ModuleList()
            if use_5x5kernel:
                self.conv_dw2.append(
                    ConvBNLayer(
                        in_channels=out_channels2,
                        out_channels=out_channels2,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        num_groups=out_channels2))
            else:
                self.conv_dw2.append(
                    ConvBNLayer(
                        in_channels=out_channels2,
                        out_channels=out_channels2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        num_groups=out_channels2))
                self.conv_dw2.append(
                    ConvBNLayer(
                        in_channels=out_channels2,
                        out_channels=out_channels2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        num_groups=out_channels2))
            self.conv_pw2 = ConvBNLayer(
                in_channels=out_channels2,
                out_channels=double_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        # shortcut
        if self.use_pool:
            shortcut_channel = double_channels or out_channels2
            self._shortcut = nn.ModuleList()
            self._shortcut.append(nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=True))
            self._shortcut.append(
               ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=shortcut_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0))

    def forward(self, x):
        y = x
        for conv_dw_block in self.conv_dw:
            y = conv_dw_block(y)
        y = self.conv_pw(y)
        if self.use_double_block:
            for conv_dw2_block in self.conv_dw2:
                y = conv_dw2_block(y)
            y = self.conv_pw2(y)
        if self.use_pool:
            for shortcut in self._shortcut:
                x = shortcut(x)
        return F.relu(x + y)



class BlazeBlockLite(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BlazeBlockLite, self).__init__()
        assert stride in [1, 2], "Please confirm your stride parameter value!"
        self.use_pool = not stride == 1
        self.use_pad = not in_channels == out_channels

        self.conv_dw = ConvBNLayer(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1,
                                   num_groups=in_channels)

        self.conv_pw = ConvBNLayer(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   num_groups=1)
        if self.use_pool:
            self.shortcut_pool = nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        if self.use_pad:



    def forward(self, x):
        pass




class BlazeNet(nn.Module):
    """
    BlazeFace, see https://arxiv.org/abs/1907.05047

    Args:
        blaze_filters (list): number of filter for each blaze block.
        double_blaze_filters (list): number of filter for each double_blaze block.
        use_5x5kernel (bool): whether or not filter size is 5x5 in depth-wise conv.
    """

    def __init__(self,
                 blaze_filters=None,
                 double_blaze_filters=None,
                 use_5x5kernel=True,
                 lite_edition=False,
                 act=None):
        super(BlazeNet, self).__init__()

        if blaze_filters is None:
            blaze_filters = [[24, 24], [24, 24], [24, 48, 2], [48, 48], [48, 48]]
        if double_blaze_filters is None:
            double_blaze_filters = [[48, 24, 96, 2], [96, 24, 96], [96, 24, 96],
                                    [96, 24, 96, 2], [96, 24, 96], [96, 24, 96]]

        if not lite_edition:
            conv1_num_filters = blaze_filters[0][0]
            self.conv1 = ConvBNLayer(
                in_channels=3,
                out_channels=conv1_num_filters,
                kernel_size=3,
                stride=2,
                padding=1)

            in_channels = conv1_num_filters
            self.blaze_ = nn.ModuleList()
            self.double_blaze_ = nn.ModuleList()
            # _out_shape = []
            for k, v in enumerate(blaze_filters):
                assert len(v) in [2, 3], "blaze_filters {} not in [2, 3]"
                if len(v) == 2:
                    self.blaze_.append(
                        BlazeBlock(in_channels, v[0], v[1], use_5x5kernel=use_5x5kernel, act=act))
                elif len(v) == 3:
                    self.blaze_.append(BlazeBlock( in_channels, v[0], v[1], stride=v[2], use_5x5kernel=use_5x5kernel, act=act))
                in_channels = v[1]

            for k, v in enumerate(double_blaze_filters):
                assert len(v) in [3, 4], "blaze_filters {} not in [3, 4]"
                if len(v) == 3:
                    self.double_blaze_.append(
                        BlazeBlock(in_channels, v[0], v[1], double_channels=v[2], use_5x5kernel=use_5x5kernel, act=act))
                elif len(v) == 4:
                    self.double_blaze_.append(
                       BlazeBlock(in_channels, v[0], v[1], double_channels=v[2],  stride=v[3], use_5x5kernel=use_5x5kernel, act=act))
                in_channels = v[2]
            # _out_shape.append(in_channels)
        else:
            pass


    def forward(self, inputs):
        outs = []
        y = self.conv1(inputs)
        for block in self.blaze_:
            y = block(y)
            outs.append(y)
        for block in self.double_blaze_:
            y = block(y)
            outs.append(y)
        return [outs[-4], outs[-1]]





if __name__ == '__main__':
    m = BlazeNet(act='hard_swish')
    print(m)
    # print(m.out_shape)
    # for k, v in m.state_dict().items():
    #     print(k, v.shape)
    # inp = {'image': torch.randn((1, 3, 320, 320))}
    # for i in m(inp):
    #     print(i.shape)