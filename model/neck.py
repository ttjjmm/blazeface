import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_swish(x):
    return x * F.relu6(x + 3) / 6.


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
        if self.act == "relu":
            x = F.relu(x)
        elif self.act == "relu6":
            x = F.relu6(x)
        elif self.act == 'leaky':
            x = F.leaky_relu(x)
        elif self.act == 'hard_swish':
            x = hard_swish(x)
        elif self.act is None:
            return x
        else:
            raise NotImplementedError
        return x


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.conv1_fpn = ConvBNLayer(
            in_channels,
            out_channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            act='leaky')
        self.conv2_fpn = ConvBNLayer(
            in_channels,
            out_channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            act='leaky')
        self.conv3_fpn = ConvBNLayer(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            padding=1,
            stride=1,
            act='leaky')

    def forward(self, x):
        output1 = self.conv1_fpn(x[0])
        output2 = self.conv2_fpn(x[1])
        up2 = F.interpolate(output2, size=output1.shape[-2:], mode='nearest')
        output1 += up2
        output1 = self.conv3_fpn(output1)
        return output1, output2


class SSH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSH, self).__init__()
        assert out_channels % 4 == 0
        self.conv0_ssh = ConvBNLayer(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            padding=1,
            stride=1,
            act=None)
        self.conv1_ssh = ConvBNLayer(
            out_channels // 2,
            out_channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            act='leaky')
        self.conv2_ssh = ConvBNLayer(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            act=None)
        self.conv3_ssh = ConvBNLayer(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            act='leaky')
        self.conv4_ssh = ConvBNLayer(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            padding=1,
            stride=1,
            act=None)

    def forward(self, x):
        conv0 = self.conv0_ssh(x)
        conv1 = self.conv1_ssh(conv0)
        conv2 = self.conv2_ssh(conv1)
        conv3 = self.conv3_ssh(conv2)
        conv4 = self.conv4_ssh(conv3)
        concat = torch.cat([conv0, conv2, conv4], dim=1)
        return F.relu(concat)


class BlazeNeck(nn.Module):
    def __init__(self, in_channel, neck_type="None"):
        super(BlazeNeck, self).__init__()
        self.neck_type = neck_type
        self.reture_input = False
        self._out_channels = in_channel
        if self.neck_type == 'None':
            self.reture_input = True
        if "fpn" in self.neck_type:
            self.fpn = FPN(self._out_channels[0], self._out_channels[1])
            self._out_channels = [self._out_channels[0] // 2, self._out_channels[1] // 2]
        if "ssh" in self.neck_type:
            self.ssh1 = SSH(self._out_channels[0], self._out_channels[0])
            self.ssh2 = SSH(self._out_channels[1], self._out_channels[1])
            self._out_channels = [self._out_channels[0], self._out_channels[1]]
        self.out_shape = [self._out_channels[0], self._out_channels[1]]

    def forward(self, inputs):
        if self.reture_input:
            return inputs
        output1, output2 = None, None
        if "fpn" in self.neck_type:
            backout_4, backout_1 = inputs
            output1, output2 = self.fpn([backout_4, backout_1])
        if self.neck_type == "only_fpn":
            return [output1, output2]
        if self.neck_type == "only_ssh":
            output1, output2 = inputs

        feature1 = self.ssh1(output1)
        feature2 = self.ssh2(output2)
        return [feature1, feature2]

    # @property
    # def out_shape(self):
    #     return [
    #         ShapeSpec(channels=c)
    #         for c in [self._out_channels[0], self._out_channels[1]]
    #     ]