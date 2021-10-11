import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .layers import ConvBNReLU, ConvBN
import cv2
import numpy as np
import matplotlib.pyplot as plt


# def SyncBatchNorm(*args, **kwargs):
#     """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm instead"""
#     if paddle.get_device() == 'cpu':
#         return nn.BatchNorm(*args, **kwargs)
#     else:
#         return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=False):
        super(ConvBNReLU, self).__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=(stride, stride), padding=kernel_size // 2, bias=bias)
        self._batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 bias=False):
        super(ConvBN, self).__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=(stride, stride), padding=kernel_size // 2, bias=bias)
        self._batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class HRNet(nn.Module):
    """
    The HRNet implementation based on PaddlePaddle.
    The original article refers to
    Jingdong Wang, et, al. "HRNet：Deep High-Resolution Representation Learning for Visual Recognition"
    (https://arxiv.org/pdf/1908.07919.pdf).
    Args:
        pretrained (str): The path of pretrained model.
        stage1_num_modules (int): Number of modules for stage1. Default 1.
        stage1_num_blocks (list): Number of blocks per module for stage1. Default [4].
        stage1_num_channels (list): Number of channels per branch for stage1. Default [64].
        stage2_num_modules (int): Number of modules for stage2. Default 1.
        stage2_num_blocks (list): Number of blocks per module for stage2. Default [4, 4]
        stage2_num_channels (list): Number of channels per branch for stage2. Default [18, 36].
        stage3_num_modules (int): Number of modules for stage3. Default 4.
        stage3_num_blocks (list): Number of blocks per module for stage3. Default [4, 4, 4]
        stage3_num_channels (list): Number of channels per branch for stage3. Default [18, 36, 72].
        stage4_num_modules (int): Number of modules for stage4. Default 3.
        stage4_num_blocks (list): Number of blocks per module for stage4. Default [4, 4, 4, 4]
        stage4_num_channels (list): Number of channels per branch for stage4. Default [18, 36, 72. 144].
        has_se (bool): Whether to use Squeeze-and-Excitation module. Default False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

    def __init__(self,
                 pretrained=None,
                 stage1_num_modules=1,
                 stage1_num_blocks=[4],
                 stage1_num_channels=[64],
                 stage2_num_modules=1,
                 stage2_num_blocks=[4, 4],
                 stage2_num_channels=[18, 36],
                 stage3_num_modules=4,
                 stage3_num_blocks=[4, 4, 4],
                 stage3_num_channels=[18, 36, 72],
                 stage4_num_modules=3,
                 stage4_num_blocks=[4, 4, 4, 4],
                 stage4_num_channels=[18, 36, 72, 144],
                 has_se=False,
                 align_corners=False):
        super(HRNet, self).__init__()
        self.pretrained = pretrained
        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels
        self.has_se = has_se
        self.align_corners = align_corners
        self.feat_channels = [sum(stage4_num_channels)]

        self.conv_layer1_1 = ConvBNReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=False)

        self.conv_layer1_2 = ConvBNReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            bias=False)

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=self.stage1_num_blocks[0],
            num_filters=self.stage1_num_channels[0],
            has_se=has_se)

        self.tr1 = TransitionLayer(
            in_channels=[self.stage1_num_channels[0] * 4],
            out_channels=self.stage2_num_channels)

        self.st2 = Stage(
            num_channels=self.stage2_num_channels,
            num_modules=self.stage2_num_modules,
            num_blocks=self.stage2_num_blocks,
            num_filters=self.stage2_num_channels,
            has_se=self.has_se,
            align_corners=align_corners)

        self.tr2 = TransitionLayer(
            in_channels=self.stage2_num_channels,
            out_channels=self.stage3_num_channels)
        self.st3 = Stage(
            num_channels=self.stage3_num_channels,
            num_modules=self.stage3_num_modules,
            num_blocks=self.stage3_num_blocks,
            num_filters=self.stage3_num_channels,
            has_se=self.has_se,
            align_corners=align_corners)

        self.tr3 = TransitionLayer(
            in_channels=self.stage3_num_channels,
            out_channels=self.stage4_num_channels)
        self.st4 = Stage(
            num_channels=self.stage4_num_channels,
            num_modules=self.stage4_num_modules,
            num_blocks=self.stage4_num_blocks,
            num_filters=self.stage4_num_channels,
            has_se=self.has_se,
            align_corners=align_corners)

    def forward(self, x):
        conv1 = self.conv_layer1_1(x)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)

        tr1 = self.tr1([la1])
        st2 = self.st2(tr1)

        tr2 = self.tr2(st2)
        st3 = self.st3(tr2)

        tr3 = self.tr3(st3)
        st4 = self.st4(tr3)

        x0_h, x0_w = st4[0].shape[2:]
        x1 = F.interpolate(
            st4[1], (x0_h, x0_w),
            mode='bilinear',
            align_corners=self.align_corners)
        x2 = F.interpolate(
            st4[2], (x0_h, x0_w),
            mode='bilinear',
            align_corners=self.align_corners)
        x3 = F.interpolate(
            st4[3], (x0_h, x0_w),
            mode='bilinear',
            align_corners=self.align_corners)
        x = torch.cat([st4[0], x1, x2, x3], dim=1)

        return [x]


class Layer1(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = nn.ModuleList()

        for i in range(num_blocks):
            bottleneck_block = BottleneckBlock(
                    num_channels=num_channels if i == 0 else num_filters * 4,
                    num_filters=num_filters,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False)
            self.bottleneck_block_list.append(bottleneck_block)

    def forward(self, x):
        # conv = x
        for block_func in self.bottleneck_block_list:
            x = block_func(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()

        num_in = len(in_channels)
        num_out = len(out_channels)
        self.conv_bn_func_list = nn.ModuleList()
        for i in range(num_out):
            residual = None
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = ConvBNReLU(
                        in_channels=in_channels[i],
                        out_channels=out_channels[i],
                        kernel_size=3,
                        bias=False)
            else:
                residual = ConvBNReLU(
                        in_channels=in_channels[-1],
                        out_channels=out_channels[i],
                        kernel_size=3,
                        stride=2,
                        bias=False)
            self.conv_bn_func_list.append(residual)

    def forward(self, x):
        outs = []
        for idx, conv_bn_func in enumerate(self.conv_bn_func_list):
            if conv_bn_func is None:
                outs.append(x[idx])
            else:
                if idx < len(x):
                    outs.append(conv_bn_func(x[idx]))
                else:
                    outs.append(conv_bn_func(x[-1]))
        return outs


class Branches(nn.Module):
    def __init__(self,
                 num_blocks,
                 in_channels,
                 out_channels,
                 has_se=False):
        super(Branches, self).__init__()

        self.basic_block_list = nn.ModuleList()

        for i in range(len(out_channels)):
            self.basic_block_list.append(nn.ModuleList())
            for j in range(num_blocks[i]):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block_func = BasicBlock(
                        num_channels=in_ch,
                        num_filters=out_channels[i],
                        has_se=has_se)
                self.basic_block_list[i].append(basic_block_func)

    def forward(self, x):
        outs = []
        assert len(x) == len(self.basic_block_list)

        for idx, conv_ls in enumerate(self.basic_block_list):
            input = x[idx]
            for conv in conv_ls:
                input = conv(input)
            outs.append(input)
        # for idx, input in enumerate(x):
        #     conv = input
        #     for basic_block_func in self.basic_block_list[idx]:
        #         for basic_conv in basic_block_func:
        #             conv = basic_conv(conv)
        #     outs.append(conv)
        return outs


class BottleneckBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            bias=False)

        self.conv2 = ConvBNReLU(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            bias=False)

        self.conv3 = ConvBN(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            bias=False)

        if self.downsample:
            self.conv_down = ConvBN(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                bias=False)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16)

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv3 = self.se(conv3)

        y = conv3 + residual
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            bias=False)
        self.conv2 = ConvBN(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            bias=False)

        if self.downsample:
            self.conv_down = ConvBNReLU(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                bias=False)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16)

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv2 = self.se(conv2)

        y = conv2 + residual
        y = F.relu(y)
        return y


class SELayer(nn.Module):
    def __init__(self, num_channels, num_filters, reduction_ratio):
        super(SELayer, self).__init__()

        self.pool2d_gap = nn.AdaptiveAvgPool2d(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)

        # TODO linear initialize with uniform (-std, std)
        # stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch)

        # stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters)

    def forward(self, x):
        pool = self.pool2d_gap(x)
        pool = torch.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = torch.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = x * excitation
        return out


class Stage(nn.Module):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 align_corners=False):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = nn.ModuleList()
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                stage_func = HighResolutionModule(
                    num_channels=num_channels,
                    num_blocks=num_blocks,
                    num_filters=num_filters,
                    has_se=has_se,
                    multi_scale_output=False,
                    align_corners=align_corners)
            else:
                stage_func = HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        align_corners=align_corners)

            self.stage_func_list.append(stage_func)
        # print(self._num_modules)

    def forward(self, x):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self,
                 num_channels,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 align_corners=False):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            num_blocks=num_blocks,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            align_corners=align_corners)

    def forward(self, x):
        out = self.branches_func(x)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 align_corners=False):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners

        self.residual_func_list = nn.ModuleList()
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                if j > i:
                    residual_func = ConvBN(
                        in_channels=in_channels[j],
                        out_channels=out_channels[i],
                        kernel_size=1,
                        bias=False)
                    self.residual_func_list.append(residual_func)
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            residual_func = ConvBN(
                                in_channels=pre_num_filters,
                                out_channels=out_channels[i],
                                kernel_size=3,
                                stride=2,
                                bias=False)
                            pre_num_filters = out_channels[i]
                        else:
                            residual_func = ConvBNReLU(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[j],
                                    kernel_size=3,
                                    stride=2,
                                    bias=False)
                            pre_num_filters = out_channels[j]
                        self.residual_func_list.append(residual_func)

    def forward(self, x):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = x[i]
            residual_shape = residual.shape[-2:]
            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](x[j])
                    residual_func_idx += 1

                    y = F.interpolate(
                        y,
                        residual_shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                    residual = residual + y
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = residual + y

            residual = F.relu(residual)
            outs.append(residual)

        return outs


def HRNet_W18_Small_V1(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[1],
        stage1_num_channels=[32],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[16, 32],
        stage3_num_modules=1,
        stage3_num_blocks=[2, 2, 2],
        stage3_num_channels=[16, 32, 64],
        stage4_num_modules=1,
        stage4_num_blocks=[2, 2, 2, 2],
        stage4_num_channels=[16, 32, 64, 128],
        **kwargs)
    return model


def HRNet_W18_Small_V2(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[2],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[18, 36],
        stage3_num_modules=1,
        stage3_num_blocks=[2, 2, 2],
        stage3_num_channels=[18, 36, 72],
        stage4_num_modules=1,
        stage4_num_blocks=[2, 2, 2, 2],
        stage4_num_channels=[18, 36, 72, 144],
        **kwargs)
    return model


def HRNet_W18(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[18, 36],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[18, 36, 72],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[18, 36, 72, 144],
        **kwargs)
    return model



class FCN(nn.Module):
    """
    A simple implementation for FCN based on PaddlePaddle.
    The original article refers to
    Evan Shelhamer, et, al. "Fully Convolutional Networks for Semantic Segmentation"
    (https://arxiv.org/abs/1411.4038).
    Args:
        num_classes (int): The unique number of target classes.
        backbone (nn.Module): Backbone networks.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1, ),
                 channels=None,
                 align_corners=False,
                 pretrained=None):
        super(FCN, self).__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = FCNHead(num_classes, backbone_indices, backbone_channels,
                            channels)

        self.align_corners = align_corners
        self.pretrained = pretrained

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]


class FCNHead(nn.Module):
    """
    A simple implementation for FCNHead based on PaddlePaddle
    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices of output of backbone.
            Default: (-1, ).
        channels (int, optional): The channels between conv layer and the last layer of FCNHead.
            If None, it will be the number of channels of input features. Default: None.
        # pretrained (str, optional): The path of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone_indices=(-1, ),
                 backbone_channels=(270, ),
                 channels=None):
        super(FCNHead, self).__init__()

        self.num_classes = num_classes
        self.backbone_indices = backbone_indices
        if channels is None:
            channels = backbone_channels[0]

        self.conv_1 = ConvBNReLU(
            in_channels=backbone_channels[0],
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True)
        self.cls = nn.Conv2d(
            in_channels=channels,
            out_channels=self.num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0)

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.conv_1(x)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list



class FaceSeg(object):
    def __init__(self, model_path='/home/ubuntu/Documents/pycharm/blazeface/weights/seg_model_384.pt'):
        self.seg = FCN(num_classes=2, backbone=HRNet_W18()).to('cuda:0')
        para_state_dict = torch.load(model_path)
        self.seg.load_state_dict(para_state_dict)
        self.seg.eval()

    @staticmethod
    def input_transform(image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image = (image / 255.)[np.newaxis, :, :, :]
        image = np.transpose(image, (0, 3, 1, 2)).astype(np.float32)
        image_input = torch.from_numpy(image)
        return image_input

    @staticmethod
    def output_transform(output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output

    def get_mask(self, image):
        image_input = self.input_transform(image)
        image_input = image_input.to('cuda:0')
        with torch.no_grad():
            logits = self.seg(image_input)
        pred = torch.argmax(logits[0], dim=1)
        pred = pred.cpu().numpy()
        mask = np.squeeze(pred).astype('uint8')

        mask = self.output_transform(mask, shape=image.shape[:2])
        return mask


if __name__ == '__main__':
    # inp = torch.randn((1, 3, 256, 256))

    # ckpt = torch.load('/home/ubuntu/Documents/pycharm/blazeface/weights/seg_model_384.pt')
    #
    # m = FCN(num_classes=2, backbone=HRNet_W18())
    # curPath = os.path.abspath(os.path.dirname(__file__))
    seg = FaceSeg()
    img = cv2.imread('/home/ubuntu/Documents/pycharm/blazeface/samples/photo_test.jpg')
    mask = seg.get_mask(img)
    plt.imshow(mask)
    plt.show()


    # print(curPath)