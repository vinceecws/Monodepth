import torch
import torch.nn as nn
import torch.nn.functional as F

#This is the Xception model separable convolution
#It first performs the pointwise (1x1 kernel) convolution
#Then follows with the depthwise (out_chn nxn kernel) convolution
#same flag produces pseudo-dimension-preservation when set to True (e.g only when stride = 1, input is auto-padded so that output_size = input_size)
# TODO: Support odd number in image shape
class SeparableConv2d(nn.Module):
        def __init__(self, in_chn, out_chn, kernel_size, stride=1, dilation=1, batchnorm=False, relu=False, same=True):
                super(SeparableConv2d, self).__init__()

                self.batchnorm = batchnorm
                self.relu = relu
                self.same = same

                if self.same:
                    p = dilation * (kernel_size - 1) // 2
                    self.padding = (p, p, p, p)
                else:
                    self.padding = (0, 0, 0, 0)
                
                self.pointwise = nn.Conv2d(in_chn, out_chn, kernel_size=1)
                self.depthwise = nn.Conv2d(out_chn, out_chn, kernel_size, stride, 0, dilation, groups=out_chn)

                if self.batchnorm:
                        self.batch_norm = nn.BatchNorm2d(out_chn)

        def forward(self, x):
                x = self.pointwise(x)
                x = F.pad(x, self.padding)
                x = self.depthwise(x)

                if self.batchnorm:
                        x = self.batch_norm(x)
                if self.relu:
                        x = F.relu(x)

                return x

class XceptionBlock(nn.Module):
        def __init__(self, in_chn, chn_list, skip_connection, stride=1, batchnorm=False, relu=False):
                super(XceptionBlock, self).__init__()

                assert(len(chn_list) == 3)
                assert(skip_connection in ['conv', 'sum', 'none'])
                
                layers = []
                for i in range(3):
                    layers += [SeparableConv2d(
                                    in_chn if i == 0 else chn_list[i-1],
                                    chn_list[i],
                                    kernel_size=3,
                                    stride=stride if i == 2 else 1,
                                    dilation=1,
                                    batchnorm=batchnorm,
                                    relu=relu,
                                    same=True)]
                self.conv = nn.Sequential(*layers)

                self.skip_con = skip_connection
                if self.skip_con == 'conv':
                        self.conv2d_con = nn.Conv2d(in_chn, chn_list[-1], 1, stride) 

        def forward(self, x):
                residual = self.conv(x)

                if self.skip_con == 'conv':
                        out = residual + self.conv2d_con(x)
                elif self.skip_con == 'sum':
                        out = residual + x
                elif self.skip_con == 'none':
                        out = residual
                
                return out

class Xception(nn.Module):
    def __init__(self, batchnorm=False, relu=False):
        super(Xception, self).__init__()

        # Entry Flow
        layers = []
        layers += [nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)]
        layers += [nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)]
        self.entry_flow = nn.Sequential(*layers)

        self.entry_block1 = XceptionBlock(
                                64, [128, 128, 128],
                                skip_connection='conv', stride=2,
                                batchnorm=batchnorm,
                                relu=relu)
        self.entry_block2 = XceptionBlock(
                                128, [256, 256, 256],
                                skip_connection='conv',
                                stride=2,
                                batchnorm=batchnorm,
                                relu=relu)
        self.entry_block3 = XceptionBlock(
                                256, [728, 728, 728],
                                skip_connection='conv',
                                stride=2,
                                batchnorm=batchnorm,
                                relu=relu)

        # Middle Flow is repeated 16 times
        block_list = []
        for i in range(16):
            block_list += [XceptionBlock(
                                728, [728, 728, 728],
                                skip_connection='sum',
                                stride=1,
                                batchnorm=batchnorm,
                                relu=relu)]
        self.middle_block = nn.Sequential(*block_list)

        # Exit Flow
        # TODO Add dilation rate=2
        self.exit_block1 = XceptionBlock(
                                728, [728, 1024, 1024],
                                skip_connection='conv',
                                stride=1,
                                batchnorm=batchnorm,
                                relu=relu)
        self.exit_block2 = XceptionBlock(
                                1024, [1536, 1536, 2048],
                                skip_connection='none',
                                stride=1,
                                batchnorm=batchnorm,
                                relu=relu)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.entry_block1(x)
        low_level_feat = x
        x = self.entry_block2(x)
        x = self.entry_block3(x)

        x = self.middle_block(x)

        x = self.exit_block1(x)
        x = self.exit_block2(x)

        return x, low_level_feat

