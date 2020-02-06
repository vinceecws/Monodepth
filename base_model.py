from __future__ import absolute_import, division, print_function
from xception import SeparableConv2d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import importlib

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        
        return x

class conv(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size, stride=1, dilation=1, batchnorm=True, relu=True, same=True):
        super(conv, self).__init__()

        self.kernel_size = kernel_size
        self.conv1 = SeparableConv2d(in_chn, 
                                    out_chn,
                                    kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    batchnorm=batchnorm,
                                    relu=relu,
                                    same=same)

    def forward(self, x):
        x = self.conv1(x)

        return x

class upconv(nn.Module):
    
    def __init__(self, in_chn, out_chn, kernel_size, scale, stride=1, dilation=1, batchnorm=True, relu=True, same=True):
        super(upconv, self).__init__()

        self.scale = scale
        self.conv1 = conv(in_chn, 
                          out_chn,
                          kernel_size,
                          stride=stride,
                          dilation=dilation,
                          batchnorm=batchnorm,
                          relu=relu,
                          same=same)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv1(x)

        return x

class get_disp(nn.Module):

    def __init__(self, in_chn, batchnorm=True, relu=False):
        super(get_disp, self).__init__()

        self.conv1 = conv(in_chn,
                          2,
                          3,
                          stride=1,
                          dilation=1,
                          batchnorm=batchnorm,
                          relu=relu,
                          same=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = 0.3 * self.sigmoid(x) #d_max = 0.3 * output image width (from paper)

        return x

class Encoder(nn.Module):
    def __init__(self, freeze=False, batchnorm=True, pretrained=True):
        super(Encoder, self).__init__()

        self.En = models.resnet101(pretrained=pretrained)

        if batchnorm:
            self.En = nn.Sequential(*list(self.En.children())[:-2])
        else:
            self.En = nn.Sequential(*self.remove_bn(list(self.En.children()))[:-2])

        if freeze:
            for param in self.En.parameters():
                param.requires_grad = False

        self.En_AvgPool = nn.AdaptiveAvgPool2d((4, 8))

    def forward(self, x):

        x = self.En(x) #H/32
        x = self.En_AvgPool(x) #H/64

        return x

    def remove_bn(self, module):
        for i in range(len(module)):
            if isinstance(module[i], nn.BatchNorm2d):
                module[i] = Identity()
            elif isinstance(module[i], nn.Sequential):
                module[i] = nn.Sequential(*self.remove_bn(list(module[i].children())))
            elif isinstance(module[i], models.resnet.Bottleneck):
                module[i].bn1 = Identity()
                module[i].bn2 = Identity()
                module[i].bn3 = Identity()
                if module[i].downsample:
                    module[i].downsample = nn.Sequential(*self.remove_bn(list(module[i].downsample.children())))

        return module

class ResNet101_md(nn.Module):

    def __init__(self, batchnorm=True, pretrained=True):
        super(ResNet101_md, self).__init__()

        #Encoder
        self.En = Encoder(freeze=False, batchnorm=batchnorm, pretrained=pretrained)

        #Decoder
        self.De_Upconv6 = upconv(2048, 512, 3, 2, stride=1, batchnorm=batchnorm)
        self.De_Conv6 = conv(512, 512, 3, stride=1, batchnorm=batchnorm)

        self.De_Upconv5 = upconv(512, 256, 3, 2, stride=1, batchnorm=batchnorm)
        self.De_Conv5 = conv(256, 256, 3, stride=1, batchnorm=batchnorm)

        self.De_Upconv4 = upconv(256, 128, 3, 2, stride=1, batchnorm=batchnorm)
        self.De_Conv4 = conv(128, 128, 3, stride=1, batchnorm=batchnorm)
        self.De_Disp4 = get_disp(128, batchnorm=batchnorm)

        self.De_Upconv3 = upconv(128, 64, 3, 2, stride=1, batchnorm=batchnorm)
        self.De_Conv3 = conv(64+2, 64, 3, stride=1, batchnorm=batchnorm) #Concat with udisp4
        self.De_Disp3 = get_disp(64, batchnorm=batchnorm)

        self.De_Upconv2 = upconv(64, 32, 3, 2, stride=1, batchnorm=batchnorm)
        self.De_Conv2 = conv(32+2, 32, 3, stride=1, batchnorm=batchnorm) #Concat with udisp3
        self.De_Disp2 = get_disp(32, batchnorm=batchnorm)

        self.De_Upconv1 = upconv(32, 16, 3, 2, stride=1, batchnorm=batchnorm)
        self.De_Conv1 = conv(16+2, 16, 3, stride=1, batchnorm=batchnorm) #Concat with udisp2
        self.De_Disp1 = get_disp(16, batchnorm=batchnorm)


    def forward(self, x):
        x = self.En(x) #H/64

        x = self.De_Upconv6(x)
        x = self.De_Conv6(x)

        x = self.De_Upconv5(x)
        x = self.De_Conv5(x)

        x = self.De_Upconv4(x)
        x = self.De_Conv4(x)
        self.disp4 = self.De_Disp4(x)
        self.udisp4 = F.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.De_Upconv3(x)
        x = torch.cat((x, self.udisp4), 1)
        x = self.De_Conv3(x)
        self.disp3 = self.De_Disp3(x)
        self.udisp3 = F.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.De_Upconv2(x)
        x = torch.cat((x, self.udisp3), 1)
        x = self.De_Conv2(x)
        self.disp2 = self.De_Disp2(x)
        self.udisp2 = F.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.De_Upconv1(x)
        x = torch.cat((x, self.udisp2), 1)
        x = self.De_Conv1(x)
        self.disp1 = self.De_Disp1(x)

        return self.disp1, self.disp2, self.disp3, self.disp4
