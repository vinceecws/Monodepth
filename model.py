import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def conv3x3(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=stride)

class IConv2d(nn.Module):
    
    def __init__(self, in_chn, out_chn, batchnorm=False):
        super(IConv2d, self).__init__()

        self.batchnorm = batchnorm
        self.conv = conv3x3(in_chn, out_chn, 1)
        self.relu = nn.ELU(inplace=True)
        nn.init.xavier_uniform_(self.conv.weight)

        if self.batchnorm:
            self.bn = nn.BatchNorm2d(out_chn)

    def forward(self, x):
        out = self.conv(x)
        if self.batchnorm:
            out = self.bn(out)
        out = self.relu(out)

        return out

class UpConv2d(nn.Module):

    def __init__(self, in_chn, out_chn, scale, batchnorm=False):
        super(UpConv2d, self).__init__()

        self.scale = scale
        self.conv = IConv2d(in_chn, out_chn, batchnorm)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        out = self.conv(out)
        
        return out

class Bottleneck(nn.Module):

    def __init__(self, in_chn, planes, stride, batchnorm=False):
        super(Bottleneck, self).__init__()

        self.batchnorm = batchnorm
        self.conv1 = conv1x1(in_chn, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = conv1x1(planes, planes * 4)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes * 4)

        self.downsample = None
        if stride != 1:
            layers = [conv1x1(in_chn, planes * 4, stride)]
            nn.init.xavier_uniform_(layers[0].weight)
            if self.batchnorm:
                layers += [nn.BatchNorm2d(planes * 4)]
            self.downsample = nn.Sequential(*layers)

        self.relu = nn.ELU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batchnorm:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Disparity2d(nn.Module):

    def __init__(self, in_chn, batchnorm=False):
        super(Disparity2d, self).__init__()

        self.conv = conv3x3(in_chn, 2)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        out = 0.3 * torch.sigmoid(out)

        return out

class VGGConv2dBlock(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, padding=1, batchnorm=False):
        super(VGGConv2dBlock, self).__init__()
        self.batchnorm = batchnorm

        self.conv1 = nn.Conv2d(in_chn, out_chn,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
        self.conv2 = nn.Conv2d(out_chn, out_chn,
                kernel_size=kernel_size,
                stride=2,
                padding=padding)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.relu = nn.ELU(inplace=True)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(out_chn)
            self.bn2 = nn.BatchNorm2d(out_chn)

    def forward(self, x):
        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm:
            out = self.bn2(out)
        out = self.relu(out)

        return out


class MonodepthVGG(nn.Module):
    def __init__(self, batchnorm=False):
        super(MonodepthVGG, self).__init__()
        self.batchnorm = batchnorm

        self.conv1 = VGGConv2dBlock(3, 32, 7, 3, self.batchnorm)
        self.conv2 = VGGConv2dBlock(32, 64, 5, 2, self.batchnorm)
        self.conv3 = VGGConv2dBlock(64, 128, 3, 1, self.batchnorm)
        self.conv4 = VGGConv2dBlock(128, 256, 3, 1, self.batchnorm)
        self.conv5 = VGGConv2dBlock(256, 512, 3, 1, self.batchnorm)
        self.conv6 = VGGConv2dBlock(512, 512, 3, 1, self.batchnorm)
        self.conv7 = VGGConv2dBlock(512, 512, 3, 1, self.batchnorm)

        upconv = UpConv2d

        self.upconv7 = upconv(512, 512, 2, self.batchnorm)
        self.iconv7 = IConv2d(1024, 512, self.batchnorm)

        self.upconv6 = upconv(512, 512, 2, self.batchnorm)
        self.iconv6 = IConv2d(1024, 512, self.batchnorm)
        
        self.upconv5 = upconv(512, 256, 2, self.batchnorm)
        self.iconv5 = IConv2d(512, 256, self.batchnorm)

        self.upconv4 = upconv(256, 128, 2, self.batchnorm)
        self.iconv4 = IConv2d(256, 128, self.batchnorm)
        self.disp4 = Disparity2d(128, self.batchnorm)

        self.upconv3 = upconv(128, 64, 2, self.batchnorm)
        self.iconv3 = IConv2d(64 + 64 + 2, 64, self.batchnorm)
        self.disp3 = Disparity2d(64, self.batchnorm)

        self.upconv2 = upconv(64, 32, 2, self.batchnorm)
        self.iconv2 = IConv2d(32 + 32 + 2, 32, self.batchnorm)
        self.disp2 = Disparity2d(32, self.batchnorm)

        self.upconv1 = upconv(32, 16, 2, self.batchnorm)
        self.iconv1 = IConv2d(16 + 2, 16, self.batchnorm)
        self.disp1 = Disparity2d(16, self.batchnorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):

        skip1 = self.conv1(x)
        skip2 = self.conv2(skip1)
        skip3 = self.conv3(skip2)
        skip4 = self.conv4(skip3)
        skip5 = self.conv5(skip4)
        skip6 = self.conv6(skip5)
        out = self.conv7(skip6)

        out = self.upconv7(out)
        out = torch.cat((out, skip6), dim=1)
        out = self.iconv7(out)

        out = self.upconv6(out)
        out = torch.cat((out, skip5), dim=1)
        out = self.iconv6(out)

        out = self.upconv5(out)
        out = torch.cat((out, skip4), dim=1)
        out = self.iconv5(out)

        out = self.upconv4(out)
        out = torch.cat((out, skip3), dim=1)
        out = self.iconv4(out)
        disp4 = self.disp4(out)
        udisp4 = F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.upconv3(out)
        out = torch.cat((out, skip2, udisp4), dim=1)
        out = self.iconv3(out)
        disp3 = self.disp3(out)
        udisp3 = F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.upconv2(out)
        out = torch.cat((out, skip1, udisp3), dim=1)
        out = self.iconv2(out)
        disp2 = self.disp2(out)
        udisp2 = F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.upconv1(out)
        out = torch.cat((out, udisp2), dim=1)
        out = self.iconv1(out)
        disp1 = self.disp1(out)

        return [disp1, disp2, disp3, disp4]

class Monodepth(nn.Module):

    def __init__(self, batchnorm=False):
        super(Monodepth, self).__init__()

        self.batchnorm = batchnorm

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_chn = 64
        self.layer1 = self._make_resblock(64, 3)
        self.layer2 = self._make_resblock(128, 4)
        self.layer3 = self._make_resblock(256, 6)
        self.layer4 = self._make_resblock(512, 3)

        # Decoder
        upconv = UpConv2d

        self.upconv6 = upconv(2048, 512, 2, self.batchnorm)
        self.iconv6 = IConv2d(512 + 1024, 512, self.batchnorm)
        
        self.upconv5 = upconv(512, 256, 2, self.batchnorm)
        self.iconv5 = IConv2d(256 + 512, 256, self.batchnorm)

        self.upconv4 = upconv(256, 128, 2, self.batchnorm)
        self.iconv4 = IConv2d(128 + 256, 128, self.batchnorm)
        self.disp4 = Disparity2d(128, self.batchnorm)

        self.upconv3 = upconv(128, 64, 2, self.batchnorm)
        self.iconv3 = IConv2d(64 + 64 + 2, 64, self.batchnorm)
        self.disp3 = Disparity2d(64, self.batchnorm)

        self.upconv2 = upconv(64, 32, 2, self.batchnorm)
        self.iconv2 = IConv2d(32 + 64 + 2, 32, self.batchnorm)
        self.disp2 = Disparity2d(32, self.batchnorm)

        self.upconv1 = upconv(32, 16, 2, self.batchnorm)
        self.iconv1 = IConv2d(16 + 2, 16, self.batchnorm)
        self.disp1 = Disparity2d(16, self.batchnorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def _make_resblock(self, planes, blocks):
        layers = []
        layers.append(Bottleneck(self.in_chn, planes, 2, self.batchnorm))
        self.in_chn = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_chn, planes, 1, self.batchnorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.batchnorm:
            out = self.bn1(out)
        x1 = self.relu(out)
        x2 = self.maxpool(x1)

        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        out = self.layer4(x5)

        out = self.upconv6(out)
        out = torch.cat((out, x5), dim=1)
        out = self.iconv6(out)

        out = self.upconv5(out)
        out = torch.cat((out, x4), dim=1)
        out = self.iconv5(out)

        out = self.upconv4(out)
        out = torch.cat((out, x3), dim=1)
        out = self.iconv4(out)
        disp4 = self.disp4(out)
        udisp4 = F.interpolate(disp4, scale_factor=2, mode='nearest')

        out = self.upconv3(out)
        out = torch.cat((out, x2, udisp4), dim=1)
        out = self.iconv3(out)
        disp3 = self.disp3(out)
        udisp3 = F.interpolate(disp3, scale_factor=2, mode='nearest')

        out = self.upconv2(out)
        out = torch.cat((out, x1, udisp3), dim=1)
        out = self.iconv2(out)
        disp2 = self.disp2(out)
        udisp2 = F.interpolate(disp2, scale_factor=2, mode='nearest')

        out = self.upconv1(out)
        out = torch.cat((out, udisp2), dim=1)
        out = self.iconv1(out)
        disp1 = self.disp1(out)

        return [disp1, disp2, disp3, disp4]

if __name__ == "__main__":
    print(MonodepthVGG(batchnorm=False).modules)

