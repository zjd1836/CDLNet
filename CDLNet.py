import math
import time
import os
import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as F
from models.attention.PoolLayer import HwPooling
from models.attention.RFAConv import RFAConv
from models.attention.DWConv import DWConv, DWConvNobr
from models.attention.PoolLayer import MixedPoolingChannel, MixedPoolingSpatial
from models.encoder.MobileViTv3V1 import MobileViTv3_v1


class CDLNet(nn.Module):
    def __init__(self, n_classes=7):
        super(CDLNet, self).__init__()
        self.encoder = MobileViTv3_v1(image_size=(512, 512), mode='small', isThree=True)
        self.encoder.load_pretrained_model('https://drive.google.com/file/d/1u6XFrSzYxhta5ZURia_ewMQan5uSCXLs/view?usp=drive_link')
        self.conv1 = RFAConv(128, 128, 3, 1)
        self.conv2 = RFAConv(256, 256, 3, 1)
        self.conv3 = RFAConv(320, 320, 3, 1)
        self.decoder1 = SS_Decoder()
        self.decoder2 = SS_Decoder()
        self.cd_decoder = CD_Decoder()
        self.classifier1 = nn.Conv2d(128, n_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, n_classes, kernel_size=1)
        self.classifierCD = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x1, x2):
        x_size = x1.size()
        ss1 = self.encoder(x1)
        ss2 = self.encoder(x2)
        ss1[0], ss1[1], ss1[2] = self.conv1(ss1[0]), self.conv2(ss1[1]), self.conv3(ss1[2])
        ss2[0], ss2[1], ss2[2] = self.conv1(ss2[0]), self.conv2(ss2[1]), self.conv3(ss2[2])
        sd1 = self.decoder1(ss1, ss2)
        sd2 = self.decoder2(ss2, ss1)
        bcd_out = self.cd_decoder(ss1, ss2)
        out1 = self.classifier1(sd1)
        out2 = self.classifier2(sd2)
        change = self.classifierCD(bcd_out)
        return F.interpolate(change, x_size[2:], mode='bilinear', align_corners=True),\
            F.interpolate(out1, x_size[2:], mode='bilinear', align_corners=True), \
            F.interpolate(out2, x_size[2:], mode='bilinear', align_corners=True)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, transform=False):
        super(BasicBlock, self).__init__()
        self.transform = transform
        self.conv1 = DWConvNobr(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DWConvNobr(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1),
            nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.transform:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Conv1x1(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, 1)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ss_block(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(ss_block, self).__init__()
        self.block1 = Conv1x1(in_cha, out_cha)
        self.block2 = DWConv(out_cha, out_cha)
        self.block3 = BasicBlock(out_cha, out_cha)
        self.pool = HwPooling(out_cha, out_cha)
        self.block4 = BasicBlock(out_cha, out_cha)
        self.block5 = nn.ConvTranspose2d(out_cha, out_cha, kernel_size=2, stride=2)

    def forward(self, x1, x2, y):
        if y.shape[1] != 2:
            x1 = self.block1(torch.cat((x1, y), dim=1))
        x1 = self.block2(x1)
        xs = x1 * F.softmax(torch.abs(x1 - x2), dim=1)
        xs = self.block3(xs)
        xs = self.pool(xs)
        xs = self.block4(xs)
        out = x1 + xs
        out = self.block5(out)
        return out

class SS_Decoder(nn.Module):
    def __init__(self):
        super(SS_Decoder, self).__init__()
        self.ss_decoder1 = ss_block(320, 320)
        self.ss_decoder2 = ss_block(576, 256)
        self.ss_decoder3 = ss_block(384, 128)
        self.ss_decoder4 = BasicBlock(128, 128)
        self._init_weight()

    def forward(self, s1, s2):

        y = torch.zeros([1, 2, 3]).cuda()
        x = self.ss_decoder1(s1[2], s2[2], y)
        x = self.ss_decoder2(s1[1], s2[1], x)
        x = self.ss_decoder3(s1[0], s2[0], x)
        x = self.ss_decoder4(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class cd_block(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(cd_block, self).__init__()
        self.cd_block1 = nn.Sequential(
            Conv1x1(in_cha, out_cha),
            DWConv(out_cha, out_cha))
        # 空间注意力
        self.pool1 = MixedPoolingChannel()
        self.cd_block2 = nn.Conv2d(3, 1, 7, padding=3, bias=False)
        self.cd_block3 = BasicBlock(out_cha, out_cha)
        # 通道注意力
        self.pool2 = MixedPoolingSpatial()
        self.cd_block4 = BasicBlock(out_cha, out_cha)
        # 融合上采样
        self.cd_block5 = BasicBlock(out_cha, out_cha)
        self.cd_block6 = nn.ConvTranspose2d(out_cha, out_cha, kernel_size=2, stride=2)

    def forward(self, x1, x2, x):
        if x.shape[1] == 2:
            xl = self.cd_block1(torch.cat((x1, x2), dim=1))
        else:
            xl = self.cd_block1(torch.cat((x1, x2, x), dim=1))
        v1 = torch.abs(x1 - x2)
        v2 = x1 + x2
        v3 = x1 * x2
        sl = self.cd_block2(torch.cat((self.pool1(v1), self.pool1(v2), self.pool1(v3)), dim=1))
        sl = torch.sigmoid(sl)
        sl = xl * sl
        sl = self.cd_block3(sl)
        sl = xl * sl
        cl = self.pool2(v1) + self.pool2(v2) + self.pool2(v3)
        cl = torch.sigmoid(cl)
        cl = xl * cl
        cl = self.cd_block4(cl)
        cl = xl * cl
        sc = sl + cl
        sc = self.cd_block5(sc)
        sc = xl + sc
        sc = self.cd_block6(sc)
        return sc

class CD_Decoder(nn.Module):
    def __init__(self):
        super(CD_Decoder, self).__init__()
        self.cd_decoder1 = cd_block(640, 320)
        self.cd_decoder2 = cd_block(832, 256)
        self.cd_decoder3 = cd_block(512, 128)
        self.cd_decoder4 = Conv1x1(128, 32)
        self.cd_decoder5 = BasicBlock(32, 32)
        self._init_weight()

    def forward(self, x1, x2):
        x = torch.zeros([1, 2, 3]).cuda()
        xl = self.cd_decoder1(x1[2], x2[2], x)
        xl = self.cd_decoder2(x1[1], x2[1], xl)
        xl = self.cd_decoder3(x1[0], x2[0], xl)
        xl = self.cd_decoder4(xl)
        xl = self.cd_decoder5(xl)
        return xl

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':

    test_data1 = torch.rand(1, 3, 512, 512).cuda()
    test_data2 = torch.rand(1, 3, 512, 512).cuda()
    model = CDLNet(n_classes=7)
    model = model.cuda()
    changes, preds1, preds2 = model(test_data1, test_data2)





