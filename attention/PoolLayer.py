import random
import torch
import torch.nn as nn


# 组合池化add/cat
class CombinedPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(CombinedPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        return (max_pooled + avg_pooled) / 2


# 混合池化
class MixedPoolingSpatial(nn.Module):
    def __init__(self):
        super(MixedPoolingSpatial, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        return (max_pooled + avg_pooled) / 2


class MixedPoolingChannel(nn.Module):
    def __init__(self):
        super(MixedPoolingChannel, self).__init__()

    def forward(self, x):
        avg_pooled = torch.mean(x, dim=1, keepdim=True)
        max_pooled = torch.max(x, dim=1, keepdim=True, out=None)[0]
        return (max_pooled + avg_pooled) / 2

class HwPooling(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(HwPooling, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        a_h = self.conv2(self.relu(self.conv1(x_h))).sigmoid()
        a_w = self.conv2(self.relu(self.conv1(x_w))).sigmoid()

        out = x * a_w * a_h


        return out