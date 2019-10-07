import torch
from torch import nn
import torch.nn.functional as F

class Conv2dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Conv2dBn, self).__init__()
        
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        return out

class Darknet19(nn.Module):

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()

        self.num_classes = num_classes

        # Input tensor
        # [3, 224, 224]
        
        self.conv_1 = Conv2dBn(3,
                                32,
                                kernel_size=3, padding=1, stride=1)
        # [32, 224, 224]

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [32, 112, 112]

        self.conv_2 = Conv2dBn(32,
                                64,
                                kernel_size=3, padding=1, stride=1)
        # [64, 112, 112]

        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [64, 56, 56]

        self.conv_3 = Conv2dBn(64,
                                128,
                                kernel_size=3, padding=1, stride=1)
        # [128, 56, 56]

        self.conv_4 = Conv2dBn(128,
                                64,
                                kernel_size=1, padding=0, stride=1)
        # [64, 56, 56]

        self.conv_5 = Conv2dBn(64,
                                128,
                                kernel_size=3, padding=1, stride=1)
        # [128, 56, 56]

        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [128, 28, 28]

        self.conv_6 = Conv2dBn(128,
                                256,
                                kernel_size=3, padding=1, stride=1)
        # [256, 28, 28]

        self.conv_7 = Conv2dBn(256,
                                128,
                                kernel_size=1, padding=0, stride=1)
        # [128, 28, 28]

        self.conv_8 = Conv2dBn(128,
                                256,
                                kernel_size=3, padding=1, stride=1)
        # [256, 28, 28]

        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [256, 14, 14]

        self.conv_9 = Conv2dBn(256,
                                512,
                                kernel_size=3, padding=1, stride=1)
        # [512, 14, 14]

        self.conv_10 = Conv2dBn(512,
                                 256,
                                 kernel_size=1, padding=0, stride=1)
        # [256, 14, 14]

        self.conv_11 = Conv2dBn(256,
                                 512,
                                 kernel_size=3, padding=1, stride=1)
        # [512, 14, 14]

        self.conv_12 = Conv2dBn(512,
                                 256,
                                 kernel_size=1, padding=0, stride=1)
        # [256, 14, 14]

        self.conv_13 = Conv2dBn(256,
                                 512,
                                 kernel_size=3, padding=1, stride=1)
        # [512, 14, 14]

        self.max_pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [512, 7, 7]

        self.conv_14 = Conv2dBn(512,
                                1024,
                                kernel_size=3, padding=1, stride=1)
        # [1024, 7, 7]

        self.conv_15 = Conv2dBn(1024,
                                512,
                                kernel_size=1, padding=0, stride=1)
        # [512, 7, 7]

        self.conv_16 = Conv2dBn(512,
                                1024,
                                kernel_size=3, padding=1, stride=1)
        # [1024, 7, 7]

        self.conv_17 = Conv2dBn(1024,
                                512,
                                kernel_size=1, padding=0, stride=1)
        # [512, 7, 7]

        self.conv_18 = Conv2dBn(512,
                                1024,
                                kernel_size=3, padding=1, stride=1)
        # [1024, 7, 7]

        self.conv_19 = Conv2dBn(1024,
                                1000,
                                kernel_size=1, padding=0, stride=1)
        # [1000, 7, 7]

        self.avg_pool = nn.AvgPool2d(kernel_size=7, padding=0, stride=1)


    def forward(self, x):

        out = self.conv_1(x)
        out = self.max_pool_1(out)

        out = self.conv_2(out)
        out = self.max_pool_2(out)
        
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.max_pool_3(out)
        
        out = self.conv_6(out)
        out = self.conv_7(out)
        out = self.conv_8(out)
        out = self.max_pool_4(out)
        
        out = self.conv_9(out)
        out = self.conv_10(out)
        out = self.conv_11(out)
        out = self.conv_12(out)
        out = self.conv_13(out)
        out = self.max_pool_5(out)

        out = self.conv_14(out)
        out = self.conv_15(out)
        out = self.conv_16(out)
        out = self.conv_17(out)
        out = self.conv_18(out)
        
        out = self.conv_19(out)

        out = self.avg_pool(out)

        return out