import torch
from torch import nn
import torch.nn.functional as F

from math import ceil

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

class Conv2dDW(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Conv2dDW, self).__init__()

        # depthwise layer
        self.conv_dw = nn.Conv2d(in_channels,
                                 in_channels,
                                 groups=in_channels,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 stride=stride,
                                 bias=False)

        # batchnorm that follows the depthwise layer
        self.conv_dw_bn = nn.BatchNorm2d(in_channels)

        # pointwise layer
        self.conv_pw = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 padding=0,
                                 stride=1,
                                 bias=False)

        # batchnorm that follows th pointwise layer
        self.conv_pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # depthwise convolution
        out = self.conv_dw(x)
        out = self.conv_dw_bn(out)
        out = F.relu(out)

        # pointwise convolution
        out = self.conv_pw(out)
        out = self.conv_pw_bn(out)
        out = F.relu(out)

        return out


class MobileNetV1Conv224(nn.Module):
    """
    MobileNet Architecture described in the paper "MobileNets: Efficient 
    Convolutional Neural Networks for Mobile Vision Applications" using
    standard convolutional layers
    https://arxiv.org/pdf/1704.04861.pdf

    Pytorch params: 29,294,088 
    Paper mentioned: 29.3 million
    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, alpha=1.0, num_classes=1000):
        """
        Parameters
        ----------
        name : str
            The name of the animal
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal (default is 4)
        """

        super(MobileNetV1Conv224, self).__init__()

        self.alpha = alpha
        self.num_classes = num_classes

        # Input tensor
        # [N, 3, 224, 224]
        
        self.conv_1 = Conv2dBn(3, 
                                ceil(self.alpha*32), 
                                kernel_size=3, padding=1, stride=2)
        # [N, 32, 112, 112]

        self.conv_2 = Conv2dBn(ceil(self.alpha*32), 
                                ceil(self.alpha*64), 
                                kernel_size=3, padding=1, stride=1)
        # [N, 64, 112, 112]

        self.conv_3 = Conv2dBn(ceil(self.alpha*64), 
                                ceil(self.alpha*128), 
                                kernel_size=3, padding=1, stride=2)
        # [N, 128, 56, 56]

        self.conv_4 = Conv2dBn(ceil(self.alpha*128),
                                ceil(self.alpha*128),
                                kernel_size=3, padding=1, stride=1)
        # [N, 128, 56, 56]

        self.conv_5 = Conv2dBn(ceil(self.alpha*128),
                                ceil(self.alpha*256),
                                kernel_size=3, padding=1, stride=2)
        # [N, 256, 28, 28]

        self.conv_6 = Conv2dBn(ceil(self.alpha*256),
                                ceil(self.alpha*256),
                                kernel_size=3, padding=1, stride=1)
        # [N, 256, 28, 28]

        self.conv_7 = Conv2dBn(ceil(self.alpha*256),
                                ceil(self.alpha*512),
                                kernel_size=3, padding=1, stride=2)
        # [N, 512, 14, 14]

        self.conv_8 = Conv2dBn(ceil(self.alpha*512),
                                ceil(self.alpha*512),
                                kernel_size=3, padding=1, stride=1)
        # [N, 512, 14, 14]
        # repeated 5 times

        self.conv_9 = Conv2dBn(ceil(self.alpha*512),
                                ceil(self.alpha*1024),
                                kernel_size=3, padding=1, stride=2)
        # [N, 1024, 7, 7]

        self.conv_10 = Conv2dBn(ceil(self.alpha*1024),
                                 ceil(self.alpha*1024),
                                 kernel_size=3, padding=4, stride=2)
        # [N, 1024, 7, 7]

        self.avg_pool = nn.AvgPool2d(kernel_size=7, padding=0, stride=1)
        # [N, 1024, 1, 1]

        # probability obtained from the keras implementation of the model
        self.dropout = nn.Dropout(p=0.001)
        # [N, 1024, 1, 1]

        self.fc = nn.Linear(ceil(self.alpha*1024),
                            self.num_classes)
        # [N, 1000]

        self.softmax = nn.Softmax()
        # [N, 1000]

    def forward(self, image):
        """
        Forward propagation.

        Parameters
        ----------
        image : tensor
            Images, a tensor of dimensions (N, 3, 224, 224)
        
        Returns
        -------
        out_name : tensor
            Predictions from 
        """
                                                    # [N, 3,     224, 224]
        out = self.conv_1(image)                    # [N, a*32,  112, 112]
        out = self.conv_2(out)                      # [N, a*64,  112, 112]
        out = self.conv_3(out)                      # [N, a*128, 56,  56]
        out = self.conv_4(out)                      # [N, a*128, 56,  56]
        out = self.conv_5(out)                      # [N, a*256, 28,  28]
        out = self.conv_6(out)                      # [N, a*256, 28,  28]
        out = self.conv_7(out)                      # [N, a*512, 14,  14]

        for i in range(5):
            out = self.conv_8(out)                  # [N, a*512, 14, 14]

        out = self.conv_9(out)                      # [N, a*1024, 7, 7]
        out = self.conv_10(out)                     # [N, a*1024, 7, 7]
        out = self.avg_pool(out)                    # [N, a*1024, 1, 1]

        out = out.view(out.size(0), -1)             # [N, a*1024]
        out = self.dropout(out)                     # [N, a*1024]
        out = self.fc(out)                          # [N, n_classes]

        out = self.softmax(out)                     # [N, n_classes]

        return out

    def fu(self):
        """Gets and prints the spreadsheet's header columns

        Parameters
        ----------
        file_loc : str
            The file location of the spreadsheet
        print_cols : bool, optional
            A flag used to print the columns to the console (default is
            False)

        Returns
        -------
        list
            a list of strings used that are the header columns
        """
        pass

class MobileNetV1Dw224(nn.Module):
    """
    MobileNet Architecture described in the paper "MobileNets: Efficient 
    Convolutional Neural Networks for Mobile Vision Applications" using
    standard convolutional layers
    https://arxiv.org/pdf/1704.04861.pdf

    Total params:  4,231,976 
    Keras version: 4,253,864
    Paper mentioned 4.2 Million

    The number of parameters from a batchnorm layer presented in the keras
    summary of the full mobilenet model, is twice the number of parameters
    from a batchnorm layer in pytorch. This could explain the difference
    in the size of the models.

    ...

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, alpha=1.0, num_classes=1000):
        """
        Parameters
        ----------
        name : str
            The name of the animal
        sound : str
            The sound the animal makes
        num_legs : int, optional
            The number of legs the animal (default is 4)
        """

        super(MobileNetV1Dw224, self).__init__()

        self.alpha = alpha
        self.num_classes = num_classes

        # Input tensor
        # [N, 3, 224, 224]
        
        self.conv_1 = Conv2dBn(3, 
                               ceil(self.alpha*32), 
                               kernel_size=3, padding=1, stride=2)
        # [N, 32, 112, 112]

        self.conv_2 = Conv2dDW(ceil(self.alpha*32), 
                               ceil(self.alpha*64), 
                               kernel_size=3, padding=1, stride=1)
        # [N, 64, 112, 112]

        self.conv_3 = Conv2dDW(ceil(self.alpha*64), 
                               ceil(self.alpha*128), 
                               kernel_size=3, padding=1, stride=2)
        # [N, 128, 56, 56]

        self.conv_4 = Conv2dDW(ceil(self.alpha*128),
                               ceil(self.alpha*128),
                               kernel_size=3, padding=1, stride=1)
        # [N, 128, 56, 56]

        self.conv_5 = Conv2dDW(ceil(self.alpha*128),
                               ceil(self.alpha*256),
                               kernel_size=3, padding=1, stride=2)
        # [N, 256, 28, 28]

        self.conv_6 = Conv2dDW(ceil(self.alpha*256),
                               ceil(self.alpha*256),
                               kernel_size=3, padding=1, stride=1)
        # [N, 256, 28, 28]

        self.conv_7 = Conv2dDW(ceil(self.alpha*256),
                               ceil(self.alpha*512),
                               kernel_size=3, padding=1, stride=2)
        # [N, 512, 14, 14]

        self.conv_8 = Conv2dDW(ceil(self.alpha*512),
                               ceil(self.alpha*512),
                               kernel_size=3, padding=1, stride=1)
        # [N, 512, 14, 14]
        # repeated 5 times

        self.conv_9 = Conv2dDW(ceil(self.alpha*512),
                               ceil(self.alpha*1024),
                               kernel_size=3, padding=1, stride=2)
        # [N, 1024, 7, 7]

        self.conv_10 = Conv2dDW(ceil(self.alpha*1024),
                                ceil(self.alpha*1024),
                                kernel_size=3, padding=4, stride=2)
        # [N, 1024, 7, 7]

        self.avg_pool = nn.AvgPool2d(kernel_size=7, padding=0, stride=1)
        # [N, 1024, 1, 1]

        # probability obtained from the keras implementation of the model
        self.dropout = nn.Dropout(p=0.001)
        # [N, 1024, 1, 1]

        self.fc = nn.Linear(ceil(self.alpha*1024),
                            self.num_classes)
        # [N, 1000]

        self.softmax = nn.Softmax()
        # [N, 1000]

    def forward(self, image):
        """
        Forward propagation.

        Parameters
        ----------
        image : tensor
            Images, a tensor of dimensions (N, 3, 224, 224)
        
        Returns
        -------
        out_name : tensor
            Predictions from 
        """
                                                    # [N, 3,     224, 224]
        out = self.conv_1(image)                    # [N, a*32,  112, 112]
        out = self.conv_2(out)                      # [N, a*64,  112, 112]
        out = self.conv_3(out)                      # [N, a*128, 56,  56]
        out = self.conv_4(out)                      # [N, a*128, 56,  56]
        out = self.conv_5(out)                      # [N, a*256, 28,  28]
        out = self.conv_6(out)                      # [N, a*256, 28,  28]
        out = self.conv_7(out)                      # [N, a*512, 14,  14]

        for i in range(5):
            out = self.conv_8(out)                  # [N, a*512, 14, 14]

        out = self.conv_9(out)                      # [N, a*1024, 7, 7]
        out = self.conv_10(out)                     # [N, a*1024, 7, 7]
        out = self.avg_pool(out)                    # [N, a*1024, 1, 1]

        out = out.view(out.size(0), -1)             # [N, a*1024]
        out = self.dropout(out)                     # [N, a*1024]
        out = self.fc(out)                          # [N, n_classes]

        out = self.softmax(out)                     # [N, n_classes]

        return out

    def fu(self):
        """Gets and prints the spreadsheet's header columns

        Parameters
        ----------
        file_loc : str
            The file location of the spreadsheet
        print_cols : bool, optional
            A flag used to print the columns to the console (default is
            False)

        Returns
        -------
        list
            a list of strings used that are the header columns
        """
        pass
