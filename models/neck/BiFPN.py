import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.LeakyReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 4))
        self.w1_relu = nn.LeakyReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 4))
        self.w2_relu = nn.LeakyReLU()
    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon
        
        p7_td = p7_x
        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, size=p6_x.shape[-2:]))        
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, size=p5_x.shape[-2:]))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, size=p4_x.shape[-2:]))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, size=p3_x.shape[-2:]))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.Upsample(size=p4_x.shape[-2:])(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.Upsample(size=p5_x.shape[-2:])(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.Upsample(size=p6_x.shape[-2:])(p5_out))
        p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * nn.Upsample(size=p7_x.shape[-2:])(p6_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]
    
class BiFPN(nn.Module):
    def __init__(self, in_channels, inner_channels=64, num_layers=2, epsilon=0.0001):
        super(BiFPN, self).__init__()

        inplace = True

        inner_channels = inner_channels // 4
        self.p3 = nn.Conv2d(in_channels[0], inner_channels, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(in_channels[1], inner_channels, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(in_channels[2], inner_channels, kernel_size=1, stride=1, padding=0)
        
        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(in_channels[3], inner_channels, kernel_size=1, stride=1, padding=0)
        
        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = ConvBlock(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(inner_channels))
        self.bifpn = nn.Sequential(*bifpns)

        self.out_channels = inner_channels * 5

        self.conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, inputs):
        c2, c3, c4, c5 = inputs
        
        # Calculate the input column of BiFPN
        p3_x = self.p3(c2)        
        p4_x = self.p4(c3)
        p5_x = self.p5(c4)
        p6_x = self.p6(c5)
        p7_x = self.p7(p6_x)
        
        features = [p3_x, p4_x, p5_x, p6_x, p7_x]
        p3_x, p4_x, p5_x, p6_x, p7_x = self.bifpn(features)
        out = self._upsample_cat(p3_x, p4_x, p5_x, p6_x, p7_x)
        out = self.conv(out)
        return out
    

    def _upsample_cat(self, p3, p4, p5, p6, p7):
        h, w = p3.size()[2:]
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        p6 = F.interpolate(p6, size=(h, w))
        p7 = F.interpolate(p7, size=(h, w))
        return torch.cat([p3, p4, p5, p6, p7], dim=1)