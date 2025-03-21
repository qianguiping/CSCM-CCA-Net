# -*- coding: utf-8 -*-
# @Time    : 2023/7/8 8:59 上午
# @File    : SRAUnet.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import cv2
class CCA(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        auto_padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=auto_padding, groups=dim, dilation=dilation)


        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv3 = nn.Conv2d(2*dim, dim, 1)
    def forward(self, x):
        u = x.clone()
        attn = self.conv(x)

        attna = self.conv0(attn)

        attn_0 = self.conv0_1(attna)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attna)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attna)
        attn_2 = self.conv2_2(attn_2)
        attn = (attn + attna + attn_0 + attn_1 + attn_2)/5
        attn = torch.cat([attn,u],dim=1)
        return self.conv3(attn)
class Attention(nn.Module):
    def __init__(self, d_model, dilation=3):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(d_model, d_model, 1)
        self.act = nn.GELU()
        self.ams = CCA(d_model, dilation=dilation)
        self.conv2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.ams(x)
        x = self.conv2(x)
        return x

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCC(nn.Module):
    """
    CCC Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential( Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential( Flatten(), nn.Linear(F_g, F_x))
        self.mlink = nn.Linear(F_x, F_x)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(F_x+F_g, F_x,kernel_size=(1,1),stride=(1,1))
    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool_x =self.mlp_x(avg_pool_x)
        channel_att_x = self.mlp_x(avg_pool_x).unsqueeze(2)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        avg_pool_g = self.mlp_g(avg_pool_g)
        channel_att_g = self.mlp_g(avg_pool_g).unsqueeze(2)
        channel_att = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att).unsqueeze(3).expand_as(x)
        x_after_channel = torch.cat([x, scale], dim=1)
        x_after_channel = self.conv(x_after_channel)
        #x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class SCC(nn.Module):  # spatial wise cross attention
    def __init__(self, F_g, F_x):
        super().__init__()
        self.hw = 64*512//F_g
        self.conv2dx = nn.Conv2d(F_x, 1,kernel_size=(1,1),stride=(1,1))
        self.mlpx  =  nn.Linear(self.hw, self.hw)
        self.conv2dg=  nn.Conv2d(F_g, 1,kernel_size=(1,1),stride=(1,1))
        self.mlpgx = nn.Linear(self.hw, self.hw)
        self.relu = nn.ReLU(inplace=True)
        self.retconv2d = nn.Conv2d(1, F_g,kernel_size=(1,1),stride=(1,1))
        self.lastconv2d = nn.Conv2d(2*F_g,F_g, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, g, x):
        # spatial-wise attention

        spx = torch.mean(x, dim=1, keepdim=True)  # spx = self.conv2dx(x)
        sp_att_x = self.mlpx(spx)
        sp_att_x = self.relu(sp_att_x)
        spg = torch.mean(g, dim=1, keepdim=True)  # spg = self.conv2dx(g)
        sp_att_g = self.mlpgx(spg)
        sp_att_g = self.relu(sp_att_g)

        sp_att_sum = (sp_att_x + sp_att_g) / 2.0
        sp_att_sum = torch.sigmoid(sp_att_sum)
        sp_att_sum.unsqueeze(1)
        sp_att_sum2 = self.retconv2d(sp_att_sum)
        x_after_sp = torch.cat([x,sp_att_sum2],dim =1)
        x_after_sp = self.lastconv2d(x_after_sp)

        out = self.relu(x_after_sp)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.inchannels =  in_channels//2
        self.outchannels = out_channels
        self.up = nn.Upsample(scale_factor=2)
        self.soatt = SCC(F_g=in_channels//2, F_x=in_channels//2 )
        self.coatt = CCC(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.Parallelconv2d = nn.Conv2d(3*in_channels//2,in_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x, skip_x):  #最好的串联
        up = self.up(x)
        skip_x_att = self.soatt(g=up, x=skip_x)
        skip_x_att = self.soatt(g=skip_x_att, x=skip_x )
        skip_x_att2 = self.coatt(g=skip_x_att, x=skip_x)
        skip_x_att2 = self.coatt(g=skip_x_att2, x=skip_x)
        x = torch.cat([skip_x_att,skip_x_att2, up], dim=1)  # dim 1 is the channel dimension
        x = self.Parallelconv2d(x)
        return self.nConvs(x)


class CSCM_CCA_Net(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)

        self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss
        self.ams = CCA(in_channels, dilation=3)
                             #self.amsattention = Attention(in_channels, dilation=3)
    
    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
                                #x = self.amsattention(x)
        x = self.ams(x)


        if self.n_classes ==1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1

        if self.vis: # visualize the attention maps
            return logits, att_weights
        else:
            return logits




