"""Faithful baseline of CPC/UNet3D.py, with context-grid resize only for tests.

For the original 64^3 patches the layer widths, BN, transposed convolutions,
context concatenation and heads match the original source implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.double_conv=nn.Sequential(nn.Conv3d(ci,co,3,padding=1),nn.BatchNorm3d(co),nn.ReLU(inplace=True),
                                       nn.Conv3d(co,co,3,padding=1),nn.BatchNorm3d(co),nn.ReLU(inplace=True))
    def forward(self,x):return self.double_conv(x)


class Down(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.maxpool_conv=nn.Sequential(nn.MaxPool3d(2),DoubleConv(ci,co))
    def forward(self,x):return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self,ci,skip,co):
        super().__init__()
        self.up=nn.ConvTranspose3d(ci,ci,kernel_size=2,stride=2)
        self.conv=DoubleConv(ci+skip,co)
    def forward(self,x,skip):
        x=self.up(x)
        diff=[skip.shape[4]-x.shape[4],skip.shape[3]-x.shape[3],skip.shape[2]-x.shape[2]]
        if any(d != 0 for d in diff):
            x=F.pad(x, [diff[0]//2,diff[0]-diff[0]//2,
                        diff[1]//2,diff[1]-diff[1]//2,
                        diff[2]//2,diff[2]-diff[2]//2])
        return self.conv(torch.cat((skip,x),dim=1))


class OutConv(nn.Module):
    def __init__(self,ci,co):
        super().__init__();self.conv=nn.Conv3d(ci,co,1)
    def forward(self,x):return self.conv(x)


class LegacyUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc=DoubleConv(1,64)
        self.down1=Down(64,64)
        self.down2=Down(64,128)
        self.down3=Down(128,256)
        self.context_conv=nn.Linear(8,64*8*8*8)
        self.up1=Up(320,128,128)
        self.up2=Up(128,64,64)
        self.up3=Up(64,64,64)
        self.seg_outc=OutConv(64,2)
        self.flow_outc=OutConv(64,3)

    def forward(self,image,profile,features=None):
        if features is not None:raise ValueError('Legacy model has no pretrained feature input')
        x1=self.inc(image)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        context=self.context_conv(profile).view(-1,64,8,8,8)
        if x4.shape[2:]!=(8,8,8):
            # Test-only compatibility for inputs smaller than 64^3.
            context=F.interpolate(context,size=x4.shape[2:],mode='trilinear',align_corners=False)
        x4=torch.cat([x4,context],dim=1)
        x=self.up1(x4,x3)
        x=self.up2(x,x2)
        x=self.up3(x,x1)
        return self.seg_outc(x),F.normalize(self.flow_outc(x),p=2,dim=1)
