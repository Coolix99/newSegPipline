import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in1_channels, in2_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # Correctly calculate total input channels for convolution
            self.conv = DoubleConv(in1_channels+in2_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in1_channels,in1_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in1_channels+in2_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, context_size, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # Initialize the network layers
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.context_conv = nn.Linear(context_size, 64 * 8 * 8 * 8)  
        self.up1 = Up(320, 128, 128, bilinear)
        self.up2 = Up(128, 64, 64, bilinear) 
        self.up3 = Up(64, 64, 64, bilinear)
        self.seg_outc = OutConv(64, 2)  # Output layer for segmentation
        self.flow_outc = OutConv(64, 3)          # Output layer for flow field

    def forward(self, x, context_vector):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Inject context
        context_features = self.context_conv(context_vector).view(-1, 64, 8, 8, 8)  
        
        x4 = torch.cat([x4, context_features], dim=1)  
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output for segmentation
        seg_logits = self.seg_outc(x)
        seg_probs = torch.sigmoid(seg_logits)

        # Output for flow field
        flow_field = self.flow_outc(x)
        flow_field = torch.nn.functional.normalize(flow_field, p=2, dim=1)

        return seg_probs, flow_field
