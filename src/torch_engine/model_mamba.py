import torch
import torch.nn as nn
from mamba_ssm import Mamba

class DownsrUNetMamba(nn.Module):
    def __init__(self, in_channels_dyn=9, in_channels_static=4, dim=32):
        super().__init__()
        
        # --- ENCODER (CNN) ---
        self.enc1 = self.conv_block(in_channels_dyn + in_channels_static, dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(dim, dim*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(dim*2, dim*4)
        self.pool3 = nn.MaxPool2d(2)
        
        # --- BOTTLENECK: MAMBA (SSM) ---
        self.d_model = dim*4
        self.mamba = Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(self.d_model)

        # --- DECODER ---
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(dim*4 + dim*4, dim*2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(dim*2 + dim*2, dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(dim + dim, 32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x_dyn, x_static):
        x = torch.cat([x_dyn, x_static], dim=1)
        c1 = self.enc1(x); p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        
        b, c, h, w = p3.shape
        x_flat = p3.view(b, c, -1).permute(0, 2, 1)
        x_flat = self.norm(x_flat)
        x_mamba = self.mamba(x_flat)
        x_neck = x_mamba.permute(0, 2, 1).view(b, c, h, w)
        
        u3 = self.up3(x_neck); d3 = self.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, c1], dim=1))
        return self.final(d1)
