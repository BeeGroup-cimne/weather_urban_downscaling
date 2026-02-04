import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class DownsrUNetMamba(nn.Module):
    def __init__(self, in_channels_dyn=9, in_channels_static=4, dim=32, seq_len=None):
        super().__init__()
        
        # --- CONFIG ---
        self.seq_len = seq_len # Optional, for shape checking
        
        # --- ENCODER (CNN) - Legacy Style (No BN, Pure ReLU) ---
        # Input channels = dynamic + static
        self.enc1 = self.conv_block(in_channels_dyn + in_channels_static, dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(dim, dim*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(dim*2, dim*4)
        self.pool3 = nn.MaxPool2d(2)
        
        # --- BOTTLENECK: MAMBA (SSM) ---
        self.d_model = dim*4
        # Mamba requires (Batch, Seq_Len, D_Model)
        # We will flatten Time * H * W into Seq_Len
        self.mamba1 = Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2) # Double block like TF
        
        # Norm layer for Mamba stability
        self.norm = nn.LayerNorm(self.d_model)

        # --- DECODER ---
        # Upsampling + Skip Connections
        # TF uses UpSampling2D -> Resizing (if needed) -> Concat -> Conv
        # We use bilinear interpolation
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = self.conv_block(dim*4 + dim*4, dim*2) # +skip
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self.conv_block(dim*2 + dim*2, dim)   # +skip
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = self.conv_block(dim + dim, 32)        # +skip
        
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
    def conv_block(self, in_ch, out_ch):
        """Legacy Block: Conv -> ReLU -> Conv -> ReLU (No BN)"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def time_distributed(self, layer, x):
        """
        Applies a 2D layer to a 5D tensor (Batch, Time, Chan, H, W)
        by merging Batch and Time.
        """
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)
        y = layer(x_reshaped)
        # Handle output shape changes (e.g. pooling or different channels)
        # y is (b*t, c_out, h_out, w_out)
        _, c_out, h_out, w_out = y.shape
        return y.view(b, t, c_out, h_out, w_out)

    def forward(self, x_dyn, x_static):
        # x_dyn: (Batch, Time, C_dyn, H, W)
        # x_static: (Batch, Time, C_st, H, W)
        
        # If static grid is HR and dynamic is LR, downsample static to match dynamic.
        if x_static.shape[-2:] != x_dyn.shape[-2:]:
            b, t, c, h, w = x_static.shape
            target_h, target_w = x_dyn.shape[-2], x_dyn.shape[-1]
            x_static = x_static.view(b * t, c, h, w)
            x_static = F.interpolate(
                x_static, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            x_static = x_static.view(b, t, c, target_h, target_w)

        # 1. Concatenate Dynamic + Static along Channels
        x = torch.cat([x_dyn, x_static], dim=2) # Dim 2 is channels in (B, T, C, H, W)
        
        # 2. Encoder (Time Distributed)
        # We reuse the utility to apply Conv blocks frame by frame
        c1 = self.time_distributed(self.enc1, x)
        p1 = self.time_distributed(self.pool1, c1)
        
        c2 = self.time_distributed(self.enc2, p1)
        p2 = self.time_distributed(self.pool2, c2)
        
        c3 = self.time_distributed(self.enc3, p2)
        p3 = self.time_distributed(self.pool3, c3)
        
        # 3. Mamba Bottleneck
        # TF Shape: (Batch, Time, H, W, C) -> Flattened to (Batch, T*H*W, C)
        # PyTorch p3 Shape: (Batch, Time, C, H, W)
        b, t, c, h, w = p3.shape
        
        # Permute to (Batch, Time, H, W, C) to match TF logic conceptually
        x_flat = p3.permute(0, 1, 3, 4, 2) # (B, T, H, W, C)
        # Flatten Time, H, W into a single sequence dimension
        x_flat = x_flat.contiguous().view(b, -1, c) # (B, T*H*W, C)
        
        # Apply Norm & Mamba
        x_mamba = self.norm(x_flat)
        x_mamba = self.mamba1(x_mamba)
        x_mamba = self.mamba2(x_mamba)
        
        # Unflatten back to (B, T, H, W, C)
        x_neck = x_mamba.view(b, t, h, w, c)
        # Permute back to PyTorch (B, T, C, H, W)
        x_neck = x_neck.permute(0, 1, 4, 2, 3)
        
        # 4. Decoder (Time Distributed)
        
        # Up 3
        u3 = self.time_distributed(self.up3, x_neck)
        # Resize/Crop check optional, but Upsample should match if shapes are powers of 2.
        # Concatenate with skip connection c3
        u3 = torch.cat([u3, c3], dim=2) 
        d3 = self.time_distributed(self.dec3, u3)
        
        # Up 2
        u2 = self.time_distributed(self.up2, d3)
        u2 = torch.cat([u2, c2], dim=2)
        d2 = self.time_distributed(self.dec2, u2)
        
        # Up 1
        u1 = self.time_distributed(self.up1, d2)
        u1 = torch.cat([u1, c1], dim=2)
        d1 = self.time_distributed(self.dec1, u1)
        
        # Final
        out = self.time_distributed(self.final, d1)
        
        return out
