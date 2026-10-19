"""Reusable building blocks for CNN architectures.

All blocks are designed to be togglable via CLI.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.channel_gate = nn.Sigmoid()
        # Spatial attention
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel
        avg = self.channel_attn(x)
        mx = self.channel_attn(torch.max(x, dim=0, keepdim=True)[0].expand_as(x))  # placeholder — use adaptive max
        # Simpler: use both avg and max pool
        avg_out = self.channel_attn(x)
        max_out = self.channel_attn(F.adaptive_max_pool2d(x, 1))
        ch = self.channel_gate(avg_out + max_out)
        x = x * ch
        # Spatial
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp = torch.cat([avg_sp, max_sp], dim=1)
        x = x * self.spatial_gate(self.spatial_attn(sp))
        return x


class CoordAttn(nn.Module):
    """Coordinate Attention (CVPR 2021).

    Decomposes channel attention into two 1D feature encodings.
    """
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)  # B, C, H, 1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # B, C, W → B, C, 1, W
        y = torch.cat([x_h, x_w], dim=3)  # B, C, H+W, 1
        y = self.act(self.bn(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # B, C, 1, W → B, C, W, 1
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class ResidualBlock(nn.Module):
    """Simple residual block with optional SE."""
    def __init__(self, channels: int, use_se: bool = False, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        return self.act(out + identity)
