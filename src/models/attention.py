"""Attention module wrappers — provides a unified interface for attention blocks.

Supports: none, se, cbam, coordattn
"""
import torch.nn as nn


def create_attention(attn_type: str, channels: int, **kwargs) -> nn.Module:
    """Factory for attention modules.

    Args:
        attn_type: one of "none", "se", "cbam", "coordattn"
        channels: number of input channels
        **kwargs: passed to the attention module constructor
    """
    attn_type = attn_type.lower().strip()

    if attn_type in ("none", ""):
        return nn.Identity()

    if attn_type == "se":
        from src.models.blocks import SEBlock
        return SEBlock(channels, reduction=kwargs.get("reduction", 16))

    if attn_type == "cbam":
        from src.models.blocks import CBAM
        return CBAM(channels, reduction=kwargs.get("reduction", 16))

    if attn_type in ("coordattn", "coord_attn", "coordinate"):
        from src.models.blocks import CoordAttn
        return CoordAttn(channels, reduction=kwargs.get("reduction", 32))

    raise ValueError(f"Unknown attention type: {attn_type}. Choices: none, se, cbam, coordattn")
