"""CNN model factory using timm pretrained backbones.

Supports: convnext_tiny, efficientnetv2_s, mobilenetv3_large, resnet50, resnet34, resnet18.
"""
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import timm
import torch.nn as nn

SUPPORTED_BACKBONES = [
    "convnext_tiny",
    "efficientnetv2_s",
    "mobilenetv3_large_100",
    "resnet50",
    "resnet34",
    "resnet18",
]


def _get_num_features(model: nn.Module, backbone_name: str) -> int:
    """Extract number of output features from a timm model."""
    # Try common attribute names
    for attr in ["num_features", "classifier.in_features", "head.in_features", "fc.in_features"]:
        parts = attr.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            if isinstance(obj, int):
                return obj
        except AttributeError:
            continue

    # Fallback: infer from classifier head
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        return model.classifier.in_features
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return model.head.in_features
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc.in_features

    # Last resort for convnext
    if hasattr(model, "head"):
        head = model.head
        if hasattr(head, "fc") and isinstance(head.fc, nn.Linear):
            return head.fc.in_features
        if hasattr(head, "in_features"):
            return head.in_features

    raise ValueError(f"Cannot determine num_features for backbone: {backbone_name}")


def _replace_head(model: nn.Module, backbone_name: str, num_classes: int, dropout: float = 0.0, activation: str = "relu"):
    """Replace classification head with custom one for our task."""
    act_map = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "mish": nn.Mish(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "none": nn.Identity(),
    }
    act_fn = act_map.get(activation.lower(), nn.ReLU(inplace=True))

    in_features = _get_num_features(model, backbone_name)

    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(in_features, num_classes))

    head = nn.Sequential(*layers) if layers else nn.Linear(in_features, num_classes)

    # Replace the appropriate head
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = head
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        model.classifier = head
    elif hasattr(model, "head"):
        if hasattr(model.head, "fc"):
            model.head.fc = nn.Linear(in_features, num_classes)
            if dropout > 0:
                # Insert dropout between GAP and fc for convnext
                model.head = nn.Sequential(model.head.norm_pre if hasattr(model.head, "norm_pre") else nn.Identity(),
                                          model.head.global_pool if hasattr(model.head, "global_pool") else nn.AdaptiveAvgPool2d(1),
                                          nn.Flatten(1),
                                          nn.Dropout(dropout),
                                          nn.Linear(in_features, num_classes))
        else:
            model.head = head
    elif hasattr(model, "fc"):
        model.fc = head
    else:
        raise ValueError(f"Don't know how to replace head for backbone: {backbone_name}")


def build_model(
    backbone: str = "resnet50",
    num_classes: int = 89,
    pretrained: bool = True,
    dropout: float = 0.0,
    activation: str = "relu",
    attention: str = "none",
    attention_reduction: int = 16,
    freeze_backbone_epochs: int = 0,
) -> nn.Module:
    """Build a timm-based CNN model with a configurable classification head.

    Args:
        backbone: timm model name
        num_classes: number of output classes
        pretrained: use ImageNet pretrained weights
        dropout: dropout rate before the final linear layer
        activation: activation function ('relu', 'leaky_relu', 'mish', 'gelu', 'silu', 'none')
        attention: attention module type ('none', 'se', 'cbam', 'coordattn')
        attention_reduction: reduction ratio for attention modules
        freeze_backbone_epochs: if > 0, backbone will be frozen initially (handled in train loop)
    """
    if backbone not in SUPPORTED_BACKBONES:
        try:
            # Check if it's a valid timm model at least
            timm.create_model(backbone, pretrained=False, num_classes=num_classes)
        except Exception:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {SUPPORTED_BACKBONES} or any timm model.")

    # Get timm model
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    # Replace head with our configurable one
    _replace_head(model, backbone, num_classes, dropout, activation)

    # Attention is not directly injected into timm backbones by default
    # because each backbone has different internal structure.
    # For custom attention, we'd need backbone-specific hooks.
    # We save attention config in model for reference.
    model._attention_type = attention
    model._backbone = backbone
    model._dropout = dropout

    return model
