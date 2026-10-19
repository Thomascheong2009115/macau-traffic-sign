from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torchvision import models

try:
    import timm
except Exception:  # pragma: no cover - optional transformer dependency
    timm = None


TIMM_MODEL_ALIASES = {
    "vit_tiny_pretrained": "vit_tiny_patch16_224.augreg_in21k",
    "vit_tiny": "vit_tiny_patch16_224.augreg_in21k",
    "vit_s": "vit_small_patch16_224.augreg_in21k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "swin_tiny_pretrained": "swin_tiny_patch4_window7_224.ms_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "mambavision_t": "mambaout_femto.in1k",
    "mambavision_tiny": "mambaout_femto.in1k",
    "dinov3_s": "vit_small_patch16_dinov3.lvd1689m",
}


def _adapt_resnet_first_conv(model: nn.Module, input_channels: int) -> None:
    if input_channels == 3:
        return
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        channels_to_copy = min(3, input_channels)
        new_conv.weight[:, :channels_to_copy] = old_conv.weight[:, :channels_to_copy]
        if input_channels > 3:
            extra_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = extra_weight.repeat(1, input_channels - 3, 1, 1)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.conv1 = new_conv


def build_model(model_name: str = "resnet50", num_classes: int = 89, pretrained: bool = True, input_channels: int = 3) -> nn.Module:
    """默认 ResNet50 预训练模型，并把最后分类层替换成 89 类。"""
    model_name = model_name.lower()
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        _adapt_resnet_first_conv(model, input_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        _adapt_resnet_first_conv(model, input_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet101":
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        model = models.resnet101(weights=weights)
        _adapt_resnet_first_conv(model, input_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnext50_32x4d":
        if input_channels != 3:
            raise ValueError("wavelet input channels are only supported for resnet18 and resnet50")
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        model = models.resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        if input_channels != 3:
            raise ValueError("wavelet input channels are only supported for resnet18 and resnet50")
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "efficientnet_b2":
        if input_channels != 3:
            raise ValueError("wavelet input channels are only supported for resnet18 and resnet50")
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "efficientnet_v2_s":
        if input_channels != 3:
            raise ValueError("wavelet input channels are only supported for resnet18 and resnet50")
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_large":
        if input_channels != 3:
            raise ValueError("wavelet input channels are only supported for resnet18 and resnet50")
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "regnet_y_400mf":
        if input_channels != 3:
            raise ValueError("wavelet input channels are only supported for resnet18 and resnet50")
        weights = models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
        model = models.regnet_y_400mf(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "regnet_y_8gf":
        if input_channels != 3:
            raise ValueError("wavelet input channels are only supported for resnet18 and resnet50")
        weights = models.RegNet_Y_8GF_Weights.DEFAULT if pretrained else None
        model = models.regnet_y_8gf(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name in TIMM_MODEL_ALIASES:
        if input_channels != 3:
            raise ValueError("wavelet input channels are only supported for resnet18 and resnet50")
        if timm is None:
            raise ImportError("timm is required for ViT, ConvNeXt, Mamba, and DINOv3 aliases")
        return timm.create_model(TIMM_MODEL_ALIASES[model_name], pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"不支持的模型: {model_name}")


def save_checkpoint(path: str | Path, model: nn.Module, extra: Dict[str, Any] | None = None) -> None:
    """保存 checkpoint，包含模型权重和训练指标等附加信息。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, device: str | torch.device = "cpu") -> Dict[str, Any]:
    """加载 checkpoint；兼容只保存 state_dict 或保存完整字典两种格式。"""
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {"model_state": checkpoint}
