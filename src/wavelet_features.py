from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.augment import IMAGENET_MEAN, IMAGENET_STD


WaveletMode = Literal["none", "low", "high", "fusion"]
WAVELET_MODES = ("none", "low", "high", "fusion")
SUPPORTED_WAVELET_LEVELS = (1, 2)
DEFAULT_WAVELET = "db4"
SUPPORTED_WAVELETS = ("haar", "db2", "db4")


def _validate_mode(mode: str) -> WaveletMode:
    if mode not in WAVELET_MODES:
        raise ValueError(f"wavelet_mode must be one of {WAVELET_MODES}; got {mode}")
    return mode  # type: ignore[return-value]


def _validate_level(level: int) -> int:
    if level not in SUPPORTED_WAVELET_LEVELS:
        raise ValueError("wavelet level must be 1 or 2; level=3 is intentionally not enabled")
    return level


def wavelet_input_channels(mode: str) -> int:
    mode = _validate_mode(mode)
    if mode == "none":
        return 3
    if mode in {"low", "high"}:
        return 4
    return 5


def _normalize_map(values: np.ndarray) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-8:
        return torch.zeros(array.shape, dtype=torch.float32)
    normalized = (array - min_value) / (max_value - min_value)
    return torch.from_numpy(normalized.astype(np.float32, copy=False))


def _resize_map(channel: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(
        channel.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _tensor_to_unit_rgb(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] < 3:
        raise ValueError("wavelet tensor input must have shape C,H,W with at least 3 RGB channels")
    rgb = tensor[:3].detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN, dtype=rgb.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=rgb.dtype).view(3, 1, 1)
    return (rgb * std + mean).clamp(0.0, 1.0)


def _image_to_gray_array(image: Image.Image | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        rgb = _tensor_to_unit_rgb(image)
        gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return gray.numpy().astype(np.float32, copy=False)

    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    return (0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]).astype(np.float32, copy=False)


def compute_wavelet_maps(
    image: Image.Image | torch.Tensor,
    wavelet: str = DEFAULT_WAVELET,
    level: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if wavelet not in SUPPORTED_WAVELETS:
        raise ValueError(f"wavelet must be one of {SUPPORTED_WAVELETS}; got {wavelet}")
    _validate_level(level)
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("PyWavelets is required for wavelet features. Install the PyWavelets package.") from exc

    gray = _image_to_gray_array(image)
    height, width = gray.shape
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level, mode="symmetric")

    low_freq = _resize_map(_normalize_map(coeffs[0]), height, width)
    high_energy = torch.zeros((height, width), dtype=torch.float32)
    for detail_coeffs in coeffs[1:]:
        c_h, c_v, c_d = detail_coeffs
        energy = np.sqrt(np.square(c_h) + np.square(c_v) + np.square(c_d))
        high_energy = high_energy + _resize_map(_normalize_map(energy), height, width)
    if len(coeffs) > 1:
        high_energy = high_energy / float(len(coeffs) - 1)
    return low_freq.clamp(0.0, 1.0), high_energy.clamp(0.0, 1.0)


def append_wavelet_channels(
    tensor: torch.Tensor,
    mode: str = "none",
    wavelet: str = DEFAULT_WAVELET,
    level: int = 1,
) -> torch.Tensor:
    mode = _validate_mode(mode)
    if mode == "none":
        return tensor
    low_freq, high_energy = compute_wavelet_maps(tensor, wavelet=wavelet, level=level)
    channels = [tensor]
    if mode in {"low", "fusion"}:
        channels.append(low_freq.unsqueeze(0).to(dtype=tensor.dtype))
    if mode in {"high", "fusion"}:
        channels.append(high_energy.unsqueeze(0).to(dtype=tensor.dtype))
    return torch.cat(channels, dim=0)


class WaveletFeatureTransform:
    def __init__(
        self,
        base_transform,
        mode: str = "none",
        wavelet: str = DEFAULT_WAVELET,
        level: int = 1,
    ) -> None:
        self.base_transform = base_transform
        self.mode = _validate_mode(mode)
        self.wavelet = wavelet
        self.level = _validate_level(level)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor = self.base_transform(image)
        return append_wavelet_channels(tensor, mode=self.mode, wavelet=self.wavelet, level=self.level)
