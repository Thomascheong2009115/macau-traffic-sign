from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DIRECTION_SIGN_CLASSES = {str(value) for value in range(201, 220)}


@dataclass(frozen=True)
class CropMeta:
    input_bbox: tuple[float, float, float, float] | None
    clamped_bbox: tuple[int, int, int, int] | None
    expanded_bbox: tuple[int, int, int, int] | None
    used_full_image_fallback: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


class UnifiedTransform:
    def __init__(
        self,
        image_size: int = 256,
        preprocessing_profile: str = "clean_pad",
        bbox: Sequence[float] | None = None,
        bbox_expansion_ratio: float = 0.08,
        train: bool = False,
        label: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.image_size = image_size
        self.preprocessing_profile = preprocessing_profile
        self.bbox = tuple(float(value) for value in bbox) if bbox is not None else None
        self.bbox_expansion_ratio = bbox_expansion_ratio
        self.train = train
        self.label = str(label).strip() if label is not None else None
        self.normalize = normalize
        # Horizontal flips can reverse directional signs; keep them disabled.
        self.allows_horizontal_flip = False

    def __call__(self, image: Image.Image | np.ndarray) -> torch.Tensor:
        processed, _ = preprocess_image(
            image,
            bbox=self.bbox,
            image_size=self.image_size,
            bbox_expansion_ratio=self.bbox_expansion_ratio,
            preprocessing_profile=self.preprocessing_profile,
        )
        return pil_to_normalized_tensor(processed, normalize=self.normalize)


def pil_from_bgr_array(array: np.ndarray) -> Image.Image:
    arr = np.asarray(array)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"BGR array must have shape HxWx3, got {arr.shape}")
    rgb = arr[:, :, ::-1].astype(np.uint8, copy=True)
    return Image.fromarray(rgb, mode="RGB")


def ensure_rgb_pil(image: Image.Image | np.ndarray, *, input_color: str = "rgb") -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    color = input_color.lower()
    if color == "bgr":
        return pil_from_bgr_array(image)
    if color != "rgb":
        raise ValueError(f"input_color must be 'rgb' or 'bgr', got {input_color}")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"RGB array must have shape HxWx3, got {arr.shape}")
    return Image.fromarray(arr.astype(np.uint8, copy=True), mode="RGB")


def load_image_rgb(path: str | Path) -> Image.Image:
    path = Path(path)
    if cv2 is not None:
        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is not None:
            return pil_from_bgr_array(bgr)
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def clamp_bbox(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    left = int(max(0, min(width, min(x1, x2))))
    top = int(max(0, min(height, min(y1, y2))))
    right = int(max(0, min(width, max(x1, x2))))
    bottom = int(max(0, min(height, max(y1, y2))))
    return left, top, right, bottom


def expand_bbox(
    bbox: Sequence[int],
    width: int,
    height: int,
    expansion_ratio: float,
) -> tuple[int, int, int, int]:
    if expansion_ratio < 0:
        raise ValueError(f"expansion_ratio must be >= 0, got {expansion_ratio}")
    left, top, right, bottom = [int(value) for value in bbox]
    box_w = max(0, right - left)
    box_h = max(0, bottom - top)
    pad = int(round(max(box_w, box_h) * expansion_ratio))
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def crop_with_bbox(
    image: Image.Image | np.ndarray,
    bbox: Sequence[float] | None,
    *,
    expansion_ratio: float = 0.08,
    input_color: str = "rgb",
) -> tuple[Image.Image, CropMeta]:
    rgb = ensure_rgb_pil(image, input_color=input_color)
    if bbox is None:
        return rgb, CropMeta(None, None, None, used_full_image_fallback=True, notes=("missing_bbox",))

    width, height = rgb.size
    input_bbox = tuple(float(value) for value in bbox)
    clamped = clamp_bbox(input_bbox, width, height)
    left, top, right, bottom = clamped
    notes: list[str] = []
    if right <= left or bottom <= top:
        notes.append("empty_bbox")
        return rgb, CropMeta(input_bbox, clamped, clamped, used_full_image_fallback=True, notes=tuple(notes))

    expanded = expand_bbox(clamped, width, height, expansion_ratio)
    e_left, e_top, e_right, e_bottom = expanded
    if e_right <= e_left or e_bottom <= e_top:
        notes.append("empty_expanded_bbox")
        return rgb, CropMeta(input_bbox, clamped, expanded, used_full_image_fallback=True, notes=tuple(notes))
    return rgb.crop((e_left, e_top, e_right, e_bottom)), CropMeta(input_bbox, clamped, expanded, notes=tuple(notes))


def crop_by_bbox(
    image: Image.Image | np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    expansion_ratio: float = 0.08,
    input_color: str = "rgb",
) -> tuple[Image.Image, CropMeta]:
    return crop_with_bbox(
        image,
        (x1, y1, x2, y2),
        expansion_ratio=expansion_ratio,
        input_color=input_color,
    )


def pad_to_square(
    image: Image.Image,
    *,
    mode: str = "edge",
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    rgb = ensure_rgb_pil(image)
    width, height = rgb.size
    size = max(width, height)
    if width == height:
        return rgb.copy()

    pad_left = (size - width) // 2
    pad_right = size - width - pad_left
    pad_top = (size - height) // 2
    pad_bottom = size - height - pad_top

    if mode == "constant":
        out = Image.new("RGB", (size, size), fill)
        out.paste(rgb, (pad_left, pad_top))
        return out
    if mode != "edge":
        raise ValueError(f"pad mode must be 'edge' or 'constant', got {mode}")
    arr = np.asarray(rgb)
    padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    return Image.fromarray(padded.astype(np.uint8), mode="RGB")


def resize_preserving_aspect(
    image: Image.Image,
    image_size: int,
    *,
    pad_mode: str = "edge",
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    squared = pad_to_square(image, mode=pad_mode, fill=fill)
    return squared.resize((image_size, image_size), Image.BILINEAR)


def apply_clahe_light(image: Image.Image) -> Image.Image:
    rgb = ensure_rgb_pil(image)
    if cv2 is None:
        return rgb
    arr = np.asarray(rgb)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced.astype(np.uint8), mode="RGB")


def apply_sharpen_light(image: Image.Image) -> Image.Image:
    return ensure_rgb_pil(image).filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))


def apply_denoise_light(image: Image.Image) -> Image.Image:
    rgb = ensure_rgb_pil(image)
    if cv2 is None:
        return rgb.filter(ImageFilter.MedianFilter(size=3))
    arr = np.asarray(rgb)
    denoised = cv2.fastNlMeansDenoisingColored(arr, None, 3, 3, 7, 21)
    return Image.fromarray(denoised.astype(np.uint8), mode="RGB")


def apply_jpeg_blur_light(image: Image.Image) -> Image.Image:
    return ensure_rgb_pil(image).filter(ImageFilter.GaussianBlur(radius=0.35))


def apply_noise_robust_light(image: Image.Image) -> Image.Image:
    return apply_sharpen_light(apply_clahe_light(image))


def preprocess_image(
    image: Image.Image | np.ndarray,
    *,
    bbox: Sequence[float] | None = None,
    image_size: int = 256,
    bbox_expansion_ratio: float = 0.08,
    preprocessing_profile: str = "clean_pad",
    input_color: str = "rgb",
) -> tuple[Image.Image, CropMeta]:
    crop, meta = crop_with_bbox(
        image,
        bbox,
        expansion_ratio=bbox_expansion_ratio,
        input_color=input_color,
    )
    if preprocessing_profile == "legacy":
        processed = resize_preserving_aspect(crop, image_size, pad_mode="constant")
    elif preprocessing_profile == "clean_pad":
        processed = resize_preserving_aspect(crop, image_size, pad_mode="edge")
    elif preprocessing_profile == "clahe_light":
        processed = resize_preserving_aspect(apply_clahe_light(crop), image_size, pad_mode="edge")
    elif preprocessing_profile == "sharpen_light":
        processed = resize_preserving_aspect(apply_sharpen_light(crop), image_size, pad_mode="edge")
    elif preprocessing_profile == "denoise_light":
        processed = resize_preserving_aspect(apply_denoise_light(crop), image_size, pad_mode="edge")
    elif preprocessing_profile == "noise_robust_light":
        processed = resize_preserving_aspect(apply_noise_robust_light(crop), image_size, pad_mode="edge")
    elif preprocessing_profile == "jpeg_blur_light":
        processed = resize_preserving_aspect(apply_jpeg_blur_light(crop), image_size, pad_mode="edge")
    else:
        raise ValueError(f"unsupported preprocessing_profile: {preprocessing_profile}")
    return processed, meta


def pil_to_normalized_tensor(
    image: Image.Image,
    *,
    normalize: bool = True,
    mean: Iterable[float] = IMAGENET_MEAN,
    std: Iterable[float] = IMAGENET_STD,
) -> torch.Tensor:
    arr = np.asarray(ensure_rgb_pil(image), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
    if normalize:
        mean_t = torch.tensor(tuple(mean), dtype=tensor.dtype).view(3, 1, 1)
        std_t = torch.tensor(tuple(std), dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t
    return tensor


def build_eval_transform(
    image_size: int = 256,
    preprocessing_profile: str = "clean_pad",
    bbox: Sequence[float] | None = None,
    bbox_expansion_ratio: float = 0.08,
) -> UnifiedTransform:
    return UnifiedTransform(
        image_size=image_size,
        preprocessing_profile=preprocessing_profile,
        bbox=bbox,
        bbox_expansion_ratio=bbox_expansion_ratio,
        train=False,
    )


def build_train_transform(
    image_size: int = 256,
    preprocessing_profile: str = "clean_pad",
    label: str | None = None,
    bbox_expansion_ratio: float = 0.08,
) -> UnifiedTransform:
    return UnifiedTransform(
        image_size=image_size,
        preprocessing_profile=preprocessing_profile,
        bbox=None,
        bbox_expansion_ratio=bbox_expansion_ratio,
        train=True,
        label=label,
    )
