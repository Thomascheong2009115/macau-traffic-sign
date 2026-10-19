"""Training and validation transforms for traffic sign classification.

No horizontal/vertical flips — traffic signs are orientation-sensitive.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUG_STRENGTHS = ("none", "light", "medium", "strong")


class LightJpegCompression:
    def __init__(self, quality_range=(70, 95)):
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class LightBlur:
    def __init__(self, probability=0.15, radius=0.6):
        self.probability = probability
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.probability:
            return image.filter(ImageFilter.GaussianBlur(self.radius))
        return image


class PadToSquare:
    def __init__(self, fill=(0, 0, 0)):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        size = max(w, h)
        if w == h:
            return image
        out = Image.new(image.mode, (size, size), self.fill)
        out.paste(image, ((size - w) // 2, (size - h) // 2))
        return out


def _resize_ops(image_size: int, pad_to_square: bool = False) -> list:
    ops = []
    if pad_to_square:
        ops.append(PadToSquare())
    ops.append(transforms.Resize((image_size, image_size)))
    return ops


def _normalize_tail() -> list:
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def get_train_transform(
    image_size: int = 224,
    aug_strength: str = "medium",
    pad_to_square: bool = False,
    use_mixup: bool = False,
    use_cutmix: bool = False,
    use_random_erasing: bool = False,
    random_erasing_p: float = 0.3,
):
    """Build training augmentation transform.

    Note: MixUp and CutMix are applied in the training loop, not here.
    RandomErasing is a transform applied to the tensor.
    """
    if aug_strength not in AUG_STRENGTHS:
        raise ValueError(f"aug_strength must be one of {AUG_STRENGTHS}")

    if aug_strength == "none":
        ops = _resize_ops(image_size, pad_to_square)
    elif aug_strength == "light":
        ops = [
            *_resize_ops(image_size, pad_to_square),
            transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.35),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
        ]
    elif aug_strength == "medium":
        ops = [
            *_resize_ops(image_size, pad_to_square),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.4),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.18, contrast=0.18),
            LightBlur(probability=0.15, radius=0.6),
            transforms.RandomApply([LightJpegCompression()], p=0.30),
        ]
    else:  # strong
        ops = [
            *_resize_ops(image_size, pad_to_square),
            transforms.RandomApply([transforms.RandomRotation(degrees=12)], p=0.45),
            transforms.RandomPerspective(distortion_scale=0.16, p=0.30),
            transforms.RandomAffine(degrees=0, translate=(0.07, 0.07), scale=(0.88, 1.12)),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.10),
            LightBlur(probability=0.20, radius=0.7),
            transforms.RandomApply([LightJpegCompression(quality_range=(65, 92))], p=0.35),
        ]

    tail_ops = _normalize_tail()
    if use_random_erasing:
        tail_ops.insert(-1, transforms.RandomErasing(p=random_erasing_p))

    return transforms.Compose(ops + tail_ops)


def get_val_transform(
    image_size: int = 224,
    pad_to_square: bool = False,
):
    """Validation transform: resize + normalize only."""
    return transforms.Compose([
        *_resize_ops(image_size, pad_to_square),
        *_normalize_tail(),
    ])
