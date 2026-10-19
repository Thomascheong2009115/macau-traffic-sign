import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

from src.preprocess_unified import preprocess_image

try:
    import cv2
except Exception:  # pragma: no cover - cv2 缺失时保持旧流程可用
    cv2 = None


AUG_STRENGTH_CHOICES = ("none", "light", "medium", "strong")
NOISE_STRENGTH_CHOICES = ("none", "light", "medium")
PREPROCESSING_PROFILE_CHOICES = ("legacy", "clean_pad", "clahe_light", "sharpen_light", "denoise_light", "noise_robust_light", "jpeg_blur_light")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LightJpegCompression:
    """中文说明：轻微 JPEG 压缩用于模拟街景照片压缩噪声，不改变交通标志语义。"""

    def __init__(self, quality_range=(70, 95)) -> None:
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class LightBlur:
    """中文说明：小概率轻微模糊，避免把交通标志主体图案抹掉。"""

    def __init__(self, probability: float = 0.15, radius: float = 0.6) -> None:
        self.probability = probability
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.probability:
            return image.filter(ImageFilter.GaussianBlur(self.radius))
        return image


class AddGaussianNoise:
    def __init__(self, probability: float = 0.20, std: float = 4.0) -> None:
        self.probability = probability
        self.std = std

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        noise = np.random.normal(0.0, self.std, array.shape).astype(np.float32)
        noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")


class PadToSquare:
    def __init__(self, fill=(0, 0, 0)) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        size = max(width, height)
        if width == height:
            return image
        output = Image.new(image.mode, (size, size), self.fill)
        output.paste(image, ((size - width) // 2, (size - height) // 2))
        return output


class ReplicatePadToSquare:
    """使用边缘复制补边，保留长宽比，避免把小交通标志强行拉伸变形。"""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        size = max(width, height)
        if width == height:
            return image
        array = np.asarray(image.convert("RGB"))
        pad_left = (size - width) // 2
        pad_right = size - width - pad_left
        pad_top = (size - height) // 2
        pad_bottom = size - height - pad_top
        padded = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
        return Image.fromarray(padded.astype(np.uint8), mode="RGB")


class UnifiedBasePreprocess:
    def __init__(self, image_size: int, preprocessing_profile: str) -> None:
        self.image_size = image_size
        self.preprocessing_profile = preprocessing_profile

    def __call__(self, image: Image.Image) -> Image.Image:
        processed, _ = preprocess_image(
            image,
            image_size=self.image_size,
            preprocessing_profile=self.preprocessing_profile,
        )
        return processed


class LightCLAHE:
    """轻度 CLAHE，对低对比度或噪点照片做保守增强。"""

    def __init__(self, probability: float = 0.35, clip_limit: float = 1.5) -> None:
        self.probability = probability
        self.clip_limit = clip_limit

    def __call__(self, image: Image.Image) -> Image.Image:
        if cv2 is None or random.random() >= self.probability:
            return image
        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        merged = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced.astype(np.uint8), mode="RGB")


class LightSharpness:
    """轻度锐化，作为可开关鲁棒预处理的一部分。"""

    def __init__(self, probability: float = 0.25, factor: float = 1.25) -> None:
        self.probability = probability
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return ImageEnhance.Sharpness(image).enhance(self.factor)


def _resize_ops(image_size: int, pad_to_square: bool, preprocessing_profile: str = "legacy") -> list:
    if preprocessing_profile not in PREPROCESSING_PROFILE_CHOICES:
        raise ValueError(f"preprocessing_profile must be one of {PREPROCESSING_PROFILE_CHOICES}, got {preprocessing_profile}")
    ops = []
    if preprocessing_profile != "legacy":
        ops.append(UnifiedBasePreprocess(image_size, preprocessing_profile))
        return ops
    elif pad_to_square:
        ops.append(PadToSquare())
    ops.append(transforms.Resize((image_size, image_size)))
    return ops


def _normalize_tail():
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def _noise_ops(noise_strength: str) -> list:
    if noise_strength not in NOISE_STRENGTH_CHOICES:
        raise ValueError(f"noise_strength must be one of {NOISE_STRENGTH_CHOICES}, got {noise_strength}")
    if noise_strength == "none":
        return []
    if noise_strength == "light":
        return [
            AddGaussianNoise(probability=0.25, std=4.0),
            LightBlur(probability=0.10, radius=0.45),
            transforms.RandomApply([LightJpegCompression(quality_range=(78, 95))], p=0.25),
        ]
    return [
        AddGaussianNoise(probability=0.30, std=7.0),
        LightBlur(probability=0.15, radius=0.55),
        transforms.RandomApply([LightJpegCompression(quality_range=(72, 92))], p=0.35),
    ]


def _profile_ops(preprocessing_profile: str) -> list:
    """按配置追加轻量鲁棒预处理，默认不改变旧流程。"""
    if preprocessing_profile in {
        "legacy",
        "clean_pad",
        "clahe_light",
        "sharpen_light",
        "denoise_light",
        "noise_robust_light",
        "jpeg_blur_light",
    }:
        return []
    raise ValueError(f"unsupported preprocessing_profile: {preprocessing_profile}")


def get_train_transform(
    image_size: int = 224,
    aug_strength: str = "medium",
    pad_to_square: bool = False,
    noise_strength: str = "none",
    preprocessing_profile: str = "legacy",
    rand_augment_n: int = 0,
    rand_augment_m: int = 9,
):
    """中文说明：训练增强分 light/medium/strong 三档，禁止翻转和直角旋转。"""
    if aug_strength not in AUG_STRENGTH_CHOICES:
        raise ValueError(f"aug_strength must be one of {AUG_STRENGTH_CHOICES}, got {aug_strength}")
    if rand_augment_n < 0:
        raise ValueError(f"rand_augment_n must be >= 0; got {rand_augment_n}")
    if rand_augment_m < 0:
        raise ValueError(f"rand_augment_m must be >= 0; got {rand_augment_m}")

    if aug_strength == "none":
        ops = _resize_ops(image_size, pad_to_square, preprocessing_profile)
    elif aug_strength == "light":
        ops = [
            *_resize_ops(image_size, pad_to_square, preprocessing_profile),
            transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.35),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
        ]
    elif aug_strength == "medium":
        ops = [
            *_resize_ops(image_size, pad_to_square, preprocessing_profile),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.4),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.18, contrast=0.18),
            LightBlur(probability=0.15, radius=0.6),
            transforms.RandomApply([LightJpegCompression()], p=0.30),
        ]
    else:
        ops = [
            *_resize_ops(image_size, pad_to_square, preprocessing_profile),
            transforms.RandomApply([transforms.RandomRotation(degrees=12)], p=0.45),
            transforms.RandomPerspective(distortion_scale=0.16, p=0.30),
            transforms.RandomAffine(degrees=0, translate=(0.07, 0.07), scale=(0.88, 1.12)),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.10),
            LightBlur(probability=0.20, radius=0.7),
            transforms.RandomApply([LightJpegCompression(quality_range=(65, 92))], p=0.35),
        ]
    randaugment_ops = []
    if rand_augment_n > 0:
        randaugment_ops.append(transforms.RandAugment(num_ops=rand_augment_n, magnitude=rand_augment_m))
    return transforms.Compose(ops + randaugment_ops + _profile_ops(preprocessing_profile) + _noise_ops(noise_strength) + _normalize_tail())


def get_val_transform(image_size: int = 224, pad_to_square: bool = False, preprocessing_profile: str = "legacy"):
    """中文说明：验证集只能 Resize + Normalize，不能包含任何随机增强。"""
    return transforms.Compose(
        [
            *_resize_ops(image_size, pad_to_square, preprocessing_profile),
            *_normalize_tail(),
        ]
    )


def build_train_transform(
    image_size: int = 224,
    aug_strength: str = "medium",
    pad_to_square: bool = False,
    noise_strength: str = "none",
    preprocessing_profile: str = "legacy",
    rand_augment_n: int = 0,
    rand_augment_m: int = 9,
):
    """兼容旧调用；新代码优先使用 get_train_transform。"""
    return get_train_transform(
        image_size=image_size,
        aug_strength=aug_strength,
        pad_to_square=pad_to_square,
        noise_strength=noise_strength,
        preprocessing_profile=preprocessing_profile,
        rand_augment_n=rand_augment_n,
        rand_augment_m=rand_augment_m,
    )


def build_val_transform(image_size: int = 224, pad_to_square: bool = False, preprocessing_profile: str = "legacy"):
    """兼容旧调用；新代码优先使用 get_val_transform。"""
    return get_val_transform(image_size=image_size, pad_to_square=pad_to_square, preprocessing_profile=preprocessing_profile)


def build_offline_augment_transform(image_size: int = 224, aug_strength: str = "medium", pad_to_square: bool = False, noise_strength: str = "none"):
    """中文说明：可选离线增强只输出 PIL 图像，不做 Normalize，也不允许翻转。"""
    train_transform = get_train_transform(image_size=image_size, aug_strength=aug_strength, pad_to_square=pad_to_square, noise_strength=noise_strength)
    return transforms.Compose(
        [item for item in train_transform.transforms if not isinstance(item, (transforms.ToTensor, transforms.Normalize))]
    )
