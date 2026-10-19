"""Image loading and validation utilities."""
from pathlib import Path

from PIL import Image, ImageOps


def load_image_rgb(path: Path) -> Image.Image:
    """Load image as RGB, auto-rotate EXIF."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def validate_image(path: Path) -> tuple[bool, str]:
    """Check if image file is readable and valid.

    Returns (is_valid, reason)
    """
    if not path.exists():
        return False, "file_not_found"
    if not path.is_file():
        return False, "not_a_file"
    try:
        img = load_image_rgb(path)
        w, h = img.size
        if w < 10 or h < 10:
            return False, f"too_small_{w}x{h}"
        return True, "ok"
    except Exception as e:
        return False, f"load_error_{type(e).__name__}"
