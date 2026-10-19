import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageOps

from src.data_io import InputRow, read_input_csv, write_missing_report


MIN_CROP_SIZE = 8


@dataclass(frozen=True)
class CropResult:
    ok: bool
    reason: str
    original_bbox: Tuple[float, float, float, float]
    clamped_bbox: Tuple[int, int, int, int]
    padded_bbox: Tuple[int, int, int, int]
    output_path: Path | None = None


def _clamp_padding_ratio(padding_ratio: float) -> float:
    return max(0.08, min(0.15, padding_ratio))


def _effective_padding_ratio(padding_ratio: float, clamp_padding: bool = True) -> float:
    if clamp_padding:
        return _clamp_padding_ratio(padding_ratio)
    return max(0.0, float(padding_ratio))


def _clamp_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = int(max(0, min(width, min(x1, x2))))
    top = int(max(0, min(height, min(y1, y2))))
    right = int(max(0, min(width, max(x1, x2))))
    bottom = int(max(0, min(height, max(y1, y2))))
    return left, top, right, bottom


def _pad_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
    clamp_padding: bool = True,
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    box_w = right - left
    box_h = bottom - top
    pad = int(round(max(box_w, box_h) * _effective_padding_ratio(padding_ratio, clamp_padding)))
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def crop_sign_image(
    image_path: str | Path,
    bbox: Tuple[float, float, float, float],
    padding_ratio: float = 0.10,
    clamp_padding: bool = True,
) -> tuple[CropResult, Image.Image | None]:
    image_path = Path(image_path)
    try:
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    except Exception as exc:
        return CropResult(False, f"image read failed: {exc}", bbox, (0, 0, 0, 0), (0, 0, 0, 0)), None

    width, height = image.size
    clamped = _clamp_bbox(bbox, width, height)
    left, top, right, bottom = clamped
    if right <= left or bottom <= top:
        return CropResult(False, "invalid bbox", bbox, clamped, clamped), None
    if right - left < MIN_CROP_SIZE or bottom - top < MIN_CROP_SIZE:
        return CropResult(False, "crop too small", bbox, clamped, clamped), None

    padded = _pad_bbox(clamped, width, height, padding_ratio, clamp_padding=clamp_padding)
    p_left, p_top, p_right, p_bottom = padded
    crop = image.crop((p_left, p_top, p_right, p_bottom))
    if crop.size[0] < MIN_CROP_SIZE or crop.size[1] < MIN_CROP_SIZE:
        return CropResult(False, "padded crop too small", bbox, clamped, padded), None
    return CropResult(True, "", bbox, clamped, padded), crop


def crop_sign(
    image_path: str | Path,
    bbox: Tuple[float, float, float, float],
    output_path: str | Path | None = None,
    debug_path: str | Path | None = None,
    padding_ratio: float = 0.10,
    clamp_padding: bool = True,
) -> CropResult:
    result, crop = crop_sign_image(image_path, bbox, padding_ratio, clamp_padding=clamp_padding)
    if not result.ok or crop is None:
        return result

    saved_path = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(saved_path)
    if debug_path is not None:
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug = image.copy()
        draw = ImageDraw.Draw(debug)
        draw.rectangle(result.clamped_bbox, outline="red", width=3)
        draw.rectangle(result.padded_bbox, outline="yellow", width=2)
        debug.save(debug_path)
    return CropResult(True, "", result.original_bbox, result.clamped_bbox, result.padded_bbox, saved_path)


def _safe_crop_name(img_name: str, row_number: int) -> str:
    stem = Path(img_name).stem.replace(" ", "_")
    return f"{row_number:05d}_{stem}.jpg"


def crop_rows(
    rows: Iterable[InputRow],
    crop_dir: str | Path,
    debug_dir: str | Path,
    bad_crop_csv: str | Path,
    padding_ratio: float = 0.10,
) -> list[tuple[InputRow, Path]]:
    crop_dir = Path(crop_dir)
    debug_dir = Path(debug_dir)
    bad_crop_csv = Path(bad_crop_csv)
    bad_crop_csv.parent.mkdir(parents=True, exist_ok=True)

    good_rows: list[tuple[InputRow, Path]] = []
    with bad_crop_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_number", "IMG", "reason", "X1", "Y1", "X2", "Y2"])
        for row in rows:
            crop_path = crop_dir / _safe_crop_name(row.img, row.row_number)
            debug_path = debug_dir / f"{crop_path.stem}.jpg"
            result = crop_sign(row.image_path, row.bbox, crop_path, debug_path, padding_ratio)
            if result.ok:
                good_rows.append((row, crop_path))
            else:
                writer.writerow([row.row_number, row.img, result.reason, row.x1, row.y1, row.x2, row.y2])
    return good_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="crop input images by bbox")
    parser.add_argument("--input_csv", default="input/input.csv")
    parser.add_argument("--image_dir", default="input/images")
    parser.add_argument("--crop_dir", default="data/real_crops")
    parser.add_argument("--debug_dir", default="output/debug_crops")
    parser.add_argument("--bad_crop_csv", default="output/bad_crop.csv")
    args = parser.parse_args()

    rows, missing = read_input_csv(args.input_csv, args.image_dir)
    write_missing_report(missing, "output/missing_images.csv")
    if missing:
        raise FileNotFoundError("missing images; see output/missing_images.csv")
    good = crop_rows(rows, args.crop_dir, args.debug_dir, args.bad_crop_csv)
    print(f"crop finished: {len(good)} images")


if __name__ == "__main__":
    main()
