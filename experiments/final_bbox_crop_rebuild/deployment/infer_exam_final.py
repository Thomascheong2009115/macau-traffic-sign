from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from src.class_filter import load_classes
from src.model import build_model, load_checkpoint
from src.preprocess_unified import crop_with_bbox, load_image_rgb, pil_to_normalized_tensor, preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def header_map(fieldnames: list[str]) -> dict[str, str]:
    return {field.strip().lower(): field for field in fieldnames}


def pick(headers: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name.lower() in headers:
            return headers[name.lower()]
    return None


def read_rows(input_csv: Path, image_dir: Path) -> list[dict[str, object]]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        headers = header_map(reader.fieldnames)
        img_col = pick(headers, "IMG", "img", "image_path", "filename")
        if img_col is None:
            raise ValueError("input CSV must contain IMG/img/image_path/filename")
        bbox_cols = [pick(headers, name) for name in ("x1", "y1", "x2", "y2")]
        rows: list[dict[str, object]] = []
        for row_number, row in enumerate(reader, start=2):
            img = str(row[img_col]).strip()
            image_path = Path(img)
            if not image_path.is_absolute():
                image_path = image_dir / img
            bbox = None
            warning = ""
            if all(bbox_cols):
                try:
                    bbox = tuple(float(row[col]) for col in bbox_cols if col is not None)
                except ValueError:
                    warning = f"invalid_bbox_row_{row_number}"
            else:
                warning = "missing_bbox_columns"
            rows.append({"IMG": img, "image_path": image_path, "bbox": bbox, "warning": warning})
    return rows


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final bbox-crop exam inference.")
    parser.add_argument("--input_csv", default="input/input.csv")
    parser.add_argument("--image_dir", default="input/images")
    parser.add_argument("--output_csv", default="output/submission.csv")
    parser.add_argument("--debug_csv", default="output/debug_predictions.csv")
    parser.add_argument("--debug_crop_dir", default="output/debug_crops")
    parser.add_argument("--config", default="experiments/final_bbox_crop_rebuild/deployment/exam_best_config.json")
    parser.add_argument("--classes", default="classes.txt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    classes = load_classes(resolve(args.classes))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = resolve(str(config["checkpoint"]))
    model_name = str(config.get("model_name", "resnet50"))
    image_size = int(config.get("image_size", 256))
    preprocessing_profile = str(config.get("preprocessing_profile", "clean_pad"))
    bbox_expansion_ratio = float(config.get("bbox_expansion_ratio", 0.08))

    model = build_model(model_name, len(classes), pretrained=False).to(device)
    payload = load_checkpoint(checkpoint, model, device)
    checkpoint_classes = payload.get("classes") if isinstance(payload, dict) else None
    if checkpoint_classes is not None and list(checkpoint_classes) != list(classes):
        raise ValueError("checkpoint classes do not match classes.txt")
    model.eval()

    rows = read_rows(resolve(args.input_csv), resolve(args.image_dir))
    output_csv = resolve(args.output_csv)
    debug_csv = resolve(args.debug_csv)
    debug_crop_dir = resolve(args.debug_crop_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_csv.parent.mkdir(parents=True, exist_ok=True)
    debug_crop_dir.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as out_handle, debug_csv.open("w", encoding="utf-8-sig", newline="") as dbg_handle:
        out_writer = csv.DictWriter(out_handle, fieldnames=["img", "class"])
        debug_writer = csv.DictWriter(
            dbg_handle,
            fieldnames=[
                "IMG",
                "pred",
                "confidence",
                "top3",
                "model_used",
                "router_used",
                "specialist_used",
                "used_bbox",
                "crop_path",
                "crop_warning",
                "warning",
            ],
        )
        out_writer.writeheader()
        debug_writer.writeheader()
        with torch.no_grad():
            for index, row in enumerate(rows, start=1):
                warning = str(row["warning"])
                used_bbox = False
                crop_path = ""
                try:
                    image = load_image_rgb(row["image_path"])
                    bbox = row["bbox"]
                    if bbox is not None:
                        crop, meta = crop_with_bbox(image, bbox, expansion_ratio=bbox_expansion_ratio)
                        used_bbox = not meta.used_full_image_fallback
                        if meta.used_full_image_fallback:
                            warning = ";".join(part for part in (warning, "bbox_fallback_full_image") if part)
                    else:
                        crop = image
                    crop_path_obj = debug_crop_dir / f"{index:05d}_{Path(str(row['IMG'])).stem}.jpg"
                    crop.save(crop_path_obj, quality=92)
                    crop_path = str(crop_path_obj)
                    processed, _ = preprocess_image(
                        image,
                        bbox=bbox,
                        image_size=image_size,
                        bbox_expansion_ratio=bbox_expansion_ratio,
                        preprocessing_profile=preprocessing_profile,
                    )
                    tensor = pil_to_normalized_tensor(processed).unsqueeze(0).to(device)
                    probs = torch.softmax(model(tensor)[0], dim=0).cpu()
                    top3_probs, top3_indices = probs.topk(k=min(3, len(classes)))
                    pred = classes[int(top3_indices[0])]
                    confidence = float(top3_probs[0])
                    top3 = " ".join(f"{classes[int(idx)]}:{float(prob):.4f}" for idx, prob in zip(top3_indices, top3_probs))
                except Exception as exc:
                    pred = classes[0]
                    confidence = 0.0
                    top3 = ""
                    warning = ";".join(part for part in (warning, f"error_{type(exc).__name__}") if part)
                out_writer.writerow({"img": row["IMG"], "class": pred})
                debug_writer.writerow(
                    {
                        "IMG": row["IMG"],
                        "pred": pred,
                        "confidence": f"{confidence:.6f}",
                        "top3": top3,
                        "model_used": model_name,
                        "router_used": "single",
                        "specialist_used": "",
                        "used_bbox": int(used_bbox),
                        "crop_path": crop_path,
                        "crop_warning": warning,
                        "warning": warning,
                    }
                )
    print(f"wrote_submission={output_csv}")
    print(f"wrote_debug={debug_csv}")


if __name__ == "__main__":
    main()
