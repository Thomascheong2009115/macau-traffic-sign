"""Simple inference: load checkpoint, crop sign from image, predict class.

No TTA, no expert models, no wavelet features — clean single-model inference.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image

from src.augment import get_val_transform
from src.class_filter import load_classes
from src.data_io import InputRow, read_input_csv, write_missing_report
from src.crop import crop_sign_image
from src.models.cnn_factory import build_model


def _load_checkpoint(path: Path, model, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {"model_state": checkpoint}


def run_inference(
    checkpoint_path: str | Path,
    input_csv: str | Path,
    image_dir: str | Path,
    output_csv: str | Path,
    debug_csv: Optional[str | Path] = None,
    classes_path: str | Path = "classes.txt",
    image_size: int = 224,
    padding_ratio: float = 0.10,
    batch_size: int = 1,
) -> None:
    """Run inference and produce submission.csv + optional debug_predictions.csv.

    Args:
        checkpoint_path: path to model checkpoint
        input_csv: path to input CSV (img,x1,y1,x2,y2)
        image_dir: directory containing input images
        output_csv: path for submission output (img,class)
        debug_csv: optional path for debug predictions
        classes_path: path to classes.txt
        image_size: input image size for model
        padding_ratio: padding around crop bbox
    """
    checkpoint_path = Path(checkpoint_path)
    input_csv = Path(input_csv)
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)

    classes = load_classes(classes_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint to infer backbone and model config
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint_data, dict):
        backbone = checkpoint_data.get("backbone", "resnet50")
        dropout = checkpoint_data.get("train_args", {}).get("dropout", 0.0) if isinstance(checkpoint_data.get("train_args"), dict) else 0.0
    else:
        backbone = "resnet50"
        dropout = 0.0
    num_classes = len(classes)

    model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False, dropout=dropout).to(device)
    _load_checkpoint(checkpoint_path, model, device)
    model.eval()

    transform = get_val_transform(image_size, pad_to_square=False)

    rows, missing = read_input_csv(input_csv, image_dir)
    if missing:
        write_missing_report(missing, Path(output_csv).parent / "missing_images.csv")
        print(f"WARNING: {len(missing)} missing images, see missing_images.csv")

    # Inference
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results = []
    debug_results = []

    for row in rows:
        result, crop = crop_sign_image(row.image_path, row.bbox, padding_ratio=padding_ratio)
        if not result.ok or crop is None:
            # Fallback: predict from full image
            crop = Image.open(row.image_path).convert("RGB")

        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            topk = probs.topk(k=min(3, num_classes), dim=1)
            top_indices = topk.indices[0].tolist()
            top_confs = topk.values[0].tolist()

        pred_class = classes[top_indices[0]]
        results.append({"img": row.img, "class": pred_class})

        debug_results.append({
            "img": row.img,
            "top1_class": pred_class,
            "top1_conf": f"{top_confs[0]:.4f}",
            "top2_class": classes[top_indices[1]] if len(top_indices) > 1 else "",
            "top2_conf": f"{top_confs[1]:.4f}" if len(top_confs) > 1 else "",
            "top3_class": classes[top_indices[2]] if len(top_indices) > 2 else "",
            "top3_conf": f"{top_confs[2]:.4f}" if len(top_confs) > 2 else "",
        })

    # Write submission
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img", "class"])
        writer.writeheader()
        writer.writerows(results)

    # Write debug
    if debug_csv:
        debug_csv = Path(debug_csv)
        debug_csv.parent.mkdir(parents=True, exist_ok=True)
        with debug_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["img", "top1_class", "top1_conf",
                                                    "top2_class", "top2_conf",
                                                    "top3_class", "top3_conf"])
            writer.writeheader()
            writer.writerows(debug_results)

    print(f"Inference complete: {len(results)} predictions")
    print(f"  submission: {output_csv}")
    if debug_csv:
        print(f"  debug: {debug_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run traffic sign inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--debug_csv", default=None)
    parser.add_argument("--classes_path", default="classes.txt")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--padding_ratio", type=float, default=0.10)
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        input_csv=args.input_csv,
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        debug_csv=args.debug_csv,
        classes_path=args.classes_path,
        image_size=args.image_size,
        padding_ratio=args.padding_ratio,
    )


if __name__ == "__main__":
    main()
