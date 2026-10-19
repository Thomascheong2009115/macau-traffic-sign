"""
Inference script for wavelet-augmented models.
Handles 4-channel input (RGB + wavelet) with correct preprocessing.
"""
import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src.augment import build_val_transform, IMAGENET_MEAN, IMAGENET_STD
from src.crop import crop_sign_image
from src.data_io import InputRow, read_input_csv
from src.model import build_model, load_checkpoint
from src.wavelet_features import WaveletFeatureTransform, append_wavelet_channels


def run_wavelet_inference(
    checkpoint_path: Path,
    input_csv: Path,
    image_dir: Path,
    output_submission: Path,
    output_debug: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint metadata
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", [])
    num_classes = len(classes)
    model_name = checkpoint.get("model_name", "resnet50")
    input_channels = checkpoint.get("input_channels", 4)
    wavelet_mode = checkpoint.get("wavelet_mode", "high")
    wavelet = checkpoint.get("wavelet", "db4")
    wavelet_level = checkpoint.get("wavelet_level", 1)
    ckpt_image_size = checkpoint.get("image_size", 224)
    ckpt_pad_to_square = bool(checkpoint.get("pad_to_square", False))

    print(f"Wavelet model: {model_name}, channels={input_channels}, mode={wavelet_mode}")
    print(f"  wavelet={wavelet}, level={wavelet_level}")
    print(f"  image_size={ckpt_image_size}, pad_to_square={ckpt_pad_to_square}")
    print(f"  classes={num_classes}")

    # Build model with correct input channels
    model = build_model(model_name, num_classes, pretrained=False, input_channels=input_channels).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    # Build transform with wavelet
    base_transform = build_val_transform(ckpt_image_size, pad_to_square=ckpt_pad_to_square)
    transform = WaveletFeatureTransform(
        base_transform=base_transform,
        mode=wavelet_mode,
        wavelet=wavelet,
        level=wavelet_level,
    )

    # Read input
    rows_data, _ = read_input_csv(input_csv, image_dir)
    print(f"Images: {len(rows_data)}")

    # Predict
    predictions: list[dict] = []
    for row in rows_data:
        result, crop = crop_sign_image(
            row.image_path, row.bbox, padding_ratio=0.10, clamp_padding=False
        )
        if not result.ok or crop is None:
            # Use original image as fallback
            crop = Image.open(row.image_path).convert("RGB")

        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)[0]
        probs = torch.softmax(logits, dim=0)

        # Top 3
        top3_indices = probs.topk(3)
        top3_labels = [classes[i] for i in top3_indices.indices.tolist()]
        top3_confs = top3_indices.values.tolist()

        predictions.append({
            "img": row.img,
            "main_top1_class": top3_labels[0] if len(top3_labels) > 0 else "",
            "main_top1_conf": f"{top3_confs[0]:.10g}" if top3_confs else "0",
            "main_top2_class": top3_labels[1] if len(top3_labels) > 1 else "",
            "main_top2_conf": f"{top3_confs[1]:.10g}" if len(top3_confs) > 1 else "0",
            "main_top3_class": top3_labels[2] if len(top3_labels) > 2 else "",
            "main_top3_conf": f"{top3_confs[2]:.10g}" if len(top3_confs) > 2 else "0",
        })

    # Write debug CSV (matching infer.py format)
    output_debug.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["img", "main_top1_class", "main_top1_conf", "main_top2_class",
                   "main_top2_conf", "main_top3_class", "main_top3_conf"]
    with output_debug.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    # Write submission
    output_submission.parent.mkdir(parents=True, exist_ok=True)
    with output_submission.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["img", "class"])
        for pred in predictions:
            writer.writerow([pred["img"], pred["main_top1_class"]])

    print(f"submission: {output_submission}")
    print(f"debug: {output_debug}")


def main() -> None:
    parser = argparse.ArgumentParser(description="wavelet model inference")
    parser.add_argument("--checkpoint", default="checkpoints/wavelet_ablation/high_best.pt")
    parser.add_argument("--input_csv", default="input/input.csv")
    parser.add_argument("--image_dir", default="input/images")
    parser.add_argument("--output", default="output/_wavelet_submission.csv")
    parser.add_argument("--debug_csv", default="output/_wavelet_debug.csv")
    args = parser.parse_args()

    run_wavelet_inference(
        checkpoint_path=Path(args.checkpoint),
        input_csv=Path(args.input_csv),
        image_dir=Path(args.image_dir),
        output_submission=Path(args.output),
        output_debug=Path(args.debug_csv),
    )


if __name__ == "__main__":
    main()
