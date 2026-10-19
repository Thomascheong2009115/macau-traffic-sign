import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from src.augment import build_val_transform
from src.check_submission import check_submission
from src.class_filter import is_valid_class, load_classes
from src.crop import crop_sign, crop_sign_image
from src.data_io import InputRow, read_input_csv, write_missing_report
from src.model import build_model, load_checkpoint
from src.wavelet_features import DEFAULT_WAVELET, WAVELET_MODES, WaveletFeatureTransform, wavelet_input_channels


DEBUG_HEADER = [
    "img",
    "main_top1_class",
    "main_top1_conf",
    "main_top2_class",
    "main_top2_conf",
    "main_top3_class",
    "main_top3_conf",
    "expert_used",
    "expert_pred",
    "expert_conf",
    "final_class",
    "final_conf",
]
EXPERT_202_203_204_CLASSES = ["202", "203", "204"]
EXPERT_BIN_203_204_CLASSES = ["203", "204"]


def _load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


def parse_float_list(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            raise ValueError("tta_paddings must not be empty")
        return [float(item) for item in items]
    return [float(item) for item in value]


def _predict_logits(model, transform, image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor)[0].detach().cpu()


def _group_for_class(label: str) -> str:
    value = str(label).strip()
    if value.startswith("1"):
        return "1xx"
    if value.startswith("2"):
        return "2xx"
    if value.startswith("3"):
        return "3xx"
    return "unknown"


def detect_coarse_group(crop: Image.Image) -> str:
    rgb = np.array(crop.convert("RGB"))
    if rgb.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 60, 45]), np.array([12, 255, 255])) | cv2.inRange(
        hsv, np.array([168, 60, 45]), np.array([179, 255, 255])
    )
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 45]), np.array([135, 255, 255]))

    red_pixels = int(cv2.countNonZero(red_mask))
    blue_pixels = int(cv2.countNonZero(blue_mask))
    image_area = max(1, rgb.shape[0] * rgb.shape[1])
    if max(red_pixels, blue_pixels) < image_area * 0.02:
        return "unknown"

    color = "blue" if blue_pixels > red_pixels else "red"
    mask = blue_mask if color == "blue" else red_mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area < image_area * 0.01 or perimeter <= 0:
        return "unknown"

    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    if color == "blue" and circularity >= 0.45:
        return "2xx"
    if color == "red" and len(approx) <= 4 and circularity < 0.70:
        return "3xx"
    if color == "red" and circularity >= 0.35:
        return "1xx"
    return "unknown"


def rerank_top5_with_coarse_filter(
    logits: torch.Tensor,
    classes: list[str],
    coarse_group: str,
    bonus: float = 0.75,
) -> torch.Tensor:
    if coarse_group not in {"1xx", "2xx", "3xx"}:
        return logits
    adjusted = logits.clone()
    top_count = min(5, len(classes))
    top_indices = logits.topk(k=top_count).indices.tolist()
    for index in top_indices:
        if _group_for_class(classes[index]) == coarse_group:
            adjusted[index] += bonus
    return adjusted


def top_predictions_from_logits(logits: torch.Tensor, classes: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    probs = torch.softmax(logits, dim=0)
    topk = probs.topk(k=min(top_k, len(classes)))
    return [(classes[index], float(score)) for index, score in zip(topk.indices.tolist(), topk.values.tolist())]


def parse_generic_expert_classes(value: str) -> list[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if len(classes) < 2:
        raise ValueError("generic expert classes must contain at least two labels")
    if any(not (label.isdigit() and len(label) == 3) for label in classes):
        raise ValueError("generic expert classes must be comma-separated three-digit labels")
    if len(set(classes)) != len(classes):
        raise ValueError("generic expert classes must be unique")
    return classes


def should_use_202_203_204_expert(
    top_labels: list[str],
    focus_classes: set[str] | None = None,
    trigger_rank: int = 3,
    mode: str = "conservative",
) -> bool:
    focus = focus_classes or set(EXPERT_202_203_204_CLASSES)
    if not top_labels:
        return False
    if mode == "conservative":
        return top_labels[0] in focus
    if mode == "broad":
        return any(label in focus for label in top_labels[: max(1, trigger_rank)])
    raise ValueError("expert_mode must be conservative or broad")


def should_use_generic_expert(
    top_labels: list[str],
    current_final_pred: str,
    focus_classes: set[str],
    trigger_rank: int = 3,
    mode: str = "conservative",
) -> bool:
    if current_final_pred in focus_classes:
        return True
    if mode == "conservative":
        return bool(top_labels) and top_labels[0] in focus_classes
    if mode == "broad":
        return any(label in focus_classes for label in top_labels[: max(1, trigger_rank)])
    raise ValueError("generic_expert_mode must be conservative or broad")


def _safe_crop_name(img_name: str, row_number: int, padding: float) -> str:
    stem = Path(img_name).stem.replace(" ", "_")
    return f"{row_number:05d}_{stem}_pad{padding:.2f}.jpg"


def _build_tta_crops(
    row: InputRow,
    paddings: list[float],
    crop_dir: Path,
    debug_dir: Path,
    bad_writer: csv.writer,
    save_debug_crops: bool = True,
) -> list[tuple[float, Image.Image]]:
    crops: list[tuple[float, Image.Image]] = []
    for padding in paddings:
        crop_path = crop_dir / _safe_crop_name(row.img, row.row_number, padding)
        debug_path = debug_dir / f"{crop_path.stem}.jpg"
        if save_debug_crops:
            result = crop_sign(
                row.image_path,
                row.bbox,
                output_path=crop_path,
                debug_path=debug_path,
                padding_ratio=padding,
                clamp_padding=False,
            )
            if not result.ok:
                bad_writer.writerow([row.row_number, row.img, padding, result.reason, row.x1, row.y1, row.x2, row.y2])
                continue
            crop_result, crop = crop_sign_image(row.image_path, row.bbox, padding_ratio=padding, clamp_padding=False)
            if crop_result.ok and crop is not None:
                crops.append((padding, crop))
        else:
            result, crop = crop_sign_image(row.image_path, row.bbox, padding_ratio=padding, clamp_padding=False)
            if result.ok and crop is not None:
                crops.append((padding, crop))
            else:
                bad_writer.writerow([row.row_number, row.img, padding, result.reason, row.x1, row.y1, row.x2, row.y2])
    return crops


def predict_row(
    model,
    transform,
    row: InputRow,
    classes: list[str],
    device: torch.device,
    paddings: list[float],
    coarse_filter: bool,
    crop_dir: Path,
    debug_dir: Path,
    bad_writer: csv.writer,
) -> tuple[str, list[tuple[str, float]], str, list[float]]:
    crops = _build_tta_crops(row, paddings, crop_dir, debug_dir, bad_writer)
    if not crops:
        raise ValueError(f"failed to crop image: {row.img}")

    logits = torch.stack([_predict_logits(model, transform, crop, device) for _, crop in crops]).mean(dim=0)
    coarse_group = "unknown"
    if coarse_filter:
        coarse_votes = [detect_coarse_group(crop) for _, crop in crops]
        known_votes = [vote for vote in coarse_votes if vote != "unknown"]
        if known_votes:
            coarse_group = max(set(known_votes), key=known_votes.count)
            logits = rerank_top5_with_coarse_filter(logits, classes, coarse_group)

    top3 = top_predictions_from_logits(logits, classes, top_k=3)
    return top3[0][0], top3, coarse_group, [padding for padding, _ in crops]


def _apply_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    merged = dict(config)
    overrides = {
        "input_csv": args.input_csv,
        "image_dir": args.image_dir,
        "checkpoint_path": args.checkpoint,
        "output_csv": args.output,
        "debug_csv": args.debug_csv,
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    if args.output is not None:
        merged["submission_path"] = args.output
    if args.debug_csv is not None:
        merged["debug_predictions_path"] = args.debug_csv
    merged["tta"] = int(args.tta)
    merged["tta_paddings"] = args.tta_paddings
    merged["coarse_filter"] = int(args.coarse_filter)
    merged["use_expert_202_203_204"] = int(args.use_expert_202_203_204)
    merged["expert_202_203_204_checkpoint"] = args.expert_202_203_204_checkpoint
    merged["expert_trigger_rank"] = args.expert_trigger_rank
    merged["expert_min_main_conf"] = args.expert_min_main_conf
    merged["expert_mode"] = args.expert_mode
    merged["use_expert_tri_candidate"] = int(args.use_expert_tri_candidate)
    merged["expert_tri_candidate_checkpoint"] = args.expert_tri_candidate_checkpoint
    merged["use_expert_bin_203_204"] = int(args.use_expert_bin_203_204)
    merged["expert_bin_203_204_checkpoint"] = args.expert_bin_203_204_checkpoint
    merged["use_generic_expert"] = int(args.use_generic_expert)
    merged["generic_expert_name"] = args.generic_expert_name
    merged["generic_expert_checkpoint"] = args.generic_expert_checkpoint
    merged["generic_expert_classes"] = args.generic_expert_classes
    merged["generic_expert_trigger_rank"] = args.generic_expert_trigger_rank
    merged["generic_expert_mode"] = args.generic_expert_mode
    if args.image_size is not None:
        merged["image_size"] = args.image_size
    merged["pad_to_square"] = int(args.pad_to_square)
    merged["wavelet_mode"] = args.wavelet_mode
    merged["wavelet"] = args.wavelet
    merged["wavelet_level"] = args.wavelet_level
    return merged


def _normalize_config(config: Dict) -> Dict:
    normalized = dict(config)
    normalized["input_image_dir"] = normalized.get("input_image_dir", normalized.get("image_dir", "input/images"))
    normalized["submission_path"] = normalized.get(
        "submission_path",
        normalized.get("output_csv", "output/submission.csv"),
    )
    normalized["debug_predictions_path"] = normalized.get(
        "debug_predictions_path",
        normalized.get("debug_csv", "output/debug_predictions.csv"),
    )
    normalized.setdefault("crop_output_dir", "output/crops")
    normalized.setdefault("debug_crop_dir", "output/debug_crops")
    normalized.setdefault("bad_crop_csv", "output/bad_crop.csv")
    normalized.setdefault("image_size", 224)
    normalized.setdefault("pad_to_square", 0)
    normalized.setdefault("model_name", "resnet50")
    normalized.setdefault("wavelet_mode", "none")
    normalized.setdefault("wavelet", DEFAULT_WAVELET)
    normalized.setdefault("wavelet_level", 1)
    normalized.setdefault("device", "cpu")
    normalized.setdefault("use_expert_202_203_204", 0)
    normalized.setdefault("expert_202_203_204_checkpoint", "checkpoints/expert_202_203_204.pt")
    normalized.setdefault("expert_trigger_rank", 3)
    normalized.setdefault("expert_min_main_conf", 0.0)
    normalized.setdefault("expert_mode", "conservative")
    normalized.setdefault("use_expert_tri_candidate", 0)
    normalized.setdefault("expert_tri_candidate_checkpoint", "checkpoints/expert_tri_resnet18_202_203_204_candidate.pt")
    normalized.setdefault("use_expert_bin_203_204", 0)
    normalized.setdefault("expert_bin_203_204_checkpoint", "checkpoints/expert_bin_203_204_candidate.pt")
    normalized.setdefault("use_generic_expert", 0)
    normalized.setdefault("generic_expert_name", "")
    normalized.setdefault("generic_expert_checkpoint", "")
    normalized.setdefault("generic_expert_classes", "")
    normalized.setdefault("generic_expert_trigger_rank", 3)
    normalized.setdefault("generic_expert_mode", "conservative")
    return normalized


def _load_expert_model_from_checkpoint(
    config: Dict,
    device: torch.device,
    checkpoint_key: str,
    expected_classes: list[str],
):
    checkpoint_path = Path(config[checkpoint_key])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"expert checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint.get("classes", expected_classes) if isinstance(checkpoint, dict) else expected_classes
    if list(classes) != expected_classes:
        raise ValueError(f"expert checkpoint classes must be {expected_classes}")
    model_name = checkpoint.get("model_name", config.get("model_name", "resnet50")) if isinstance(checkpoint, dict) else config.get("model_name", "resnet50")
    model = build_model(model_name, len(expected_classes), pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()
    return model, list(classes)


def _load_expert_model(config: Dict, device: torch.device):
    return _load_expert_model_from_checkpoint(
        config,
        device,
        "expert_202_203_204_checkpoint",
        EXPERT_202_203_204_CLASSES,
    )


def _predict_expert(
    expert_model,
    transform,
    row: InputRow,
    expert_classes: list[str],
    device: torch.device,
    padding: float = 0.10,
) -> tuple[str, float]:
    result, crop = crop_sign_image(row.image_path, row.bbox, padding_ratio=padding, clamp_padding=False)
    if not result.ok or crop is None:
        raise ValueError(f"failed to crop image for expert: {row.img}")
    logits = _predict_logits(expert_model, transform, crop, device)
    return top_predictions_from_logits(logits, expert_classes, top_k=1)[0]


def run_inference(
    config_path: str | Path = "config.yaml",
    args: argparse.Namespace | None = None,
) -> None:
    config = _load_config(config_path)
    if args is not None:
        config = _apply_overrides(config, args)
    config = _normalize_config(config)

    classes = load_classes(config["classes_path"])
    rows, missing = read_input_csv(config["input_csv"], config["input_image_dir"])
    write_missing_report(missing, "output/missing_images.csv")
    if missing:
        raise FileNotFoundError("missing images; see output/missing_images.csv")

    paddings = parse_float_list(config.get("tta_paddings", "0.10")) if int(config.get("tta", 0)) else [0.10]
    crop_dir = Path(config["crop_output_dir"])
    debug_dir = Path(config["debug_crop_dir"])
    bad_crop_csv = Path(config["bad_crop_csv"])
    bad_crop_csv.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(config.get("device", "cpu")).lower())
    wavelet_mode = str(config.get("wavelet_mode", "none"))
    model = build_model(
        config.get("model_name", "resnet50"),
        len(classes),
        pretrained=False,
        input_channels=wavelet_input_channels(wavelet_mode),
    ).to(device)
    load_checkpoint(config["checkpoint_path"], model, device)
    model.eval()
    expert_transform = build_val_transform(config["image_size"], pad_to_square=bool(int(config.get("pad_to_square", 0))))
    transform = expert_transform
    if wavelet_mode != "none":
        transform = WaveletFeatureTransform(
            transform,
            mode=wavelet_mode,
            wavelet=str(config.get("wavelet", DEFAULT_WAVELET)),
            level=int(config.get("wavelet_level", 1)),
        )
    expert_model = None
    expert_classes: list[str] = EXPERT_202_203_204_CLASSES
    if int(config.get("use_expert_202_203_204", 0)):
        expert_model, expert_classes = _load_expert_model(config, device)
    tri_candidate_model = None
    tri_candidate_classes: list[str] = EXPERT_202_203_204_CLASSES
    if int(config.get("use_expert_tri_candidate", 0)):
        tri_candidate_model, tri_candidate_classes = _load_expert_model_from_checkpoint(
            config,
            device,
            "expert_tri_candidate_checkpoint",
            EXPERT_202_203_204_CLASSES,
        )
    bin_expert_model = None
    bin_expert_classes: list[str] = EXPERT_BIN_203_204_CLASSES
    if int(config.get("use_expert_bin_203_204", 0)):
        bin_expert_model, bin_expert_classes = _load_expert_model_from_checkpoint(
            config,
            device,
            "expert_bin_203_204_checkpoint",
            EXPERT_BIN_203_204_CLASSES,
        )
    generic_expert_model = None
    generic_expert_classes: list[str] = []
    generic_expert_name = str(config.get("generic_expert_name", "")).strip()
    if int(config.get("use_generic_expert", 0)):
        generic_expert_classes = parse_generic_expert_classes(str(config.get("generic_expert_classes", "")))
        generic_expert_model, generic_expert_classes = _load_expert_model_from_checkpoint(
            config,
            device,
            "generic_expert_checkpoint",
            generic_expert_classes,
        )

    submission_path = Path(config["submission_path"])
    debug_path = Path(config["debug_predictions_path"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    with bad_crop_csv.open("w", encoding="utf-8-sig", newline="") as bad_handle, submission_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as sub_handle, debug_path.open("w", encoding="utf-8-sig", newline="") as dbg_handle:
        bad_writer = csv.writer(bad_handle)
        sub_writer = csv.writer(sub_handle)
        dbg_writer = csv.writer(dbg_handle)
        bad_writer.writerow(["row_number", "img", "padding", "reason", "x1", "y1", "x2", "y2"])
        sub_writer.writerow(["img", "class"])
        dbg_writer.writerow(DEBUG_HEADER)

        for row in rows:
            main_pred, top3, coarse_group, used_paddings = predict_row(
                model,
                transform,
                row,
                classes,
                device,
                paddings,
                coarse_filter=bool(int(config.get("coarse_filter", 0))),
                crop_dir=crop_dir,
                debug_dir=debug_dir,
                bad_writer=bad_writer,
            )
            padded = top3 + [("", 0.0)] * (3 - len(top3))
            final_pred = main_pred
            final_conf = padded[0][1]
            expert_used = 0
            expert_pred = ""
            expert_conf = 0.0
            main_conf = padded[0][1]
            top_labels = [label for label, _ in padded[:3]]
            tri_candidate_pred = ""
            if expert_model is not None and main_conf >= float(config.get("expert_min_main_conf", 0.0)):
                if should_use_202_203_204_expert(
                    top_labels,
                    set(EXPERT_202_203_204_CLASSES),
                    int(config.get("expert_trigger_rank", 3)),
                    str(config.get("expert_mode", "conservative")),
                ):
                    expert_used = 1
                    expert_pred, expert_conf = _predict_expert(expert_model, expert_transform, row, expert_classes, device)
                    final_pred = expert_pred
                    final_conf = expert_conf
            if tri_candidate_model is not None and main_conf >= float(config.get("expert_min_main_conf", 0.0)):
                if should_use_202_203_204_expert(
                    top_labels,
                    set(EXPERT_202_203_204_CLASSES),
                    int(config.get("expert_trigger_rank", 3)),
                    str(config.get("expert_mode", "conservative")),
                ):
                    expert_used = 1
                    expert_pred, expert_conf = _predict_expert(tri_candidate_model, expert_transform, row, tri_candidate_classes, device)
                    tri_candidate_pred = expert_pred
                    final_pred = expert_pred
                    final_conf = expert_conf
            if bin_expert_model is not None and (
                main_pred in set(EXPERT_BIN_203_204_CLASSES) or tri_candidate_pred in set(EXPERT_BIN_203_204_CLASSES)
            ):
                expert_used = 1
                expert_pred, expert_conf = _predict_expert(bin_expert_model, expert_transform, row, bin_expert_classes, device)
                final_pred = expert_pred
                final_conf = expert_conf
            if generic_expert_model is not None:
                focus = set(generic_expert_classes)
                if should_use_generic_expert(
                    top_labels,
                    final_pred,
                    focus,
                    int(config.get("generic_expert_trigger_rank", 3)),
                    str(config.get("generic_expert_mode", "conservative")),
                ):
                    expert_used = 1
                    expert_pred, expert_conf = _predict_expert(generic_expert_model, expert_transform, row, generic_expert_classes, device)
                    final_pred = expert_pred
                    final_conf = expert_conf
                    if generic_expert_name:
                        expert_pred = f"{generic_expert_name}:{expert_pred}"
            if not is_valid_class(final_pred, classes):
                raise ValueError(f"invalid predicted class: {final_pred}")
            sub_writer.writerow([row.img, final_pred])
            dbg_writer.writerow(
                [
                    row.img,
                    padded[0][0],
                    padded[0][1],
                    padded[1][0],
                    padded[1][1],
                    padded[2][0],
                    padded[2][1],
                    expert_used,
                    expert_pred,
                    expert_conf,
                    final_pred,
                    final_conf,
                ]
            )

    check_submission(submission_path, config["input_csv"], config["classes_path"])
    print(f"submission.csv: {submission_path.resolve()}")
    print(f"debug_predictions.csv: {debug_path.resolve()}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run traffic sign inference")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input_csv")
    parser.add_argument("--image_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--debug_csv")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--pad_to_square", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tta", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tta_paddings", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--coarse_filter", type=int, choices=(0, 1), default=0)
    parser.add_argument("--use_expert_202_203_204", type=int, choices=(0, 1), default=0)
    parser.add_argument("--expert_202_203_204_checkpoint", default="checkpoints/expert_202_203_204.pt")
    parser.add_argument("--expert_trigger_rank", type=int, default=3)
    parser.add_argument("--expert_min_main_conf", type=float, default=0.0)
    parser.add_argument("--expert_mode", choices=("conservative", "broad"), default="conservative")
    parser.add_argument("--use_expert_tri_candidate", type=int, choices=(0, 1), default=0)
    parser.add_argument("--expert_tri_candidate_checkpoint", default="checkpoints/expert_tri_resnet18_202_203_204_candidate.pt")
    parser.add_argument("--use_expert_bin_203_204", type=int, choices=(0, 1), default=0)
    parser.add_argument("--expert_bin_203_204_checkpoint", default="checkpoints/expert_bin_203_204_candidate.pt")
    parser.add_argument("--use_generic_expert", type=int, choices=(0, 1), default=0)
    parser.add_argument("--generic_expert_name", default="")
    parser.add_argument("--generic_expert_checkpoint", default="")
    parser.add_argument("--generic_expert_classes", default="")
    parser.add_argument("--generic_expert_trigger_rank", type=int, default=3)
    parser.add_argument("--generic_expert_mode", choices=("conservative", "broad"), default="conservative")
    parser.add_argument("--wavelet_mode", choices=WAVELET_MODES, default="none")
    parser.add_argument("--wavelet", default=DEFAULT_WAVELET)
    parser.add_argument("--wavelet_level", type=int, choices=(1, 2), default=1)
    return parser


def main() -> None:
    parser = create_arg_parser()
    args = parser.parse_args()
    run_inference(args.config, args)


if __name__ == "__main__":
    main()
