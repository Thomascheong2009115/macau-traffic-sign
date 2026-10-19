import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.augment import build_val_transform
from src.class_filter import load_classes
from src.dataset import TrafficSignDataset
from src.model import build_model, load_checkpoint
from src.validate import assert_real_only_manifest, compute_classification_metrics


def _field_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header")
    return {name.strip().lower(): name for name in fieldnames}


def _get(row: dict[str, str], mapping: dict[str, str], candidates: Iterable[str], default: str = "") -> str:
    for candidate in candidates:
        key = candidate.lower()
        if key in mapping:
            return row.get(mapping[key], "").strip()
    if default != "":
        return default
    raise ValueError(f"missing required column: one of {', '.join(candidates)}")


def read_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            label = _get(row, mapping, ("label", "class", "Class", "CLASS", "true_class"))
            source = _get(row, mapping, ("source",), default="real") or "real"
            sample_path = _get(row, mapping, ("path", "crop_path", "image", "img"), default="")
            rows.append({"path": sample_path, "label": label, "source": source})
    return rows


def write_source_class_counts(rows: list[dict[str, str]], output_csv: str | Path) -> None:
    counts = Counter((row["source"], row["label"]) for row in rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "class", "count"])
        for (source, label), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1])):
            writer.writerow([source, label, count])


def write_class_counts(rows: list[dict[str, str]], output_csv: str | Path) -> None:
    counts = Counter(row["label"] for row in rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "count"])
        for label, count in sorted(counts.items()):
            writer.writerow([label, count])


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def evaluate_validation(
    checkpoint_path: str | Path,
    val_csv: str | Path,
    classes_path: str | Path,
    model_name: str,
    image_size: int,
    pad_to_square: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    classes = load_classes(classes_path)
    dataset = TrafficSignDataset(
        val_csv,
        classes_path,
        transform=build_val_transform(image_size=image_size, pad_to_square=bool(pad_to_square)),
        allowed_sources={"real"},
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(model_name, len(classes), pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    targets: list[int] = []
    preds: list[int] = []
    top3_indices: list[list[int]] = []
    predictions: list[dict[str, str | float]] = []
    paths = [str(path) for path, _ in dataset.samples]

    with torch.no_grad():
        offset = 0
        for images, batch_targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top3_probs, top3 = probs.topk(k=min(3, len(classes)), dim=1)
            batch_preds = top3[:, 0].cpu().tolist()
            batch_targets_list = batch_targets.cpu().tolist()
            targets.extend(batch_targets_list)
            preds.extend(batch_preds)
            top3_indices.extend(top3.cpu().tolist())
            for row_index, true_idx in enumerate(batch_targets_list):
                labels = top3[row_index].cpu().tolist()
                confs = top3_probs[row_index].cpu().tolist()
                predictions.append(
                    {
                        "path": paths[offset + row_index],
                        "true_class": classes[true_idx],
                        "pred_class": classes[labels[0]],
                        "top1_conf": float(confs[0]),
                        "top2_class": classes[labels[1]] if len(labels) > 1 else "",
                        "top2_conf": float(confs[1]) if len(confs) > 1 else 0.0,
                        "top3_class": classes[labels[2]] if len(labels) > 2 else "",
                        "top3_conf": float(confs[2]) if len(confs) > 2 else 0.0,
                        "correct": int(labels[0] == true_idx),
                    }
                )
            offset += len(batch_targets_list)

    metrics = compute_classification_metrics(targets, preds, top3_indices, len(classes))
    metrics.update({"total": len(targets), "correct": sum(1 for target, pred in zip(targets, preds) if target == pred)})
    return {
        "classes": classes,
        "targets": targets,
        "preds": preds,
        "top3_indices": top3_indices,
        "predictions": predictions,
        "metrics": metrics,
    }


def write_val_predictions(predictions: list[dict[str, str | float]], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "true_class",
        "pred_class",
        "top1_conf",
        "top2_class",
        "top2_conf",
        "top3_class",
        "top3_conf",
        "correct",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def write_per_class_metrics(classes: list[str], targets: list[int], preds: list[int], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "support", "correct", "recall"])
        for index, label in enumerate(classes):
            support = sum(1 for target in targets if target == index)
            correct = sum(1 for target, pred in zip(targets, preds) if target == index and pred == index)
            recall = correct / support if support else 0.0
            writer.writerow([label, support, correct, recall])


def write_confusion_pairs(classes: list[str], targets: list[int], preds: list[int], output_csv: str | Path) -> None:
    pairs = Counter((classes[target], classes[pred]) for target, pred in zip(targets, preds) if target != pred)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "pred_class", "count"])
        for (true_class, pred_class), count in pairs.most_common():
            writer.writerow([true_class, pred_class, count])


def read_best_log_row(log_path: str | Path) -> dict[str, str] | None:
    path = Path(log_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("score", "0") or 0))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train/val split and validation predictions without training.")
    parser.add_argument("--train_csv", default="data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv")
    parser.add_argument("--val_csv", default="splits/val_real.csv")
    parser.add_argument("--checkpoint", default="checkpoints/full_reinforced_student_aug_best.pt")
    parser.add_argument("--classes", default="classes.txt")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pad_to_square", type=int, choices=(0, 1), default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train_log", default="output/train_log.csv")
    parser.add_argument("--output_dir", default="output/validation_audit_latest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_real_only_manifest(args.val_csv)

    train_rows = read_manifest_rows(args.train_csv)
    val_rows = read_manifest_rows(args.val_csv)
    write_source_class_counts(train_rows, output_dir / "train_source_class_counts.csv")
    write_source_class_counts(val_rows, output_dir / "val_source_class_counts.csv")
    write_class_counts(train_rows, output_dir / "train_class_counts.csv")
    write_class_counts(val_rows, output_dir / "val_class_counts.csv")

    device = resolve_device(args.device)
    result = evaluate_validation(
        checkpoint_path=args.checkpoint,
        val_csv=args.val_csv,
        classes_path=args.classes,
        model_name=args.model,
        image_size=args.image_size,
        pad_to_square=args.pad_to_square,
        batch_size=args.batch_size,
        device=device,
    )

    write_val_predictions(result["predictions"], output_dir / "val_predictions.csv")
    write_per_class_metrics(result["classes"], result["targets"], result["preds"], output_dir / "val_per_class_metrics.csv")
    write_confusion_pairs(result["classes"], result["targets"], result["preds"], output_dir / "val_confusion_pairs.csv")

    summary = {
        "checkpoint": str(args.checkpoint),
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "device": str(device),
        "image_size": args.image_size,
        "pad_to_square": args.pad_to_square,
        "train_total": len(train_rows),
        "val_total": len(val_rows),
        **result["metrics"],
    }
    best_log_row = read_best_log_row(args.train_log)
    if best_log_row is not None:
        summary["best_log_epoch"] = int(best_log_row["epoch"])
        summary["best_log_train_top1_acc"] = float(best_log_row["train_top1_acc"])
        summary["best_log_val_top1_acc"] = float(best_log_row["val_top1_acc"])
        summary["best_log_gap_acc"] = float(best_log_row["gap_acc"])
        summary["best_log_overfit_flag"] = best_log_row["overfit_flag"]

    with (output_dir / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"val_correct={summary['correct']}/{summary['total']}")
    print(f"val_acc={summary['top1']:.4f}")
    print(f"val_top3={summary['top3']:.4f}")
    print(f"macro_f1_present={summary['macro_f1_present_classes']:.4f}")
    print(f"weighted_f1={summary['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
