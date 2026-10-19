import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.augment import build_val_transform
from src.class_filter import load_classes
from src.dataset import TrafficSignDataset
from src.model import build_model, load_checkpoint


FORBIDDEN_VAL_MARKERS = ("pdf", "synthetic", "diffusion", "optional_synthetic")


def assert_real_only_manifest(manifest_path: str | Path) -> None:
    """中文说明：验证集只能是真实照片，禁止 PDF、synthetic、diffusion 和增强图。"""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            sample_path = row.get("path", row.get("crop_path", "")).lower()
            source = row.get("source", "real").strip()
            if source != "real" or any(marker in sample_path for marker in FORBIDDEN_VAL_MARKERS):
                raise ValueError(f"validation row {row_number}: non-real sample is forbidden: {sample_path}")


def compute_classification_metrics(targets: List[int], preds: List[int], top3_indices: List[List[int]], num_classes: int) -> Dict[str, float]:
    """中文说明：同时计算全类别 F1、验证集中出现类别 F1、weighted F1 和 top-k accuracy。"""
    if not targets:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "macro_f1_all_classes": 0.0,
            "macro_f1_present_classes": 0.0,
            "weighted_f1": 0.0,
        }
    labels_all = list(range(num_classes))
    labels_present = sorted(set(targets))
    top1 = float(np.mean(np.array(targets) == np.array(preds)))
    top3_acc = float(np.mean([target in top3 for target, top3 in zip(targets, top3_indices)]))
    macro_f1_all = float(f1_score(targets, preds, labels=labels_all, average="macro", zero_division=0))
    macro_f1_present = float(f1_score(targets, preds, labels=labels_present, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, preds, labels=labels_present, average="weighted", zero_division=0))
    return {
        "top1": top1,
        "top3": top3_acc,
        "macro_f1_all_classes": macro_f1_all,
        "macro_f1_present_classes": macro_f1_present,
        "weighted_f1": weighted_f1,
    }


def evaluate_model(model, loader, device, classes: List[str], criterion=None) -> Dict:
    """中文说明：计算 loss、top1、top3、三种 F1，并保留错例分析需要的预测结果。"""
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_top3: List[List[int]] = []
    paths: List[str] = [str(path) for path, _ in getattr(loader.dataset, "samples", [])]
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            if criterion is not None:
                total_loss += criterion(logits, targets).item() * targets.size(0)
                total += targets.size(0)
            probs = torch.softmax(logits, dim=1)
            top3 = probs.topk(k=min(3, len(classes)), dim=1).indices.cpu().numpy().tolist()
            preds = probs.argmax(dim=1).cpu().numpy().tolist()
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds)
            all_top3.extend(top3)
    val_loss = float(total_loss / max(1, total)) if criterion is not None else 0.0
    metrics = compute_classification_metrics(all_targets, all_preds, all_top3, len(classes))
    return {
        "loss": val_loss,
        **metrics,
        "macro_f1": metrics["macro_f1_present_classes"],
        "targets": all_targets,
        "preds": all_preds,
        "top3_indices": all_top3,
        "paths": paths,
    }


def plot_confusion_matrix(targets: List[int], preds: List[int], classes: List[str], output_path: str | Path) -> None:
    matrix = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(18, 16))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    tick_positions = np.arange(len(classes))
    plt.xticks(tick_positions, classes, rotation=90, fontsize=6)
    plt.yticks(tick_positions, classes, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_wrong_cases(paths: List[str], targets: List[int], preds: List[int], top3_indices: List[List[int]], classes: List[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crop_path", "true_label", "pred_label", "top3"])
        for path, target, pred, top3 in zip(paths, targets, preds, top3_indices):
            if target != pred:
                writer.writerow([path, classes[target], classes[pred], " ".join(classes[index] for index in top3)])


def write_class_metrics(targets: List[int], preds: List[int], classes: List[str], output_path: str | Path) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=list(range(len(classes))),
        zero_division=0,
    )
    with Path(output_path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class", "precision", "recall", "f1", "support"])
        for index, label in enumerate(classes):
            writer.writerow([label, precision[index], recall[index], f1[index], support[index]])


def write_top_confusions(targets: List[int], preds: List[int], classes: List[str], output_path: str | Path, top_k: int = 20) -> None:
    pairs = Counter((target, pred) for target, pred in zip(targets, preds) if target != pred)
    with Path(output_path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "pred_class", "count"])
        for (target, pred), count in pairs.most_common(top_k):
            writer.writerow([classes[target], classes[pred], count])


def main() -> None:
    parser = argparse.ArgumentParser(description="验证真实照片验证集并输出分类指标")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = args.manifest or config["val_manifest"]
    checkpoint = args.checkpoint or config["checkpoint_path"]
    assert_real_only_manifest(manifest)
    classes = load_classes(config["classes_path"])
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "auto" else "cpu")
    dataset = TrafficSignDataset(manifest, config["classes_path"], build_val_transform(config["image_size"]))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    model = build_model(config["model_name"], len(classes), pretrained=False).to(device)
    load_checkpoint(checkpoint, model, device)
    metrics = evaluate_model(model, loader, device, classes)
    plot_confusion_matrix(metrics["targets"], metrics["preds"], classes, "output/confusion_matrix.png")
    write_wrong_cases(metrics["paths"], metrics["targets"], metrics["preds"], metrics["top3_indices"], classes, "output/wrong_cases.csv")
    write_class_metrics(metrics["targets"], metrics["preds"], classes, "output/class_metrics.csv")
    write_top_confusions(metrics["targets"], metrics["preds"], classes, "output/top_confusions.csv")
    print(
        f"top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} "
        f"macro_f1_all_classes={metrics['macro_f1_all_classes']:.4f} "
        f"macro_f1_present_classes={metrics['macro_f1_present_classes']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
