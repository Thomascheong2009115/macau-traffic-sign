"""Model evaluation: compute metrics on validation set."""
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.augment import get_val_transform
from src.class_filter import load_classes
from src.data.datasets import TrafficSignDataset
from src.model import build_model, load_checkpoint
from src.training.metrics import compute_metrics, confusion_pairs, per_class_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: Optional[nn.Module] = None,
) -> Dict:
    """Evaluate model on validation set.

    Returns dict with keys: loss, top1, top3, top5, macro_f1_all,
    macro_f1_present, weighted_f1, targets, preds, topk_indices, paths.
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_topk: List[List[int]] = []
    paths: List[str] = [str(p) for p, _ in getattr(loader.dataset, "samples", [])]
    total_loss = 0.0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)

        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)

        probs = torch.softmax(logits, dim=1)
        topk = probs.topk(k=min(5, num_classes), dim=1).indices.cpu().tolist()
        preds = [t[0] for t in topk]
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds)
        all_topk.extend(topk)

    val_loss = total_loss / max(1, total) if criterion is not None else 0.0
    metrics = compute_metrics(all_targets, all_preds, all_topk, num_classes)

    return {
        "loss": val_loss,
        **metrics,
        "targets": all_targets,
        "preds": all_preds,
        "topk_indices": all_topk,
        "paths": paths,
    }


def write_wrong_cases(paths, targets, preds, topk_indices, classes, output_path):
    """Write wrong predictions to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label", "top3_labels", "top3_confs"])
        for path, t, p, topk in zip(paths, targets, preds, topk_indices):
            if t != p:
                top_labels = [classes[i] for i in topk[:3]]
                writer.writerow([path, classes[t], classes[p], " ".join(top_labels), ""])


def write_confusion_csv(targets, preds, classes, output_path):
    """Write full confusion matrix as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_pairs(targets, preds, classes, top_k=len(classes) * len(classes))
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", "pred_class", "count"])
        for true_cls, pred_cls, count in cm:
            writer.writerow([true_cls, pred_cls, count])


def write_per_class_csv(targets, preds, classes, output_path):
    """Write per-class metrics CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pc = per_class_metrics(targets, preds, classes)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for label in classes:
            m = pc.get(label, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
            writer.writerow([label, m["precision"], m["recall"], m["f1"], m["support"]])
