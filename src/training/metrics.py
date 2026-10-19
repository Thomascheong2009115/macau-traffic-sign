"""Classification metrics: accuracy, F1 scores, per-class breakdown."""
from collections import Counter
from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def compute_metrics(
    targets: List[int],
    preds: List[int],
    topk_preds: List[List[int]],
    num_classes: int,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        targets: list of ground truth class indices
        preds: list of predicted class indices (top-1)
        topk_preds: list of top-k predicted class indices per sample
        num_classes: total number of classes
    """
    if not targets:
        return {
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "macro_f1_all": 0.0,
            "macro_f1_present": 0.0,
            "weighted_f1": 0.0,
        }

    labels_all = list(range(num_classes))
    labels_present = sorted(set(targets))

    top1 = float(np.mean(np.array(targets) == np.array(preds)))

    k = max(len(r) for r in topk_preds) if topk_preds else 1
    topk_acc = 0.0
    if k >= 3:
        top3 = float(np.mean([t in tp[:3] for t, tp in zip(targets, topk_preds)]))
    else:
        top3 = top1
    top5 = float(np.mean([t in tp[:5] for t, tp in zip(targets, topk_preds)])) if k >= 5 else top3

    macro_f1_all = float(f1_score(targets, preds, labels=labels_all, average="macro", zero_division=0))
    macro_f1_present = float(f1_score(targets, preds, labels=labels_present, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, preds, labels=labels_present, average="weighted", zero_division=0))

    return {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "macro_f1_all": macro_f1_all,
        "macro_f1_present": macro_f1_present,
        "weighted_f1": weighted_f1,
    }


def compute_score(top1_acc: float, macro_f1_present: float, top3_acc: float, weighted_f1: float) -> float:
    """Composite score for model selection."""
    return 0.50 * top1_acc + 0.25 * macro_f1_present + 0.15 * top3_acc + 0.10 * weighted_f1


def per_class_metrics(targets: List[int], preds: List[int], classes: List[str]) -> Dict[str, Dict]:
    """Compute per-class precision, recall, F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(classes))), zero_division=0
    )
    result = {}
    for i, label in enumerate(classes):
        result[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return result


def confusion_pairs(targets: List[int], preds: List[int], classes: List[str], top_k: int = 20) -> List[tuple]:
    """Find most confused class pairs."""
    pairs = Counter((t, p) for t, p in zip(targets, preds) if t != p)
    return [(classes[t], classes[p], count) for (t, p), count in pairs.most_common(top_k)]


def train_val_gap(train_metrics: Dict[str, float], val_metrics: Dict[str, float], keys=("top1", "macro_f1_present")) -> Dict[str, float]:
    """Compute train/val gap for specified keys."""
    gaps = {}
    for k in keys:
        gaps[k] = train_metrics.get(k, 0.0) - val_metrics.get(k, 0.0)
    return gaps
