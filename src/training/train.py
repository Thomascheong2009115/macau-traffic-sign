"""Training loop for traffic sign CNN with full experiment tracking.

Supports: AdamW/Adam/SGD, cosine/plateau/onecycle schedulers,
mixed precision (AMP), early stopping, gradient clipping, MixUp/CutMix.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.class_filter import load_classes
from src.data.datasets import TrafficSignDataset
from src.data.transforms import get_train_transform, get_val_transform
from src.models.cnn_factory import build_model
from src.training.evaluate import evaluate_model, write_confusion_csv, write_per_class_csv, write_wrong_cases
from src.training.losses import create_loss
from src.training.metrics import compute_metrics, compute_score, train_val_gap
from src.utils.logging import save_json, timestamp
from src.utils.seed import set_seed

TRAIN_LOG_FIELDS = [
    "epoch", "train_loss", "train_top1", "train_macro_f1_present",
    "val_loss", "val_top1", "val_top3", "val_macro_f1_all",
    "val_macro_f1_present", "val_weighted_f1",
    "score", "best_score", "best_epoch", "lr", "gap_acc", "overfit_flag",
]


def _label_indices(dataset: TrafficSignDataset) -> List[int]:
    return [dataset.class_to_idx[l] for _, l in dataset.samples]


def build_weighted_sampler(labels: List[int], num_samples: int | None = None):
    """Inverse-frequency weighted sampler for class balance."""
    counts = Counter(labels)
    weights = [1.0 / max(1, counts[label]) for label in labels]
    n = num_samples or len(weights)
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=n, replacement=True)


def _mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def _cutmix_data(x, y, alpha=0.2):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, _, H, W = x.size()
    rx = np.random.randint(H)
    ry = np.random.randint(W)
    rh = int(H * np.sqrt(1 - lam))
    rw = int(W * np.sqrt(1 - lam))
    x1 = max(0, rx - rh // 2)
    x2 = min(H, rx + rh // 2)
    y1 = max(0, ry - rw // 2)
    y2 = min(W, ry + rw // 2)
    lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))

    index = torch.randperm(batch_size, device=x.device)
    x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model, loader, criterion, optimizer, device, use_amp, scaler,
    scheduler=None, scheduler_per_batch=False,
    use_mixup=False, use_cutmix=False, mixup_alpha=0.2,
    gradient_clip_norm=1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_mixup and np.random.random() < 0.5:
                images, targets_a, targets_b, lam = _mixup_data(images, targets, mixup_alpha)
                logits = model(images)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            elif use_cutmix and np.random.random() < 0.5:
                images, targets_a, targets_b, lam = _cutmix_data(images, targets, mixup_alpha)
                logits = model(images)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        if scheduler is not None and scheduler_per_batch:
            scheduler.step()

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(logits.detach().argmax(dim=1).cpu().tolist())

    top1 = sum(int(a == b) for a, b in zip(all_targets, all_preds)) / max(1, len(all_targets))
    return {
        "loss": total_loss / max(1, total),
        "top1": top1,
        "targets": all_targets,
        "preds": all_preds,
    }


def train_model(args: argparse.Namespace) -> Dict:
    """Full training pipeline. Saves all artifacts to args.output_dir."""
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(args.classes_path)
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.use_amp and device.type == "cuda"

    # Transforms
    train_transform = get_train_transform(
        args.image_size, args.aug_strength, bool(args.pad_to_square),
        use_random_erasing=bool(args.random_erasing),
    )
    val_transform = get_val_transform(args.image_size, bool(args.pad_to_square))

    # Datasets
    train_set = TrafficSignDataset(args.train_csv, args.classes_path, train_transform)
    val_set = TrafficSignDataset(args.val_csv, args.classes_path, val_transform)
    train_labels = _label_indices(train_set)

    # Sampler
    if args.balanced_sampler:
        sampler = build_weighted_sampler(train_labels)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_model(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
        activation=args.activation,
        attention=args.attention,
    ).to(device)

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)

    # Scheduler
    if args.scheduler == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                               steps_per_epoch=len(train_loader))
        scheduler_per_batch = True
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
        scheduler_per_batch = False
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler_per_batch = False

    # Loss
    class_weights = None
    if args.class_weight_loss:
        counts = Counter(train_labels)
        class_weights = torch.ones(num_classes, device=device)
        for i in range(num_classes):
            if counts.get(i, 0) > 0:
                class_weights[i] = len(train_labels) / (num_classes * counts[i])
    criterion = create_loss(args.loss, args.label_smoothing, class_weights)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training state
    best_score = -1.0
    best_epoch = 0
    best_metrics = {}
    epochs_no_improve = 0
    log_path = output_dir / "metrics.csv"

    # Clear log
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler,
            scheduler, scheduler_per_batch,
            use_mixup=bool(args.mixup), use_cutmix=bool(args.cutmix),
            mixup_alpha=args.mixup_alpha, gradient_clip_norm=args.gradient_clip_norm,
        )

        if args.scheduler == "plateau":
            # We'll step after eval
            pass
        elif not scheduler_per_batch:
            scheduler.step()

        # Evaluate
        eval_result = evaluate_model(model, val_loader, device, num_classes, criterion)

        # Compute composite score
        score = compute_score(
            eval_result["top1"], eval_result["macro_f1_present"],
            eval_result["top3"], eval_result["weighted_f1"],
        )

        gap = train_val_gap(
            {"top1": train_result["top1"]},
            {"top1": eval_result["top1"]},
            ["top1"],
        )
        overfit_flag = "YES" if gap.get("top1", 0) > 0.20 else "NO"

        # Scheduler step for plateau
        if args.scheduler == "plateau":
            scheduler.step(score)

        # Current LR
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            best_metrics = {
                "top1": eval_result["top1"],
                "top3": eval_result["top3"],
                "macro_f1_all": eval_result["macro_f1_all"],
                "macro_f1_present": eval_result["macro_f1_present"],
                "weighted_f1": eval_result["weighted_f1"],
            }
            epochs_no_improve = 0

            # Save best checkpoint
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "classes": classes,
                "backbone": args.backbone,
                "image_size": args.image_size,
                "best_score": score,
                "metrics": best_metrics,
                "train_args": vars(args),
            }, output_dir / "best.pth")

            # Write evaluation artifacts
            write_confusion_csv(eval_result["targets"], eval_result["preds"], classes,
                                output_dir / "confusion_matrix.csv")
            write_wrong_cases(eval_result["paths"], eval_result["targets"], eval_result["preds"],
                              eval_result["topk_indices"], classes, output_dir / "wrong_cases.csv")
            write_per_class_csv(eval_result["targets"], eval_result["preds"], classes,
                                output_dir / "per_class_accuracy.csv")
        else:
            epochs_no_improve += 1

        # Append log
        log_exists = log_path.exists()
        with log_path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS)
            if not log_exists:
                writer.writeheader()
            writer.writerow({
                "epoch": epoch, "train_loss": f"{train_result['loss']:.6f}",
                "train_top1": f"{train_result['top1']:.6f}",
                "train_macro_f1_present": f"{eval_result['macro_f1_present']:.6f}",
                "val_loss": f"{eval_result['loss']:.6f}",
                "val_top1": f"{eval_result['top1']:.6f}",
                "val_top3": f"{eval_result['top3']:.6f}",
                "val_macro_f1_all": f"{eval_result['macro_f1_all']:.6f}",
                "val_macro_f1_present": f"{eval_result['macro_f1_present']:.6f}",
                "val_weighted_f1": f"{eval_result['weighted_f1']:.6f}",
                "score": f"{score:.6f}", "best_score": f"{best_score:.6f}",
                "best_epoch": best_epoch, "lr": f"{current_lr:.8f}",
                "gap_acc": f"{gap.get('top1', 0):.6f}",
                "overfit_flag": overfit_flag,
            })

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_result['loss']:.4f} | train_acc={train_result['top1']:.4f} | "
              f"val_loss={eval_result['loss']:.4f} | val_acc={eval_result['top1']:.4f} | "
              f"val_f1={eval_result['macro_f1_present']:.4f} | "
              f"score={score:.4f} | best={best_score:.4f}@{best_epoch} | {overfit_flag}")

        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
            break

    # Save last checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "classes": classes,
    }, output_dir / "last.pth")

    # Save config
    save_json(vars(args), output_dir / "config.json")

    # Save metrics summary
    save_json({
        "best_epoch": best_epoch,
        "best_score": best_score,
        **{f"best_{k}": v for k, v in best_metrics.items()},
    }, output_dir / "metrics.json")

    print(f"\nTraining complete: best_score={best_score:.4f} @ epoch {best_epoch}")
    print(f"  best_top1={best_metrics.get('top1', 0):.4f}")
    print(f"  best_macro_f1_present={best_metrics.get('macro_f1_present', 0):.4f}")
    return {"best_score": best_score, "best_epoch": best_epoch, **best_metrics}


def create_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train traffic sign CNN")
    # Data
    p.add_argument("--train_csv", default="outputs/manifests/train.csv")
    p.add_argument("--val_csv", default="outputs/manifests/val.csv")
    p.add_argument("--classes_path", default="classes.txt")
    # Model
    p.add_argument("--backbone", default="resnet50",
                   choices=["convnext_tiny", "efficientnetv2_s", "mobilenetv3_large_100",
                            "resnet50", "resnet34", "resnet18"])
    p.add_argument("--pretrained", type=int, default=1, choices=[0, 1])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--activation", default="relu",
                   choices=["relu", "leaky_relu", "mish", "gelu", "silu"])
    p.add_argument("--attention", default="none",
                   choices=["none", "se", "cbam", "coordattn"])
    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau", "onecycle"])
    p.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "focal"])
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--balanced_sampler", type=int, default=1, choices=[0, 1])
    p.add_argument("--class_weight_loss", type=int, default=0, choices=[0, 1])
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    # Augmentation
    p.add_argument("--aug_strength", default="medium", choices=["none", "light", "medium", "strong"])
    p.add_argument("--mixup", type=int, default=0, choices=[0, 1])
    p.add_argument("--cutmix", type=int, default=0, choices=[0, 1])
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--random_erasing", type=int, default=0, choices=[0, 1])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--pad_to_square", type=int, default=0, choices=[0, 1])
    # Technical
    p.add_argument("--use_amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs/experiments/test")
    return p


def main():
    args = create_arg_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
