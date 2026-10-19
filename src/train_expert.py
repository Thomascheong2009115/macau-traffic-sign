import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.augment import get_train_transform, get_val_transform
from src.dataset import resolve_image_path
from src.model import build_model
from src.train import make_optimizer
from src.validate import compute_classification_metrics


LOG_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "macro_f1", "per_class_acc", "lr", "best_epoch"]
ALLOWED_TRAIN_SOURCES = {"real", "augmented_real", "real_student_images", "augmented_student"}


def parse_expert_classes(value: str) -> tuple[list[str], dict[str, int]]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if len(classes) < 2:
        raise ValueError("expert classes must contain at least two labels")
    if any(not (label.isdigit() and len(label) == 3) for label in classes):
        raise ValueError("expert classes must be comma-separated three-digit labels")
    if len(set(classes)) != len(classes):
        raise ValueError("expert classes must be unique")
    return classes, {label: index for index, label in enumerate(classes)}


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        classes: list[str],
        transform,
        project_root: str | Path | None = None,
        allowed_sources: set[str] | None = None,
        filter_to_classes: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.classes = classes
        self.class_to_idx = {label: index for index, label in enumerate(classes)}
        self.transform = transform
        self.allowed_sources = allowed_sources
        self.samples: list[tuple[Path, str]] = []
        self.skipped_rows = 0
        self._load(filter_to_classes)

    def _load(self, filter_to_classes: bool) -> None:
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["path", "label", "source"]:
                raise ValueError("manifest columns must be path,label,source")
            for row_number, row in enumerate(reader, start=2):
                label = row["label"].strip()
                if label not in self.class_to_idx:
                    if filter_to_classes:
                        self.skipped_rows += 1
                        continue
                    raise ValueError(f"row {row_number}: label is not in expert classes: {label}")
                source = row["source"].strip()
                if self.allowed_sources is not None and source not in self.allowed_sources:
                    raise ValueError(f"row {row_number}: source is not allowed: {source}")
                image_path = resolve_image_path(row["path"], self.project_root, self.manifest_dir)
                self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[label]


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


def _per_class_acc(targets: list[int], preds: list[int], classes: list[str]) -> str:
    totals = Counter(targets)
    correct = Counter(target for target, pred in zip(targets, preds) if target == pred)
    parts = []
    for index, label in enumerate(classes):
        total = totals[index]
        value = correct[index] / total if total else 0.0
        parts.append(f"{label}:{value:.4f}")
    return " ".join(parts)


def _evaluate(model, loader, criterion, device, classes: list[str]) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total = 0
    targets_all: list[int] = []
    preds_all: list[int] = []
    top3_all: list[list[int]] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            total += targets.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().tolist()
            top3 = probs.topk(k=min(3, len(classes)), dim=1).indices.cpu().tolist()
            targets_cpu = targets.cpu().tolist()
            targets_all.extend(targets_cpu)
            preds_all.extend(preds)
            top3_all.extend(top3)
    metrics = compute_classification_metrics(targets_all, preds_all, top3_all, len(classes))
    return {
        "loss": total_loss / max(1, total),
        "acc": metrics["top1"],
        "macro_f1": metrics["macro_f1_present_classes"],
        "per_class_acc": _per_class_acc(targets_all, preds_all, classes),
    }


def _train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool, scaler) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, targets in tqdm(loader, desc="expert_train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
    return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}


def _make_balanced_sampler(dataset: ExpertDataset) -> WeightedRandomSampler:
    counts = Counter(label for _, label in dataset.samples)
    weights = [1.0 / counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def run_training(args: argparse.Namespace) -> dict[str, object]:
    classes, _ = parse_expert_classes(args.classes)
    device = _resolve_device(args.device)
    use_amp = bool(args.use_amp and device.type == "cuda" and torch.cuda.is_available())
    train_set = ExpertDataset(
        args.train_csv,
        classes,
        get_train_transform(args.image_size, args.aug_strength),
        allowed_sources=ALLOWED_TRAIN_SOURCES,
    )
    val_set = ExpertDataset(
        args.val_csv,
        classes,
        get_val_transform(args.image_size),
        allowed_sources={"real"},
        filter_to_classes=True,
    )
    if len(val_set) == 0:
        print("WARNING: validation set has no 202/203/204 samples; validation metrics will be zero", file=sys.stderr)

    train_sampler = _make_balanced_sampler(train_set) if int(args.balanced_sampler) else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = build_model(args.model, len(classes), pretrained=args.pretrained).to(device)
    optimizer = make_optimizer(model, args.lr, args.head_lr, args.weight_decay, "adamw")
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_path = Path(args.checkpoint_path)
    log_path = Path(args.log_path)
    metrics_path = Path(args.metrics_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    best_val_acc = -1.0
    best_macro_f1 = 0.0
    best_epoch = 0
    stale_epochs = 0
    val_loss_history: list[float] = []
    val_acc_history: list[float] = []
    with log_path.open("a", encoding="utf-8-sig", newline="") as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = _train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
            val_metrics = _evaluate(model, val_loader, criterion, device, classes) if len(val_set) else {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0, "per_class_acc": ""}
            scheduler.step()
            lr = float(optimizer.param_groups[0]["lr"])
            current_val_loss = float(val_metrics["loss"])
            current_val_acc = float(val_metrics["acc"])
            val_loss_history.append(current_val_loss)
            val_acc_history.append(current_val_acc)
            improved = float(val_metrics["acc"]) > best_val_acc
            if improved:
                best_val_acc = current_val_acc
                best_macro_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                stale_epochs = 0
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": classes,
                        "model_name": args.model,
                        "image_size": args.image_size,
                        "best_val_acc": best_val_acc,
                        "best_macro_f1": best_macro_f1,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_acc": f"{train_metrics['acc']:.6f}",
                "val_loss": f"{current_val_loss:.6f}",
                "val_acc": f"{current_val_acc:.6f}",
                "macro_f1": f"{float(val_metrics['macro_f1']):.6f}",
                "per_class_acc": val_metrics["per_class_acc"],
                "lr": f"{lr:.10g}",
                "best_epoch": best_epoch,
            }
            writer.writerow(row)
            log_handle.flush()
            print(
                "epoch={epoch} train_loss={train_loss} train_acc={train_acc} "
                "val_loss={val_loss} val_acc={val_acc} macro_f1={macro_f1} "
                "per_class_acc={per_class_acc} lr={lr} best_epoch={best_epoch}".format(**row),
                flush=True,
            )
            if len(val_loss_history) >= 5:
                recent_losses = val_loss_history[-4:]
                previous_best_acc = max(val_acc_history[:-4]) if len(val_acc_history) > 4 else -1.0
                losses_rising = all(earlier < later for earlier, later in zip(recent_losses, recent_losses[1:]))
                acc_not_improving = max(val_acc_history[-4:]) <= previous_best_acc
                if losses_rising and acc_not_improving:
                    print("early_stop_reason=val_loss_rising_4_epochs_without_val_acc_improvement", flush=True)
                    break
            if stale_epochs >= args.early_stop_patience:
                print("early_stop_reason=val_acc_patience", flush=True)
                break

    result = {"best_epoch": best_epoch, "best_val_acc": best_val_acc, "best_macro_f1": best_macro_f1}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train 202/203/204 expert classifier")
    parser.add_argument("--train_csv", default="data/manifests/expert_202_203_204_train.csv")
    parser.add_argument("--val_csv", default="splits/val_real.csv")
    parser.add_argument("--classes", default="202,203,204")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--use_amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--balanced_sampler", type=int, choices=[0, 1], default=1)
    parser.add_argument("--aug_strength", default="light", choices=["none", "light", "medium", "strong"])
    parser.add_argument("--checkpoint_path", default="checkpoints/expert_202_203_204.pt")
    parser.add_argument("--log_path", default="output/expert_202_203_204_train_log.csv")
    parser.add_argument("--metrics_path", default="output/expert_202_203_204_metrics.json")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run_training(create_arg_parser().parse_args())


if __name__ == "__main__":
    main()
