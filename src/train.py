import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.augment import (
    AUG_STRENGTH_CHOICES,
    NOISE_STRENGTH_CHOICES,
    PREPROCESSING_PROFILE_CHOICES,
    get_train_transform,
    get_val_transform,
)
from src.class_filter import load_classes
from src.dataset import TrafficSignDataset
from src.losses import BatchBalancedFocalLoss, FocalLoss
from src.model import build_model, load_checkpoint
from src.optimizers import Lion, SAM
from src.validate import (
    compute_classification_metrics,
    evaluate_model,
    plot_confusion_matrix,
    write_class_metrics,
    write_top_confusions,
    write_wrong_cases,
)
from src.wavelet_features import DEFAULT_WAVELET, WAVELET_MODES, WaveletFeatureTransform, wavelet_input_channels


MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_HEAD_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LABEL_SMOOTHING = 0.05
DEFAULT_EARLY_STOP_PATIENCE = 10
TRAIN_LOG_FIELDS = [
    "epoch",
    "train_loss",
    "train_top1_acc",
    "train_macro_f1_all_classes",
    "train_macro_f1_present_classes",
    "train_weighted_f1",
    "val_loss",
    "val_top1_acc",
    "val_top3_acc",
    "val_macro_f1_all_classes",
    "val_macro_f1_present_classes",
    "val_weighted_f1",
    "score",
    "best_score",
    "best_epoch",
    "lr",
    "head_lr",
    "optimizer",
    "weight_decay",
    "label_smoothing",
    "mixup_alpha",
    "cutmix_alpha",
    "loss",
    "focal_gamma",
    "rand_augment_n",
    "rand_augment_m",
    "preprocessing_profile",
    "gap_acc",
    "overfit_flag",
]
FORBIDDEN_VAL_SOURCES = {
    "augmented_real",
    "real_student_images",
    "augmented_student",
    "augmented_hardcase",
    "synthetic",
    "pdf",
    "diffusion",
}
ALLOWED_VAL_SOURCES = {"real", "student"}
FORBIDDEN_VAL_PATH_MARKERS = FORBIDDEN_VAL_SOURCES - {"real_student_images"}


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="generalization-first real-image traffic sign trainer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train_csv", default="splits/train_real.csv")
    parser.add_argument("--val_csv", default="splits/val_real.csv")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--head_lr", type=float, default=DEFAULT_HEAD_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd", "lion", "sam"], default="adamw")
    parser.add_argument("--loss", choices=["ce", "focal", "bbfl"], default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)
    parser.add_argument("--use_amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--balanced_sampler", type=int, choices=[0, 1], default=1)
    parser.add_argument("--class_weight_loss", type=int, choices=[0, 1], default=0)
    parser.add_argument("--early_stop_patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--aug_strength", choices=AUG_STRENGTH_CHOICES, default="medium")
    parser.add_argument("--noise_strength", choices=NOISE_STRENGTH_CHOICES, default="none")
    parser.add_argument("--preprocessing_profile", choices=PREPROCESSING_PROFILE_CHOICES, default="legacy")
    parser.add_argument("--rand_augment_n", type=int, default=0)
    parser.add_argument("--rand_augment_m", type=int, default=9)
    parser.add_argument("--scheduler", choices=["cosine", "onecycle"], default="cosine")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--metrics_path", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--pad_to_square", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--unfreeze_layer4_only", type=int, choices=[0, 1], default=1)
    parser.add_argument("--val_loss_stop_patience", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wavelet_mode", choices=WAVELET_MODES, default="none")
    parser.add_argument("--wavelet", default=DEFAULT_WAVELET)
    parser.add_argument("--wavelet_level", type=int, choices=[1, 2], default=1)
    return parser


def _load_config(path: str | Path) -> Dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_device_legacy_unused(configured: str) -> torch.device:
    # 中文说明：默认优先 CUDA；如果没有 CUDA，再自动回退 CPU。
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)
    # 中文说明：默认优先 CUDA；如果没有 CUDA，再自动回退 CPU。
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs > MAX_EPOCHS:
        raise ValueError(f"epochs must be <= {MAX_EPOCHS}; got {args.epochs}")
    if args.head_lr < args.lr:
        raise ValueError(f"head_lr must be >= lr; got head_lr={args.head_lr}, lr={args.lr}")
    if args.mixup_alpha < 0:
        raise ValueError(f"mixup_alpha must be >= 0; got {args.mixup_alpha}")
    if args.cutmix_alpha < 0:
        raise ValueError(f"cutmix_alpha must be >= 0; got {args.cutmix_alpha}")
    if args.rand_augment_n < 0:
        raise ValueError(f"rand_augment_n must be >= 0; got {args.rand_augment_n}")
    if args.rand_augment_m < 0:
        raise ValueError(f"rand_augment_m must be >= 0; got {args.rand_augment_m}")
    if args.focal_gamma < 0:
        raise ValueError(f"focal_gamma must be >= 0; got {args.focal_gamma}")
    if args.wavelet_mode != "none" and args.model.lower() not in {"resnet18", "resnet50"}:
        raise ValueError("wavelet_mode is only supported with resnet18 or resnet50")


def assert_validation_manifest_clean(val_csv: str | Path) -> None:
    """中文说明：验证集只能包含原始 real，严禁 augmented_real/synthetic/pdf/diffusion。"""
    val_csv = Path(val_csv)
    with val_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label", "source"]:
            raise ValueError("val_csv columns must be path,label,source")
        for row_number, row in enumerate(reader, start=2):
            source = row["source"].strip()
            path_text = row["path"].lower()
            if source not in ALLOWED_VAL_SOURCES or source in FORBIDDEN_VAL_SOURCES:
                raise ValueError(f"validation row {row_number}: forbidden source={source}")
            if any(marker in path_text for marker in FORBIDDEN_VAL_PATH_MARKERS):
                raise ValueError(f"validation row {row_number}: forbidden validation path={row['path']}")


def _split_head_backbone_params(model: nn.Module) -> Tuple[list[nn.Parameter], list[nn.Parameter]]:
    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc.") or name.startswith("classifier.") or name.startswith("head."):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def make_optimizer(model: nn.Module, lr: float, head_lr: float, weight_decay: float, optimizer_name: str = "adamw"):
    """中文说明：backbone 用较小 lr，分类头 fc/classifier 用较大 head_lr。"""
    backbone_params, head_params = _split_head_backbone_params(model)
    param_groups = [
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": head_lr},
    ]
    if optimizer_name == "adamw":
        return AdamW(param_groups, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return Adam(param_groups, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return SGD(param_groups, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    if optimizer_name == "lion":
        return Lion(param_groups, weight_decay=weight_decay)
    if optimizer_name == "sam":
        return SAM(param_groups, AdamW, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def build_loss(loss_name: str, class_weights: torch.Tensor | None, label_smoothing: float, focal_gamma: float):
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    if loss_name == "bbfl":
        return BatchBalancedFocalLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"unsupported loss: {loss_name}")


def build_weighted_sampler(label_indices: Iterable[int]) -> tuple[WeightedRandomSampler, list[float]]:
    """中文说明：按训练集中每类数量反比采样，少样本类别获得更高概率。"""
    labels = list(label_indices)
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)
    return sampler, weights


def build_class_weight_tensor(label_indices: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    """中文说明：可选 class-balanced loss；默认不开，避免和 sampler 叠加过度补偿。"""
    labels = list(label_indices)
    counts = Counter(labels)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for class_index in range(num_classes):
        if counts[class_index] > 0:
            weights[class_index] = len(labels) / (num_classes * counts[class_index])
    return weights.to(device)


def compute_score(top1_acc: float, macro_f1_present_classes: float, top3_acc: float, weighted_f1: float) -> float:
    return 0.50 * top1_acc + 0.25 * macro_f1_present_classes + 0.15 * top3_acc + 0.10 * weighted_f1


def apply_trainable_state(model: nn.Module, epoch: int, freeze_backbone_epochs: int, unfreeze_layer4_only: bool) -> None:
    """中文说明：小数据 staged training；先只训 fc，之后可只解冻 layer4 + fc。"""
    if freeze_backbone_epochs > 0 and epoch <= freeze_backbone_epochs:
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.startswith("fc.") or name.startswith("classifier.") or name.startswith("head."):
                param.requires_grad = True
        return
    if unfreeze_layer4_only:
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if (
                name.startswith("layer4.")
                or name.startswith("stages.3.")
                or name.startswith("blocks.10.")
                or name.startswith("blocks.11.")
                or name.startswith("fc.")
                or name.startswith("classifier.")
                or name.startswith("head.")
            ):
                param.requires_grad = True
        return
    for _, param in model.named_parameters():
        param.requires_grad = True


def compute_overfit_flag(
    train_top1_acc: float,
    val_top1_acc: float,
    train_macro_f1: float,
    val_macro_f1: float,
    train_loss_history: list[float],
    val_loss_history: list[float],
) -> str:
    """中文说明：根据训练/验证差距和 loss 趋势给出过拟合预警。"""
    gap_acc = train_top1_acc - val_top1_acc
    gap_f1 = train_macro_f1 - val_macro_f1
    if gap_acc > 0.20 or gap_f1 > 0.30:
        return "YES"
    if len(train_loss_history) >= 2 and len(val_loss_history) >= 3:
        train_loss_down = train_loss_history[-1] < train_loss_history[-2]
        val_loss_rising = val_loss_history[-3] < val_loss_history[-2] < val_loss_history[-1]
        if train_loss_down and val_loss_rising:
            return "YES"
    return "NO"


def _optimizer_lrs(optimizer) -> tuple[float, float]:
    lr = float(optimizer.param_groups[0]["lr"])
    head_lr = float(optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"])
    return lr, head_lr


def _mixup_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    """训练集专用 Mixup；验证集不会调用，label smoothing 仍由 CE 在两组硬标签上分别处理。"""
    if alpha <= 0.0 or images.size(0) < 2:
        return images, targets, targets, 1.0
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, targets, targets[index], lam


def _cutmix_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    if alpha <= 0.0 or images.size(0) < 2:
        return images, targets, targets, 1.0
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    batch_size, _, height, width = images.shape
    index = torch.randperm(batch_size, device=images.device)
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = int(torch.randint(width, (1,), device=images.device).item())
    cy = int(torch.randint(height, (1,), device=images.device).item())
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    patched_area = max(0, x2 - x1) * max(0, y2 - y1)
    lam = 1.0 - patched_area / float(width * height)
    return mixed_images, targets, targets[index], lam


def _mixup_loss(criterion, logits: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)


def _augment_batch_for_mixed_loss(images: torch.Tensor, targets: torch.Tensor, mixup_alpha: float, cutmix_alpha: float):
    if cutmix_alpha > 0.0:
        return _cutmix_batch(images, targets, cutmix_alpha)
    if mixup_alpha > 0.0:
        return _mixup_batch(images, targets, mixup_alpha)
    return images, targets, targets, 1.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    use_amp: bool,
    scaler,
    num_classes: int,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    all_targets: list[int] = []
    all_preds: list[int] = []
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        try:
            mixed_images, targets_a, targets_b, lam = _augment_batch_for_mixed_loss(images, targets, mixup_alpha, cutmix_alpha)
            mixed_targets = mixup_alpha > 0.0 or cutmix_alpha > 0.0
            if isinstance(optimizer, SAM):
                with torch.cuda.amp.autocast(enabled=False):
                    logits = model(mixed_images)
                    loss = _mixup_loss(criterion, logits, targets_a, targets_b, lam) if mixed_targets else criterion(logits, targets)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                with torch.cuda.amp.autocast(enabled=False):
                    second_logits = model(mixed_images)
                    second_loss = _mixup_loss(criterion, second_logits, targets_a, targets_b, lam) if mixed_targets else criterion(second_logits, targets)
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(mixed_images)
                    loss = _mixup_loss(criterion, logits, targets_a, targets_b, lam) if mixed_targets else criterion(logits, targets)
            if use_amp and not isinstance(optimizer, SAM):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif not isinstance(optimizer, SAM):
                loss.backward()
                optimizer.step()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CUDA OOM: batch_size=64 may be too large. Please retry with --batch_size 32.") from exc
            raise
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(logits.detach().argmax(dim=1).cpu().tolist())
    top1_acc = float(sum(int(a == b) for a, b in zip(all_targets, all_preds)) / max(1, len(all_targets)))
    f1_metrics = compute_classification_metrics(all_targets, all_preds, [[pred] for pred in all_preds], num_classes)
    return {
        "loss": total_loss / max(1, total),
        "top1": top1_acc,
        "macro_f1_all_classes": f1_metrics["macro_f1_all_classes"],
        "macro_f1_present_classes": f1_metrics["macro_f1_present_classes"],
        "weighted_f1": f1_metrics["weighted_f1"],
    }


def _append_train_log(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _print_epoch_summary(
    epoch: int,
    epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    score: float,
    best_score: float,
    best_epoch: int,
    lr: float,
    head_lr: float,
    optimizer_name: str,
    weight_decay: float,
    label_smoothing: float,
    gap_acc: float,
    gap_f1: float,
    overfit_flag: str,
    checkpoint_path: Path,
) -> None:
    line = "=" * 60
    print(line)
    print(f"Epoch {epoch:02d}/{epochs}")
    print(
        f"train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['top1']:.4f} | "
        f"train_f1_present={train_metrics['macro_f1_present_classes']:.4f} | train_weighted_f1={train_metrics['weighted_f1']:.4f}"
    )
    print(
        f"val_loss={val_metrics['loss']:.4f}   | val_acc={val_metrics['top1']:.4f}   | val_top3={val_metrics['top3']:.4f} | "
        f"macro_f1_all_classes={val_metrics['macro_f1_all_classes']:.4f} | "
        f"macro_f1_present_classes={val_metrics['macro_f1_present_classes']:.4f} | weighted_f1={val_metrics['weighted_f1']:.4f}"
    )
    print(f"score={score:.4f}      | best_score={best_score:.4f} | best_epoch={best_epoch:02d}")
    print(f"lr={lr:.8f}      | head_lr={head_lr:.8f} | optimizer={optimizer_name} | weight_decay={weight_decay:.8f} | label_smoothing={label_smoothing:.4f}")
    print(f"gap_acc={gap_acc:.4f}    | gap_f1={gap_f1:.4f} | overfit_flag={overfit_flag}")
    print(f"checkpoint={checkpoint_path.as_posix()}")
    print(line)


def _label_indices(dataset: TrafficSignDataset) -> list[int]:
    return [dataset.class_to_idx[label] for _, label in dataset.samples]


def _checkpoint_payload(
    model: nn.Module,
    optimizer,
    epoch: int,
    classes: list[str],
    args: argparse.Namespace,
    metrics: Dict[str, float],
    score: float,
) -> Dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "classes": classes,
        "model_name": args.model,
        "image_size": args.image_size,
        "pad_to_square": int(args.pad_to_square),
        "wavelet_mode": args.wavelet_mode,
        "wavelet": args.wavelet,
        "wavelet_level": args.wavelet_level,
        "noise_strength": args.noise_strength,
        "preprocessing_profile": args.preprocessing_profile,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "rand_augment_n": args.rand_augment_n,
        "rand_augment_m": args.rand_augment_m,
        "input_channels": wavelet_input_channels(args.wavelet_mode),
        "best_score": score,
        "best_top1_acc": metrics["top1"],
        "best_top3_acc": metrics["top3"],
        "best_macro_f1_all_classes": metrics["macro_f1_all_classes"],
        "best_macro_f1_present_classes": metrics["macro_f1_present_classes"],
        "best_weighted_f1": metrics["weighted_f1"],
        "train_args": vars(args),
    }


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


def run_training(args: argparse.Namespace) -> Dict[str, object]:
    _validate_args(args)
    assert_validation_manifest_clean(args.val_csv)
    config = _load_config(args.config)
    classes_path = config.get("classes_path", "classes.txt")
    classes = load_classes(classes_path)
    device = _resolve_device(args.device)
    use_amp = bool(args.use_amp and device.type == "cuda" and torch.cuda.is_available())
    train_transform = get_train_transform(
        args.image_size,
        args.aug_strength,
        pad_to_square=bool(args.pad_to_square),
        noise_strength=args.noise_strength,
        preprocessing_profile=args.preprocessing_profile,
        rand_augment_n=args.rand_augment_n,
        rand_augment_m=args.rand_augment_m,
    )
    val_transform = get_val_transform(
        args.image_size,
        pad_to_square=bool(args.pad_to_square),
        preprocessing_profile=args.preprocessing_profile,
    )
    if args.wavelet_mode != "none":
        train_transform = WaveletFeatureTransform(train_transform, mode=args.wavelet_mode, wavelet=args.wavelet, level=args.wavelet_level)
        val_transform = WaveletFeatureTransform(val_transform, mode=args.wavelet_mode, wavelet=args.wavelet, level=args.wavelet_level)

    train_set = TrafficSignDataset(
        args.train_csv,
        classes_path,
        train_transform,
        allowed_sources={
            "real",
            "augmented_real",
            "real_student_images",
            "augmented_student",
            "augmented_hardcase",
            "student",
            "diffusion",
            "targeted_aug",
        },
    )
    val_set = TrafficSignDataset(
        args.val_csv,
        classes_path,
        val_transform,
        allowed_sources=ALLOWED_VAL_SOURCES,
    )
    train_labels = _label_indices(train_set)
    if args.balanced_sampler:
        sampler, _ = build_weighted_sampler(train_labels)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model(args.model, len(classes), pretrained=args.pretrained, input_channels=wavelet_input_channels(args.wavelet_mode)).to(device)
    if args.init_checkpoint:
        payload = load_checkpoint(args.init_checkpoint, model, device)
        checkpoint_classes = payload.get("classes") if isinstance(payload, dict) else None
        if checkpoint_classes is not None and list(checkpoint_classes) != list(classes):
            raise ValueError("init_checkpoint classes do not match classes.txt")
    optimizer = make_optimizer(model, args.lr, args.head_lr, args.weight_decay, args.optimizer)
    class_weights = build_class_weight_tensor(train_labels, len(classes), device) if args.class_weight_loss else None
    criterion = build_loss(args.loss, class_weights, args.label_smoothing, args.focal_gamma)
    if args.scheduler == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=[args.lr, args.head_lr], epochs=args.epochs, steps_per_epoch=max(1, len(train_loader)))
        scheduler_step_per_batch = True
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
        scheduler_step_per_batch = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_path or Path("checkpoints") / f"best_{args.model}.pt")
    log_path = Path(args.log_path or output_dir / "train_log.csv")
    metrics_path = Path(args.metrics_path or output_dir / "metrics.json")
    if log_path.exists():
        log_path.unlink()

    best_score = -1.0
    best_epoch = 0
    epochs_ran = 0
    epochs_without_improve = 0
    stopped_by_early_stop = False
    stopped_by_val_loss_rise = False
    stopped_by_target_acc = False
    stop_at_acc = getattr(args, "stop_at_acc", None)
    best_metrics = {
        "top1": 0.0,
        "top3": 0.0,
        "macro_f1_all_classes": 0.0,
        "macro_f1_present_classes": 0.0,
        "weighted_f1": 0.0,
    }
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        apply_trainable_state(model, epoch, args.freeze_backbone_epochs, bool(args.unfreeze_layer4_only))
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp,
            scaler,
            len(classes),
            scheduler,
            scheduler_step_per_batch,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
        )
        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()
        metrics = evaluate_model(model, val_loader, device, classes, criterion=criterion)
        score = compute_score(
            top1_acc=metrics["top1"],
            macro_f1_present_classes=metrics["macro_f1_present_classes"],
            top3_acc=metrics["top3"],
            weighted_f1=metrics["weighted_f1"],
        )
        train_loss_history.append(train_metrics["loss"])
        val_loss_history.append(metrics["loss"])
        gap_acc = train_metrics["top1"] - metrics["top1"]
        gap_f1 = train_metrics["macro_f1_present_classes"] - metrics["macro_f1_present_classes"]
        overfit_flag = compute_overfit_flag(
            train_metrics["top1"],
            metrics["top1"],
            train_metrics["macro_f1_present_classes"],
            metrics["macro_f1_present_classes"],
            train_loss_history,
            val_loss_history,
        )
        current_lr, current_head_lr = _optimizer_lrs(optimizer)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            best_metrics = {
                "top1": metrics["top1"],
                "top3": metrics["top3"],
                "macro_f1_all_classes": metrics["macro_f1_all_classes"],
                "macro_f1_present_classes": metrics["macro_f1_present_classes"],
                "weighted_f1": metrics["weighted_f1"],
            }
            epochs_without_improve = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_checkpoint_payload(model, optimizer, epoch, classes, args, metrics, score), checkpoint_path)
            plot_confusion_matrix(metrics["targets"], metrics["preds"], classes, output_dir / "confusion_matrix.png")
            write_wrong_cases(metrics["paths"], metrics["targets"], metrics["preds"], metrics["top3_indices"], classes, output_dir / "wrong_cases.csv")
            write_class_metrics(metrics["targets"], metrics["preds"], classes, output_dir / "per_class_accuracy.csv")
            write_top_confusions(metrics["targets"], metrics["preds"], classes, output_dir / "confusion_pairs.csv")
        else:
            epochs_without_improve += 1
        _append_train_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_top1_acc": f"{train_metrics['top1']:.6f}",
                "train_macro_f1_all_classes": f"{train_metrics['macro_f1_all_classes']:.6f}",
                "train_macro_f1_present_classes": f"{train_metrics['macro_f1_present_classes']:.6f}",
                "train_weighted_f1": f"{train_metrics['weighted_f1']:.6f}",
                "val_loss": f"{metrics['loss']:.6f}",
                "val_top1_acc": f"{metrics['top1']:.6f}",
                "val_top3_acc": f"{metrics['top3']:.6f}",
                "val_macro_f1_all_classes": f"{metrics['macro_f1_all_classes']:.6f}",
                "val_macro_f1_present_classes": f"{metrics['macro_f1_present_classes']:.6f}",
                "val_weighted_f1": f"{metrics['weighted_f1']:.6f}",
                "score": f"{score:.6f}",
                "best_score": f"{best_score:.6f}",
                "best_epoch": best_epoch,
                "lr": f"{current_lr:.10g}",
                "head_lr": f"{current_head_lr:.10g}",
                "optimizer": args.optimizer,
                "weight_decay": f"{args.weight_decay:.10g}",
                "label_smoothing": f"{args.label_smoothing:.10g}",
                "mixup_alpha": f"{args.mixup_alpha:.10g}",
                "cutmix_alpha": f"{args.cutmix_alpha:.10g}",
                "loss": args.loss,
                "focal_gamma": f"{args.focal_gamma:.10g}",
                "rand_augment_n": args.rand_augment_n,
                "rand_augment_m": args.rand_augment_m,
                "preprocessing_profile": args.preprocessing_profile,
                "gap_acc": f"{gap_acc:.6f}",
                "overfit_flag": overfit_flag,
            },
        )
        _print_epoch_summary(
            epoch,
            args.epochs,
            train_metrics,
            metrics,
            score,
            best_score,
            best_epoch,
            current_lr,
            current_head_lr,
            args.optimizer,
            args.weight_decay,
            args.label_smoothing,
            gap_acc,
            gap_f1,
            overfit_flag,
            checkpoint_path,
        )
        if not is_best and epochs_without_improve >= args.early_stop_patience:
            stopped_by_early_stop = True
            break
        if stop_at_acc is not None and metrics["top1"] >= float(stop_at_acc):
            stopped_by_target_acc = True
            break
        if len(val_loss_history) >= args.val_loss_stop_patience >= 3:
            recent_losses = val_loss_history[-args.val_loss_stop_patience :]
            val_loss_rising = all(left < right for left, right in zip(recent_losses, recent_losses[1:]))
            if val_loss_rising and not is_best:
                stopped_by_val_loss_rise = True
                break

    result = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_top1_acc": best_metrics["top1"],
        "best_top3_acc": best_metrics["top3"],
        "best_macro_f1_all_classes": best_metrics["macro_f1_all_classes"],
        "best_macro_f1_present_classes": best_metrics["macro_f1_present_classes"],
        "best_weighted_f1": best_metrics["weighted_f1"],
        "epochs_ran": epochs_ran,
        "stopped_by_early_stop": stopped_by_early_stop,
        "stopped_by_val_loss_rise": stopped_by_val_loss_rise,
        "stopped_by_target_acc": stopped_by_target_acc,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"best_checkpoint={checkpoint_path.as_posix()}")
    print(f"best_score={best_score:.6f}")
    return result


def main() -> None:
    args = create_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
