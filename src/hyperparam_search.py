import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MAX_EPOCHS = 50
SEARCH_RESULTS_FIELDS = [
    "trial",
    "optimizer",
    "lr",
    "head_lr",
    "weight_decay",
    "label_smoothing",
    "aug_strength",
    "balanced_sampler",
    "class_weight_loss",
    "scheduler",
    "freeze_backbone_epochs",
    "unfreeze_layer4_only",
    "best_top1_acc",
    "best_top3_acc",
    "best_macro_f1_present_classes",
    "best_score",
    "best_epoch",
    "checkpoint",
]


@dataclass(frozen=True)
class TrialConfig:
    trial: int
    optimizer: str
    lr: float
    head_lr: float
    weight_decay: float
    label_smoothing: float
    aug_strength: str
    balanced_sampler: int
    class_weight_loss: int
    scheduler: str
    freeze_backbone_epochs: int
    unfreeze_layer4_only: int
    epochs: int
    batch_size: int

    @property
    def trial_id(self) -> str:
        return f"trial_{self.trial:03d}"


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def generate_trials(trials: int, epochs_per_trial: int, batch_size: int, seed: int = 42, search_space: str = "default") -> List[TrialConfig]:
    """中文说明：生成轻量随机搜索配置；epochs 永远不会超过 50。"""
    rng = random.Random(seed)
    configs: List[TrialConfig] = []
    for index in range(1, trials + 1):
        if search_space == "narrow_aug":
            lr = rng.choice([3e-5, 5e-5, 8e-5, 1e-4])
            head_lr = rng.choice([1e-4, 2e-4, 3e-4, 5e-4])
            if head_lr < lr:
                head_lr = lr
            optimizer = rng.choice(["adamw", "adam"])
            weight_decay = rng.choice([1e-5, 3e-5, 1e-4, 3e-4])
            label_smoothing = rng.choice([0.03, 0.05, 0.08, 0.1])
            aug_strength = rng.choice(["none", "light"])
            balanced_sampler = 1
            class_weight_loss = 0
            scheduler = "cosine"
            freeze_backbone_epochs = rng.choice([0, 2, 3])
            unfreeze_layer4_only = rng.choice([0, 1])
        else:
            lr = _log_uniform(rng, 1e-5, 3e-4)
            head_lr = _log_uniform(rng, 3e-5, 1e-3)
            if head_lr < lr:
                head_lr = lr
            optimizer = "adamw"
            weight_decay = rng.choice([1e-5, 3e-5, 1e-4, 3e-4, 5e-4])
            label_smoothing = rng.choice([0.0, 0.03, 0.05, 0.08, 0.1])
            aug_strength = rng.choice(["none", "light", "medium", "strong"])
            balanced_sampler = 1
            class_weight_loss = rng.choice([0, 1])
            scheduler = rng.choice(["cosine", "onecycle"])
            freeze_backbone_epochs = 2
            unfreeze_layer4_only = 1
        configs.append(
            TrialConfig(
                trial=index,
                optimizer=optimizer,
                lr=lr,
                head_lr=head_lr,
                weight_decay=weight_decay,
                label_smoothing=label_smoothing,
                aug_strength=aug_strength,
                balanced_sampler=balanced_sampler,
                class_weight_loss=class_weight_loss,
                scheduler=scheduler,
                freeze_backbone_epochs=freeze_backbone_epochs,
                unfreeze_layer4_only=unfreeze_layer4_only,
                epochs=min(int(epochs_per_trial), MAX_EPOCHS),
                batch_size=int(batch_size),
            )
        )
    return configs


def append_search_result(path: str | Path, row: Dict[str, object]) -> None:
    """中文说明：每个 trial 训练结束后追加一行结果，便于后续选最佳 checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEARCH_RESULTS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SEARCH_RESULTS_FIELDS})


def _trial_command(args: argparse.Namespace, trial: TrialConfig, checkpoint: Path, output_dir: Path, log_path: Path, metrics_path: Path) -> list[str]:
    return [
        sys.executable,
        "src/train.py",
        "--train_csv",
        args.train_csv,
        "--val_csv",
        args.val_csv,
        "--model",
        args.model,
        "--epochs",
        str(trial.epochs),
        "--batch_size",
        str(trial.batch_size),
        "--lr",
        f"{trial.lr:.10g}",
        "--head_lr",
        f"{trial.head_lr:.10g}",
        "--weight_decay",
        f"{trial.weight_decay:.10g}",
        "--optimizer",
        trial.optimizer,
        "--label_smoothing",
        f"{trial.label_smoothing:.10g}",
        "--use_amp",
        "1",
        "--balanced_sampler",
        str(trial.balanced_sampler),
        "--class_weight_loss",
        str(trial.class_weight_loss),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--aug_strength",
        trial.aug_strength,
        "--scheduler",
        trial.scheduler,
        "--freeze_backbone_epochs",
        str(trial.freeze_backbone_epochs),
        "--unfreeze_layer4_only",
        str(trial.unfreeze_layer4_only),
        "--output_dir",
        str(output_dir),
        "--checkpoint_path",
        str(checkpoint),
        "--log_path",
        str(log_path),
        "--metrics_path",
        str(metrics_path),
    ]


def run_search(args: argparse.Namespace) -> Dict[str, object]:
    trials = generate_trials(args.trials, args.epochs_per_trial, args.batch_size, args.seed, args.search_space)
    results_path = Path(args.results)
    if results_path.exists():
        results_path.unlink()

    best_row: Dict[str, object] | None = None
    for trial in trials:
        checkpoint = Path("checkpoints/search") / f"{trial.trial_id}_best.pt"
        output_dir = Path("output/search") / trial.trial_id
        log_path = Path("output/search") / f"{trial.trial_id}_train_log.csv"
        metrics_path = Path("output/search") / f"{trial.trial_id}_metrics.json"
        command = _trial_command(args, trial, checkpoint, output_dir, log_path, metrics_path)
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {
                "trial": trial.trial_id,
                "optimizer": trial.optimizer,
                "lr": trial.lr,
                "head_lr": trial.head_lr,
                "weight_decay": trial.weight_decay,
                "label_smoothing": trial.label_smoothing,
                "aug_strength": trial.aug_strength,
                "balanced_sampler": trial.balanced_sampler,
                "class_weight_loss": trial.class_weight_loss,
                "scheduler": trial.scheduler,
                "freeze_backbone_epochs": trial.freeze_backbone_epochs,
                "unfreeze_layer4_only": trial.unfreeze_layer4_only,
                "best_top1_acc": metrics.get("best_top1_acc", 0.0),
                "best_top3_acc": metrics.get("best_top3_acc", 0.0),
                "best_macro_f1_present_classes": metrics.get("best_macro_f1_present_classes", 0.0),
                "best_score": metrics.get("best_score", 0.0),
                "best_epoch": metrics.get("best_epoch", 0),
                "checkpoint": checkpoint.as_posix(),
            }
            append_search_result(results_path, row)
            if best_row is None or float(row["best_score"]) > float(best_row["best_score"]):
                best_row = row
    if best_row is not None:
        print(f"best_trial={best_row['trial']} best_score={best_row['best_score']} checkpoint={best_row['checkpoint']}")
        return best_row
    return {}


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="random hyperparameter search without Optuna")
    parser.add_argument("--train_csv", default="splits/train_real.csv")
    parser.add_argument("--val_csv", default="splits/val_real.csv")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--epochs_per_trial", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--search_space", choices=["default", "narrow_aug"], default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="output/search/search_results.csv")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    if args.epochs_per_trial > MAX_EPOCHS:
        args.epochs_per_trial = MAX_EPOCHS
    run_search(args)


if __name__ == "__main__":
    main()
