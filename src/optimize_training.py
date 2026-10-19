import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SEARCH_SPACE = {
    "optimizer": ["adam", "adamw"],
    "lr": [3e-5, 5e-5, 8e-5],
    "head_lr": [1e-4, 2e-4, 3e-4],
    "weight_decay": [1e-4, 3e-4],
    "label_smoothing": [0.03, 0.05],
    "image_size": [256],
    "pad_to_square": [0, 1],
    "aug_strength": ["light", "medium"],
    "freeze_backbone_epochs": [0, 2],
    "class_weight_loss": [0],
}

OPTUNA_TRIAL_FIELDS = [
    "trial",
    "optimizer",
    "lr",
    "head_lr",
    "weight_decay",
    "label_smoothing",
    "image_size",
    "pad_to_square",
    "aug_strength",
    "freeze_backbone_epochs",
    "class_weight_loss",
    "val_acc",
    "val_top3",
    "macro_f1_present",
    "weighted_f1",
    "external_correct",
    "203_correct_after",
    "wrong_107",
    "wrong_203",
    "train_val_gap",
    "overfit_flag",
    "adoption_decision",
    "checkpoint",
]


@dataclass(frozen=True)
class TrialConfig:
    trial_number: int
    optimizer: str
    lr: float
    head_lr: float
    weight_decay: float
    label_smoothing: float
    image_size: int
    pad_to_square: int
    aug_strength: str
    freeze_backbone_epochs: int
    class_weight_loss: int = 0
    epochs: int = 10
    early_stop_patience: int = 3
    batch_size: int = 64

    @property
    def trial_id(self) -> str:
        return f"trial_{self.trial_number:03d}"


def _fmt_float(value: float) -> str:
    return f"{value:.10g}"


def _trial_dir(trial: TrialConfig, output_dir: str | Path = "output/optimax_student_batch") -> Path:
    return Path(output_dir) / trial.trial_id


def _checkpoint_path(trial: TrialConfig, checkpoint_dir: str | Path = "checkpoints/optimax_student_batch") -> Path:
    return Path(checkpoint_dir) / f"{trial.trial_id}_best.pt"


def generate_trials(
    max_trials: int = 6,
    seed: int = 42,
    include_class_weight_loss_control: bool = False,
    epochs: int = 10,
    early_stop_patience: int = 3,
    batch_size: int = 64,
) -> list[TrialConfig]:
    space = dict(SEARCH_SPACE)
    if include_class_weight_loss_control:
        space["class_weight_loss"] = [0, 1]
    keys = list(space.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]
    rng = random.Random(seed)
    rng.shuffle(combos)
    trials: list[TrialConfig] = []
    for index, combo in enumerate(combos[:max(0, max_trials)], start=1):
        trials.append(
            TrialConfig(
                trial_number=index,
                optimizer=str(combo["optimizer"]),
                lr=float(combo["lr"]),
                head_lr=float(combo["head_lr"]),
                weight_decay=float(combo["weight_decay"]),
                label_smoothing=float(combo["label_smoothing"]),
                image_size=int(combo["image_size"]),
                pad_to_square=int(combo["pad_to_square"]),
                aug_strength=str(combo["aug_strength"]),
                freeze_backbone_epochs=int(combo["freeze_backbone_epochs"]),
                class_weight_loss=int(combo["class_weight_loss"]),
                epochs=min(int(epochs), 10),
                early_stop_patience=int(early_stop_patience),
                batch_size=int(batch_size),
            )
        )
    return trials


def build_train_command(
    trial: TrialConfig,
    train_csv: str = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv",
    val_csv: str = "splits/val_real.csv",
    model: str = "resnet50",
    checkpoint_dir: str | Path = "checkpoints/optimax_student_batch",
    output_dir: str | Path = "output/optimax_student_batch",
    init_checkpoint: str | None = None,
) -> list[str]:
    trial_dir = _trial_dir(trial, output_dir)
    command = [
        sys.executable,
        "src/train.py",
        "--train_csv",
        train_csv,
        "--val_csv",
        val_csv,
        "--model",
        model,
        "--epochs",
        str(min(trial.epochs, 10)),
        "--batch_size",
        str(trial.batch_size),
        "--lr",
        _fmt_float(trial.lr),
        "--head_lr",
        _fmt_float(trial.head_lr),
        "--optimizer",
        trial.optimizer,
        "--weight_decay",
        _fmt_float(trial.weight_decay),
        "--label_smoothing",
        _fmt_float(trial.label_smoothing),
        "--use_amp",
        "1",
        "--balanced_sampler",
        "1",
        "--class_weight_loss",
        str(trial.class_weight_loss),
        "--early_stop_patience",
        str(trial.early_stop_patience),
        "--aug_strength",
        trial.aug_strength,
        "--scheduler",
        "cosine",
        "--freeze_backbone_epochs",
        str(trial.freeze_backbone_epochs),
        "--unfreeze_layer4_only",
        "0",
        "--image_size",
        str(trial.image_size),
        "--pad_to_square",
        str(trial.pad_to_square),
        "--checkpoint_path",
        _checkpoint_path(trial, checkpoint_dir).as_posix(),
        "--output_dir",
        trial_dir.as_posix(),
        "--log_path",
        (trial_dir / "train_log.csv").as_posix(),
        "--metrics_path",
        (trial_dir / "metrics.json").as_posix(),
    ]
    if init_checkpoint:
        command.extend(["--init_checkpoint", init_checkpoint])
    return command


def build_eval_commands(
    trial: TrialConfig,
    expert_checkpoint: str = "checkpoints/expert_202_203_204.pt",
    baseline_debug: str = "output/debug_expert_202_203_204.csv",
    checkpoint_dir: str | Path = "checkpoints/optimax_student_batch",
    output_dir: str | Path = "output/optimax_student_batch",
) -> tuple[list[str], list[str], list[str]]:
    trial_dir = _trial_dir(trial, output_dir)
    debug_csv = trial_dir / "debug.csv"
    submission_csv = trial_dir / "submission.csv"
    audit_dir = trial_dir / "audit"
    impact_csv = trial_dir / "impact.csv"
    infer_cmd = [
        sys.executable,
        "src/infer.py",
        "--input_csv",
        "input/input.csv",
        "--image_dir",
        "input/images",
        "--checkpoint",
        _checkpoint_path(trial, checkpoint_dir).as_posix(),
        "--output",
        submission_csv.as_posix(),
        "--debug_csv",
        debug_csv.as_posix(),
        "--image_size",
        str(trial.image_size),
        "--pad_to_square",
        str(trial.pad_to_square),
        "--tta",
        "0",
        "--coarse_filter",
        "0",
        "--use_expert_202_203_204",
        "1",
        "--expert_202_203_204_checkpoint",
        expert_checkpoint,
        "--expert_trigger_rank",
        "3",
        "--expert_mode",
        "conservative",
    ]
    eval_cmd = [
        sys.executable,
        "src/evaluate_test_predictions.py",
        "--pred",
        debug_csv.as_posix(),
        "--gt",
        "input/ground_truth.csv",
        "--output_dir",
        audit_dir.as_posix(),
    ]
    impact_cmd = [
        sys.executable,
        "src/evaluate_expert_impact.py",
        "--baseline_debug",
        baseline_debug,
        "--expert_debug",
        debug_csv.as_posix(),
        "--gt",
        "input/ground_truth.csv",
        "--output_csv",
        impact_csv.as_posix(),
    ]
    return infer_cmd, eval_cmd, impact_cmd


def append_trial_result(path: str | Path, row: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTUNA_TRIAL_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in OPTUNA_TRIAL_FIELDS})


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _read_last_best_log_row(path: Path, best_epoch: int) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if int(row.get("epoch", "0") or 0) == best_epoch:
            return row
    return rows[-1] if rows else {}


def adoption_decision(external_correct: int) -> str:
    if external_correct >= 87:
        return "strong_candidate"
    if 83 <= external_correct <= 86:
        return "keep_candidate"
    return "reject"


def _wrong_counts(audit_dir: Path) -> tuple[int, int]:
    wrong_csv = audit_dir / "wrong_cases.csv"
    if not wrong_csv.exists():
        return 0, 0
    wrong_107 = 0
    wrong_203 = 0
    with wrong_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            wrong_107 += int(row.get("true_class", "") == "107")
            wrong_203 += int(row.get("true_class", "") == "203")
    return wrong_107, wrong_203


def _correct_for_class(audit_dir: Path, label: str) -> int:
    per_class_csv = audit_dir / "per_class_accuracy.csv"
    if not per_class_csv.exists():
        return 0
    with per_class_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Class") == label:
                return int(float(row.get("correct", 0) or 0))
    return 0


def collect_trial_result(
    trial: TrialConfig,
    checkpoint_dir: str | Path = "checkpoints/optimax_student_batch",
    output_dir: str | Path = "output/optimax_student_batch",
) -> dict[str, object]:
    trial_dir = _trial_dir(trial, output_dir)
    metrics_path = trial_dir / "metrics.json"
    test_metrics_path = trial_dir / "audit" / "test_metrics.txt"
    train_log_path = trial_dir / "train_log.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_epoch = int(metrics.get("best_epoch", 0))
    log_row = _read_last_best_log_row(train_log_path, best_epoch)
    external_metrics = _read_key_values(test_metrics_path)
    external_correct = int(float(external_metrics.get("correct", 0)))
    val_acc = float(metrics.get("best_top1_acc", 0.0))
    train_acc = float(log_row.get("train_top1_acc", 0.0) or 0.0)
    wrong_107, wrong_203 = _wrong_counts(trial_dir / "audit")
    return {
        "trial": trial.trial_id,
        "optimizer": trial.optimizer,
        "lr": trial.lr,
        "head_lr": trial.head_lr,
        "weight_decay": trial.weight_decay,
        "label_smoothing": trial.label_smoothing,
        "image_size": trial.image_size,
        "pad_to_square": trial.pad_to_square,
        "aug_strength": trial.aug_strength,
        "freeze_backbone_epochs": trial.freeze_backbone_epochs,
        "class_weight_loss": trial.class_weight_loss,
        "val_acc": val_acc,
        "val_top3": float(metrics.get("best_top3_acc", 0.0)),
        "macro_f1_present": float(metrics.get("best_macro_f1_present_classes", 0.0)),
        "weighted_f1": float(metrics.get("best_weighted_f1", 0.0)),
        "external_correct": external_correct,
        "203_correct_after": _correct_for_class(trial_dir / "audit", "203"),
        "wrong_107": wrong_107,
        "wrong_203": wrong_203,
        "train_val_gap": train_acc - val_acc,
        "overfit_flag": log_row.get("overfit_flag", ""),
        "adoption_decision": adoption_decision(external_correct),
        "checkpoint": _checkpoint_path(trial, checkpoint_dir).as_posix(),
    }


def write_best_summary(path: str | Path, rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(rows)
    best = max(
        rows,
        key=lambda row: (
            int(row.get("external_correct", 0)),
            float(row.get("val_acc", 0.0)),
            float(row.get("macro_f1_present", 0.0)),
        ),
        default={},
    )
    summary = {
        "selection_metric": "external_correct",
        "baseline_external_correct": "",
        "best_trial": best,
        "adoption_standard": "strong if external_correct >= 87; 83-86 keeps candidate only",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_search(args: argparse.Namespace) -> dict[str, object]:
    trials = generate_trials(
        args.trials,
        args.seed,
        args.include_class_weight_loss_control,
        args.max_epochs,
        args.early_stop_patience,
        args.batch_size,
    )
    results_path = Path(args.results)
    if results_path.exists():
        results_path.unlink()
    rows: list[dict[str, object]] = []
    for trial in trials:
        commands = [
            build_train_command(
                trial,
                args.train_csv,
                args.val_csv,
                args.model,
                args.checkpoint_dir,
                args.output_dir,
                args.init_checkpoint,
            ),
            *build_eval_commands(trial, args.expert_checkpoint, args.baseline_debug, args.checkpoint_dir, args.output_dir),
        ]
        print(f"trial={trial.trial_id}")
        for command in commands:
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)
        if not args.dry_run:
            row = collect_trial_result(trial, args.checkpoint_dir, args.output_dir)
            append_trial_result(results_path, row)
            rows.append(row)
    if not args.dry_run:
        return write_best_summary(args.summary, rows)
    return {}


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="small Optuna-style training search with external test evaluation")
    parser.add_argument("--train_csv", default="data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv")
    parser.add_argument("--val_csv", default="splits/val_real.csv")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert_checkpoint", default="checkpoints/expert_202_203_204.pt")
    parser.add_argument("--baseline_debug", default="output/debug_expert_202_203_204.csv")
    parser.add_argument("--results", default="output/optimax_student_batch/results.csv")
    parser.add_argument("--summary", default="output/optimax_student_batch/best_summary.json")
    parser.add_argument("--checkpoint_dir", default="checkpoints/optimax_student_batch")
    parser.add_argument("--output_dir", default="output/optimax_student_batch")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--include_class_weight_loss_control", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    run_search(create_arg_parser().parse_args())


if __name__ == "__main__":
    main()
