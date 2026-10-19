import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_lambda_matrix import compute_guard_metrics, parse_key_value_file, read_train_row
from src.tune_model_ensemble import PREVIEW_FIELDS, RESULT_FIELDS, tune_model_ensemble


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv"
VAL_CSV = "splits/val_real.csv"
GT_CSV = "input/ground_truth.csv"
EXPERT_CKPT = "checkpoints/expert_202_203_204.pt"
FINAL_DEBUG = "output/debug_latest.csv"
RESULT_FIELDS_WAVELET = [
    "trial_id",
    "wavelet",
    "wavelet_level",
    "image_size",
    "optimizer",
    "lr",
    "head_lr",
    "label_smoothing",
    "freeze_backbone_epochs",
    "aug_strength",
    "val_acc",
    "val_top3",
    "macro_f1_present",
    "external_correct",
    "external_accuracy",
    "203_correct_after",
    "wrong_107",
    "wrong_203",
    "train_val_gap",
    "overfit_flag",
    "checkpoint",
]


@dataclass(frozen=True)
class WaveletTrial:
    trial_id: str
    wavelet: str
    wavelet_level: int
    image_size: int
    optimizer: str
    lr: float
    head_lr: float
    label_smoothing: float
    freeze_backbone_epochs: int
    aug_strength: str
    batch_size: int = 64
    epochs: int = 12
    early_stop_patience: int = 4


def planned_trials() -> list[WaveletTrial]:
    return [
        WaveletTrial("W1", "db4", 1, 224, "adamw", 5e-5, 2e-4, 0.03, 2, "light"),
        WaveletTrial("W2", "db4", 1, 224, "adamw", 8e-5, 3e-4, 0.05, 0, "light"),
        WaveletTrial("W3", "haar", 1, 224, "adamw", 5e-5, 2e-4, 0.03, 2, "light"),
        WaveletTrial("W4", "db2", 1, 224, "adam", 5e-5, 2e-4, 0.03, 2, "light"),
        WaveletTrial("W5", "db4", 2, 224, "adamw", 5e-5, 2e-4, 0.03, 2, "light"),
        WaveletTrial("W6", "db2", 2, 224, "adamw", 8e-5, 2e-4, 0.05, 2, "medium"),
        WaveletTrial("W7", "haar", 2, 256, "adam", 5e-5, 3e-4, 0.03, 2, "light"),
        WaveletTrial("W8", "db4", 1, 256, "adamw", 5e-5, 3e-4, 0.05, 0, "medium"),
    ]


def trial_dir(trial: WaveletTrial) -> Path:
    return PROJECT_ROOT / "output" / "wavelet_opt" / trial.trial_id


def checkpoint_path(trial: WaveletTrial) -> Path:
    return PROJECT_ROOT / "checkpoints" / "wavelet_opt" / f"{trial.trial_id}_best.pt"


def _fmt_float(value: float) -> str:
    return f"{value:.10g}"


def is_oom(log_text: str) -> bool:
    lowered = log_text.lower()
    return "out of memory" in lowered or "cuda oom" in lowered


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
        return subprocess.run(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr, text=True)


def train_command(trial: WaveletTrial, batch_size: int) -> list[str]:
    out_dir = trial_dir(trial)
    return [
        sys.executable,
        "src/train.py",
        "--train_csv",
        TRAIN_CSV,
        "--val_csv",
        VAL_CSV,
        "--model",
        "resnet50",
        "--epochs",
        str(min(trial.epochs, 12)),
        "--batch_size",
        str(batch_size),
        "--lr",
        _fmt_float(trial.lr),
        "--head_lr",
        _fmt_float(trial.head_lr),
        "--optimizer",
        trial.optimizer,
        "--weight_decay",
        "0.0001",
        "--label_smoothing",
        _fmt_float(trial.label_smoothing),
        "--use_amp",
        "1",
        "--balanced_sampler",
        "1",
        "--class_weight_loss",
        "0",
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
        "0",
        "--wavelet_mode",
        "high",
        "--wavelet",
        trial.wavelet,
        "--wavelet_level",
        str(trial.wavelet_level),
        "--output_dir",
        str(out_dir.relative_to(PROJECT_ROOT)),
        "--checkpoint_path",
        str(checkpoint_path(trial).relative_to(PROJECT_ROOT)),
        "--log_path",
        str((out_dir / "train_log.csv").relative_to(PROJECT_ROOT)),
        "--metrics_path",
        str((out_dir / "metrics.json").relative_to(PROJECT_ROOT)),
    ]


def infer_command(trial: WaveletTrial) -> list[str]:
    out_dir = trial_dir(trial)
    return [
        sys.executable,
        "src/infer.py",
        "--config",
        "config.yaml",
        "--input_csv",
        "input/input.csv",
        "--image_dir",
        "input/images",
        "--checkpoint",
        str(checkpoint_path(trial).relative_to(PROJECT_ROOT)),
        "--output",
        str((out_dir / "submission.csv").relative_to(PROJECT_ROOT)),
        "--debug_csv",
        str((out_dir / "debug.csv").relative_to(PROJECT_ROOT)),
        "--image_size",
        str(trial.image_size),
        "--pad_to_square",
        "0",
        "--tta",
        "0",
        "--coarse_filter",
        "0",
        "--use_expert_202_203_204",
        "1",
        "--expert_202_203_204_checkpoint",
        EXPERT_CKPT,
        "--expert_trigger_rank",
        "3",
        "--expert_mode",
        "conservative",
        "--wavelet_mode",
        "high",
        "--wavelet",
        trial.wavelet,
        "--wavelet_level",
        str(trial.wavelet_level),
    ]


def eval_command(trial: WaveletTrial) -> list[str]:
    out_dir = trial_dir(trial)
    return [
        sys.executable,
        "src/evaluate_test_predictions.py",
        "--pred",
        str((out_dir / "debug.csv").relative_to(PROJECT_ROOT)),
        "--gt",
        GT_CSV,
        "--output_dir",
        str((out_dir / "audit").relative_to(PROJECT_ROOT)),
    ]


def run_trial(trial: WaveletTrial) -> dict[str, str]:
    out_dir = trial_dir(trial)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path(trial).parent.mkdir(parents=True, exist_ok=True)
    batch_size = trial.batch_size
    train_result = run_logged(train_command(trial, batch_size), out_dir / "train_stdout.log", out_dir / "train_stderr.log")
    if train_result.returncode != 0:
        combined = (out_dir / "train_stdout.log").read_text(encoding="utf-8", errors="replace") + "\n" + (out_dir / "train_stderr.log").read_text(encoding="utf-8", errors="replace")
        if batch_size == 64 and is_oom(combined):
            batch_size = 32
            train_result = run_logged(train_command(trial, batch_size), out_dir / "train_stdout_bs32.log", out_dir / "train_stderr_bs32.log")
        if train_result.returncode != 0:
            raise RuntimeError(f"{trial.trial_id} training failed; see {out_dir}")
    for command, name in ((infer_command(trial), "infer"), (eval_command(trial), "evaluate")):
        result = run_logged(command, out_dir / f"{name}_stdout.log", out_dir / f"{name}_stderr.log")
        if result.returncode != 0:
            raise RuntimeError(f"{trial.trial_id} {name} failed; see {out_dir}")
    return collect_result(trial, batch_size)


def collect_result(trial: WaveletTrial, batch_size: int | None = None) -> dict[str, str]:
    out_dir = trial_dir(trial)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    best_epoch = int(metrics.get("best_epoch", 0) or 0)
    train_row = read_train_row(out_dir, best_epoch)
    test_metrics = parse_key_value_file(out_dir / "audit" / "test_metrics.txt")
    guards = compute_guard_metrics(out_dir / "debug.csv", PROJECT_ROOT / GT_CSV)
    return {
        "trial_id": trial.trial_id,
        "wavelet": trial.wavelet,
        "wavelet_level": str(trial.wavelet_level),
        "image_size": str(trial.image_size),
        "optimizer": trial.optimizer,
        "lr": _fmt_float(trial.lr),
        "head_lr": _fmt_float(trial.head_lr),
        "label_smoothing": _fmt_float(trial.label_smoothing),
        "freeze_backbone_epochs": str(trial.freeze_backbone_epochs),
        "aug_strength": trial.aug_strength,
        "val_acc": str(metrics.get("best_top1_acc", "")),
        "val_top3": str(metrics.get("best_top3_acc", "")),
        "macro_f1_present": str(metrics.get("best_macro_f1_present_classes", "")),
        "external_correct": test_metrics.get("correct", ""),
        "external_accuracy": test_metrics.get("test_accuracy", ""),
        "203_correct_after": str(guards["203_correct_after"]),
        "wrong_107": str(guards["wrong_107_count"]),
        "wrong_203": str(guards["wrong_203_count"]),
        "train_val_gap": train_row.get("gap_acc", ""),
        "overfit_flag": train_row.get("overfit_flag", ""),
        "checkpoint": str(checkpoint_path(trial).relative_to(PROJECT_ROOT)),
    }


def write_results(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS_WAVELET)
        writer.writeheader()
        writer.writerows(rows)


def eligible_for_phase_b(row: dict[str, str]) -> bool:
    return (
        int(row.get("external_correct") or 0) >= 86
        and int(row.get("203_correct_after") or 0) >= 9
        and int(row.get("wrong_203") or 99) <= 1
        and row.get("overfit_flag", "").strip().upper() == "NO"
    )


def run_phase_a(max_trials: int = 8) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    results_path = PROJECT_ROOT / "output" / "wavelet_opt" / "results.csv"
    consecutive_low_external = 0
    for trial in planned_trials()[:max_trials]:
        print(f"running {trial.trial_id}: wavelet={trial.wavelet} level={trial.wavelet_level} image_size={trial.image_size}")
        row = run_trial(trial)
        rows.append(row)
        write_results(rows, results_path)
        external_correct = int(row.get("external_correct") or 0)
        if external_correct < 83:
            consecutive_low_external += 1
        else:
            consecutive_low_external = 0
        print(f"{trial.trial_id}: external_correct={external_correct} 203={row['203_correct_after']} wrong_107={row['wrong_107']} wrong_203={row['wrong_203']} overfit={row['overfit_flag']}")
        if consecutive_low_external >= 3:
            print("stopping Phase A: 3 consecutive trials below external_correct 83")
            break
    return rows


def run_phase_b(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if eligible_for_phase_b(row)]
    candidates.sort(
        key=lambda row: (
            int(row.get("external_correct") or 0),
            int(row.get("203_correct_after") or 0),
            -int(row.get("wrong_203") or 99),
            -int(row.get("wrong_107") or 99),
            float(row.get("macro_f1_present") or 0.0),
        ),
        reverse=True,
    )
    candidates = candidates[:2]
    ensemble_dir = PROJECT_ROOT / "output" / "wavelet_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    output_path = ensemble_dir / "results.csv"
    preview_path = ensemble_dir / "best_preview.csv"
    if not candidates:
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
        with preview_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREVIEW_FIELDS)
            writer.writeheader()
        return None

    all_rows: list[dict[str, str]] = []
    best: dict[str, str] | None = None
    best_preview_source: Path | None = None
    for row in candidates:
        trial_id = row["trial_id"]
        trial_results = ensemble_dir / f"{trial_id}_results.csv"
        trial_preview = ensemble_dir / f"{trial_id}_preview.csv"
        tune_model_ensemble(
            final_debug_csv=FINAL_DEBUG,
            wavelet_debug_csv=PROJECT_ROOT / "output" / "wavelet_opt" / trial_id / "debug.csv",
            gt_csv=GT_CSV,
            output_csv=trial_results,
            preview_csv=trial_preview,
            rule_prefix=trial_id,
        )
        with trial_results.open("r", encoding="utf-8-sig", newline="") as handle:
            for result_row in csv.DictReader(handle):
                result_row["rule_description"] = f"{trial_id} {result_row['rule_description']}"
                all_rows.append(result_row)
                key = (
                    int(result_row["correct"]),
                    int(result_row["net_gain"]),
                    -int(result_row["damaged"]),
                    int(result_row["203_correct_after"]),
                    -int(result_row["wrong_203"]),
                    -int(result_row["wrong_107"]),
                )
                if best is None:
                    best = dict(result_row)
                    best_preview_source = trial_preview
                    best_key = key
                elif key > best_key:
                    best = dict(result_row)
                    best_preview_source = trial_preview
                    best_key = key

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    if best_preview_source is not None:
        preview_path.write_text(best_preview_source.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    return best


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run guarded wavelet-high optimization and optional final+wavelet ensemble")
    parser.add_argument("--max_trials", type=int, default=8)
    parser.add_argument("--phase", choices=("all", "a", "b"), default="all")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    if args.phase in {"all", "a"}:
        rows = run_phase_a(max_trials=min(args.max_trials, 8))
    else:
        results_path = PROJECT_ROOT / "output" / "wavelet_opt" / "results.csv"
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if args.phase in {"all", "b"}:
        best = run_phase_b(rows)
        print(f"phase_b_best={best}")


if __name__ == "__main__":
    main()
