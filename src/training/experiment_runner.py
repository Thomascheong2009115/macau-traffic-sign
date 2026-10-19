"""Experiment runner: executes training experiments, updates leaderboard, applies decision rules.

Each experiment gets a run_dir under outputs/experiments/<run_name>/.
Leaderboard is maintained at outputs/experiments/leaderboard.csv.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.train import train_model
from src.utils.logging import save_json, timestamp

LEADERBOARD_FIELDS = [
    "run_name", "backbone", "img_size", "loss_fn", "augmentation",
    "attention", "dropout", "activation",
    "accuracy", "macro_f1", "macro_f1_present", "weighted_f1",
    "best_epoch", "train_val_gap", "wrong_count", "notes",
]


def _parse_args_to_leaderboard_row(run_name: str, args, result: Dict, duration_s: float) -> Dict[str, Any]:
    """Convert training args + result to a leaderboard row."""
    return {
        "run_name": run_name,
        "backbone": args.backbone,
        "img_size": args.image_size,
        "loss_fn": args.loss,
        "augmentation": args.aug_strength,
        "attention": getattr(args, "attention", "none"),
        "dropout": getattr(args, "dropout", 0.0),
        "activation": getattr(args, "activation", "relu"),
        "accuracy": f"{result.get('top1', 0):.4f}",
        "macro_f1": f"{result.get('macro_f1_all', 0):.4f}",
        "macro_f1_present": f"{result.get('macro_f1_present', 0):.4f}",
        "weighted_f1": f"{result.get('weighted_f1', 0):.4f}",
        "best_epoch": result.get("best_epoch", 0),
        "train_val_gap": "0.0000",
        "wrong_count": result.get("wrong_count", "?"),
        "notes": f"Duration={duration_s:.0f}s",
    }


def update_leaderboard(leaderboard_path: Path, row: Dict[str, Any]) -> None:
    """Append or update a row in the leaderboard CSV."""
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if leaderboard_path.exists():
        with leaderboard_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("run_name") != row["run_name"]]
    rows.append(row)

    with leaderboard_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        # Sort by macro_f1_present descending
        rows.sort(key=lambda r: float(r.get("macro_f1_present", 0)), reverse=True)
        writer.writerows(rows)


def run_experiment(
    run_name: str,
    extra_args: Optional[list[str]] = None,
    leaderboard_path: str | Path = "outputs/experiments/leaderboard.csv",
) -> Dict:
    """Run a single experiment with given parameters.

    Args:
        run_name: experiment name (becomes output dir)
        extra_args: list of CLI args to override defaults
        leaderboard_path: path to leaderboard CSV
    """
    from src.training.train import create_arg_parser

    parser = create_arg_parser()
    default_args = [
        "--output_dir", f"outputs/experiments/{run_name}",
        "--train_csv", "outputs/manifests/train.csv",
        "--val_csv", "outputs/manifests/val.csv",
    ]

    all_args = default_args + (extra_args or [])
    args = parser.parse_args(all_args)

    print(f"\n{'='*60}")
    print(f"Experiment: {run_name}")
    print(f"Args: {vars(args)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        result = train_model(args)
        duration = time.time() - t0
        row = _parse_args_to_leaderboard_row(run_name, args, result, duration)
        update_leaderboard(Path(leaderboard_path), row)
        print(f"\nExperiment {run_name} complete: {result}")
    except Exception as e:
        print(f"\nExperiment {run_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment")
    parser.add_argument("--run_name", required=True, help="Experiment name")
    parser.add_argument("--extra_args", nargs="*", default=[], help="Extra CLI args for train.py")
    args = parser.parse_args()
    run_experiment(args.run_name, args.extra_args)


if __name__ == "__main__":
    main()
