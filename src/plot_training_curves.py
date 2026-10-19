import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_log(log_path: str | Path) -> list[dict[str, str]]:
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"train log not found: {log_path}")
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def plot_training_curves(log_path: str | Path = "output/train_log.csv", output_path: str | Path = "output/training_curves.png") -> None:
    """中文说明：输出 4 宫格曲线，便于观察 loss、acc、macro-F1 和 score。"""
    rows = _read_log(log_path)
    if not rows:
        raise ValueError("train log has no rows")
    epochs = [int(row["epoch"]) for row in rows]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, _as_float(rows, "train_loss"), label="train_loss")
    axes[0, 0].plot(epochs, _as_float(rows, "val_loss"), label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, _as_float(rows, "train_top1_acc"), label="train_top1_acc")
    axes[0, 1].plot(epochs, _as_float(rows, "val_top1_acc"), label="val_top1_acc")
    axes[0, 1].set_title("Top1 Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, _as_float(rows, "train_macro_f1_present_classes"), label="train_f1_present")
    axes[1, 0].plot(epochs, _as_float(rows, "val_macro_f1_all_classes"), label="val_f1_all")
    axes[1, 0].plot(epochs, _as_float(rows, "val_macro_f1_present_classes"), label="val_f1_present")
    axes[1, 0].plot(epochs, _as_float(rows, "val_weighted_f1"), label="val_weighted_f1")
    axes[1, 0].set_title("Macro F1")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, _as_float(rows, "score"), label="score")
    axes[1, 1].set_title("Validation Score")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="plot training curves from train_log.csv")
    parser.add_argument("--log", default="output/train_log.csv")
    parser.add_argument("--output", default="output/training_curves.png")
    args = parser.parse_args()
    plot_training_curves(args.log, args.output)
    print(f"training_curves={Path(args.output).as_posix()}")


if __name__ == "__main__":
    main()
