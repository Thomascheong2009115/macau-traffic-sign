import argparse
import csv
from pathlib import Path


DISPLAY_FIELDS = [
    "epoch",
    "train_top1_acc",
    "val_top1_acc",
    "train_loss",
    "val_loss",
    "val_macro_f1_present_classes",
    "score",
    "overfit_flag",
]


def load_tail_rows(log_path: str | Path, tail: int = 10) -> list[dict[str, str]]:
    """中文说明：读取训练日志最近 N 个 epoch，用于训练时快速查看泛化状态。"""
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"train log not found: {log_path}")
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-tail:]


def has_overfit_warning(rows: list[dict[str, str]]) -> bool:
    return any(row.get("overfit_flag", "").strip().upper() == "YES" for row in rows)


def print_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No training log rows found.")
        return
    print("\t".join(DISPLAY_FIELDS))
    for row in rows:
        print("\t".join(row.get(field, "") for field in DISPLAY_FIELDS))
    if has_overfit_warning(rows):
        print("WARNING: possible overfitting detected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="show recent training log rows")
    parser.add_argument("--log", default="output/train_log.csv")
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()
    print_rows(load_tail_rows(args.log, args.tail))


if __name__ == "__main__":
    main()
