import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


FORBIDDEN_VAL_SOURCES = {"synthetic", "pdf", "diffusion", "augmented_real"}


@dataclass(frozen=True)
class SplitSummary:
    train_count: int
    val_count: int
    warnings: list[str]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label", "source"]:
            raise ValueError("real_manifest.csv columns must be path,label,source")
        rows = [dict(row) for row in reader]
    for row_number, row in enumerate(rows, start=2):
        source = row["source"].strip()
        if source != "real":
            raise ValueError(f"manifest row {row_number}: only source=real is allowed, got {source}")
    return rows


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def split_real_dataset(
    manifest: str | Path = "data/manifests/real_manifest.csv",
    train_output: str | Path = "splits/train_real.csv",
    val_output: str | Path = "splits/val_real.csv",
    report_output: str | Path = "output/split_report.txt",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> SplitSummary:
    """中文说明：按类别分层划分真实照片；单样本类别只能放入 train 并记录 warning。"""
    manifest = Path(manifest)
    rows = _read_manifest(manifest)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    for label in sorted(grouped):
        items = grouped[label][:]
        rng.shuffle(items)
        if len(items) == 1:
            train_rows.extend(items)
            warnings.append(f"label {label}: single sample kept in train only")
            continue
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_count = min(val_count, len(items) - 1)
        val_rows.extend(items[:val_count])
        train_rows.extend(items[val_count:])

    if any(row["source"] in FORBIDDEN_VAL_SOURCES or row["source"] != "real" for row in val_rows):
        raise ValueError("validation split contains non-real samples")

    train_rows = sorted(train_rows, key=lambda row: (row["label"], row["path"]))
    val_rows = sorted(val_rows, key=lambda row: (row["label"], row["path"]))
    _write_rows(Path(train_output), train_rows)
    _write_rows(Path(val_output), val_rows)

    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"total={len(rows)}\n")
        handle.write(f"train={len(train_rows)}\n")
        handle.write(f"val={len(val_rows)}\n")
        handle.write(f"val_ratio={val_ratio}\n")
        handle.write(f"classes={len(grouped)}\n")
        handle.write("warnings:\n")
        for warning in warnings:
            handle.write(f"- {warning}\n")

    return SplitSummary(train_count=len(train_rows), val_count=len(val_rows), warnings=warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description="split real manifest into train/val CSV files")
    parser.add_argument("--manifest", default="data/manifests/real_manifest.csv")
    parser.add_argument("--train_output", default="splits/train_real.csv")
    parser.add_argument("--val_output", default="splits/val_real.csv")
    parser.add_argument("--report", default="output/split_report.txt")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = split_real_dataset(args.manifest, args.train_output, args.val_output, args.report, args.val_ratio, args.seed)
    print(f"train={summary.train_count}")
    print(f"val={summary.val_count}")
    print(f"warnings={len(summary.warnings)}")


if __name__ == "__main__":
    main()
