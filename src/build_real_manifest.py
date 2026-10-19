import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.class_filter import is_valid_class, load_classes


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class ManifestSummary:
    total_images: int
    classes_with_images: int
    missing_classes: int
    low_sample_classes: int
    counts: dict[str, int]
    invalid_classes_skipped: int = 0
    output_csv: Path | None = None


def _relative_posix(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _iter_image_files(label_dir: Path) -> Iterable[Path]:
    for item in sorted(label_dir.rglob("*")):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            yield item


def _status_for_count(count: int) -> str:
    if count == 0:
        return "missing"
    if count == 1:
        return "danger"
    if count < 3:
        return "low"
    return "ok"


def build_real_manifest(
    root: str | Path = "data/real_images",
    output: str | Path = "data/manifests/real_manifest.csv",
    classes_path: str | Path = "classes.txt",
    class_counts_path: str | Path = "output/class_counts.csv",
    low_sample_path: str | Path = "output/low_sample_classes.csv",
    source: str = "real",
) -> ManifestSummary:
    """中文说明：扫描真实照片类别文件夹，生成显式白名单内的 real manifest。"""
    root = Path(root)
    output = Path(output)
    class_counts_path = Path(class_counts_path)
    low_sample_path = Path(low_sample_path)
    project_root = root.resolve().parents[1] if root.parent.name == "data" else Path.cwd()
    classes = load_classes(classes_path)

    if not root.exists():
        raise FileNotFoundError(f"real image root not found: {root}")

    invalid_dirs = []
    for label_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        label = label_dir.name.strip()
        if not (label.isdigit() and len(label) == 3) or not is_valid_class(label, classes):
            invalid_dirs.append(str(label_dir))
    if invalid_dirs and source == "real":
        raise ValueError("invalid class folders under data/real_images: " + ", ".join(invalid_dirs))

    rows: list[tuple[str, str, str]] = []
    counts = Counter({label: 0 for label in classes})
    for label in classes:
        label_dir = root / label
        if not label_dir.exists():
            continue
        for image_path in _iter_image_files(label_dir):
            rows.append((_relative_posix(image_path, project_root), label, source))
            counts[label] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source"])
        writer.writerows(rows)

    class_counts_path.parent.mkdir(parents=True, exist_ok=True)
    with class_counts_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "count"])
        for label in classes:
            writer.writerow([label, counts[label]])

    low_sample_path.parent.mkdir(parents=True, exist_ok=True)
    with low_sample_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "count", "status"])
        for label in classes:
            writer.writerow([label, counts[label], _status_for_count(counts[label])])

    summary = ManifestSummary(
        total_images=len(rows),
        classes_with_images=sum(1 for label in classes if counts[label] > 0),
        missing_classes=sum(1 for label in classes if counts[label] == 0),
        low_sample_classes=sum(1 for label in classes if counts[label] < 3),
        counts=dict(counts),
        invalid_classes_skipped=len(invalid_dirs),
        output_csv=output,
    )
    return summary


def _print_summary(summary: ManifestSummary, source: str) -> None:
    sorted_counts = sorted(summary.counts.items(), key=lambda item: item[0])
    rare_counts = sorted(summary.counts.items(), key=lambda item: (item[1], item[0]))
    if source == "real_student_images":
        print(f"student_images_total={summary.total_images}")
        print(f"classes_with_student_images={summary.classes_with_images}")
        print(f"low_sample_student_classes={summary.low_sample_classes}")
        print(f"invalid_classes_skipped={summary.invalid_classes_skipped}")
        print(f"output_csv={summary.output_csv.as_posix() if summary.output_csv else ''}")
    else:
        print(f"total_images={summary.total_images}")
        print(f"classes_with_images={summary.classes_with_images}")
        print(f"missing_classes={summary.missing_classes}")
        print(f"classes_below_3={summary.low_sample_classes}")
    print("first10_counts=" + ",".join(f"{label}:{count}" for label, count in sorted_counts[:10]))
    print("lowest10_counts=" + ",".join(f"{label}:{count}" for label, count in rare_counts[:10]))


def main() -> None:
    parser = argparse.ArgumentParser(description="build real traffic sign manifest")
    parser.add_argument("--root", default="data/real_images")
    parser.add_argument("--output", default="data/manifests/real_manifest.csv")
    parser.add_argument("--classes", default="classes.txt")
    parser.add_argument("--class_counts", default="output/class_counts.csv")
    parser.add_argument("--low_samples", default="output/low_sample_classes.csv")
    parser.add_argument("--source", default="real")
    args = parser.parse_args()
    summary = build_real_manifest(args.root, args.output, args.classes, args.class_counts, args.low_samples, args.source)
    _print_summary(summary, args.source)


if __name__ == "__main__":
    main()
