import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ALLOWED_SOURCES = {"real", "augmented_real", "real_student_images", "augmented_student"}


@dataclass(frozen=True)
class ExpertManifestSummary:
    counts: dict[str, int]
    total: int
    output_csv: Path


def parse_classes(value: str) -> list[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise ValueError("classes must not be empty")
    if any(not (label.isdigit() and len(label) == 3) for label in classes):
        raise ValueError("classes must be comma-separated three-digit labels")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must be unique")
    return classes


def _resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def build_expert_manifest(
    input_csv: str | Path = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv",
    output_csv: str | Path = "data/manifests/expert_202_203_204_train.csv",
    classes: str | list[str] = "202,203,204",
    project_root: str | Path | None = None,
) -> ExpertManifestSummary:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    target_classes = parse_classes(classes) if isinstance(classes, str) else list(classes)
    target_set = set(target_classes)

    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter({label: 0 for label in target_classes})
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label", "source"]:
            raise ValueError("input_csv columns must be path,label,source")
        for row_number, row in enumerate(reader, start=2):
            label = row["label"].strip()
            if label not in target_set:
                continue
            source = row["source"].strip()
            if source not in ALLOWED_SOURCES:
                raise ValueError(f"row {row_number}: unsupported source={source}")
            path_text = row["path"].strip().replace("\\", "/")
            if not _resolve_path(path_text, project_root).exists():
                raise FileNotFoundError(f"row {row_number}: path not found: {path_text}")
            rows.append({"path": path_text, "label": label, "source": source})
            counts[label] += 1

    for label in target_classes:
        if counts[label] == 0:
            raise ValueError(f"{label}_count is zero")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    summary = ExpertManifestSummary(counts=dict(counts), total=len(rows), output_csv=output_csv)
    for label in target_classes:
        print(f"{label}_count={counts[label]}")
    print(f"total={len(rows)}")
    print(f"output_csv={output_csv.as_posix()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="build 202/203/204 expert training manifest")
    parser.add_argument("--input_csv", default="data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv")
    parser.add_argument("--output_csv", default="data/manifests/expert_202_203_204_train.csv")
    parser.add_argument("--classes", default="202,203,204")
    args = parser.parse_args()
    build_expert_manifest(args.input_csv, args.output_csv, args.classes)


if __name__ == "__main__":
    main()
