import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.class_filter import is_valid_class, load_classes


@dataclass(frozen=True)
class CombinedSummary:
    real_train_images: int
    augmented_images: int
    student_images: int
    augmented_student_images: int
    combined_train_images: int
    output_csv: Path


@dataclass(frozen=True)
class FullHardcaseCombinedSummary:
    base_train_images: int
    hardcase_augmented_images: int
    combined_train_images: int
    output_csv: Path


def _read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if "val" in path.name.lower():
        raise ValueError("validation CSV is forbidden in combined training manifest")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError(f"{path} columns must be {','.join(expected_fields)}")
        return [dict(row) for row in reader]


def _read_extra_csv(path: Path) -> list[dict[str, str]]:
    if "val" in path.name.lower():
        raise ValueError("validation CSV is forbidden in combined training manifest")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        allowed_fields = (
            ["path", "label", "source"],
            ["path", "label", "source", "parent_path", "aug_id"],
        )
        if reader.fieldnames not in allowed_fields:
            raise ValueError(f"{path} columns must be path,label,source or path,label,source,parent_path,aug_id")
        return [dict(row) for row in reader]


def _resolve(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def _validate_row(row: dict[str, str], row_number: int, classes: list[str], project_root: Path, expected_source: str) -> None:
    label = row["label"].strip()
    if not is_valid_class(label, classes):
        raise ValueError(f"row {row_number}: invalid label: {label}")
    if row["source"].strip() != expected_source:
        raise ValueError(f"row {row_number}: expected source={expected_source}, got {row['source']}")
    path = _resolve(project_root, row["path"].strip())
    if not path.exists():
        raise FileNotFoundError(f"row {row_number}: path not found: {path}")


def _normalize_extra_csv(extra_csv: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if extra_csv is None:
        return []
    if isinstance(extra_csv, (str, Path)):
        return [Path(extra_csv)]
    return [Path(item) for item in extra_csv]


def _write_three_column_manifest(rows: list[list[str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source"])
        writer.writerows(rows)


def build_combined_train_manifest(
    real_train_csv: str | Path = "splits/train_real.csv",
    aug_csv: str | Path = "data/manifests/augmented_train_manifest.csv",
    extra_csv: str | Path | Sequence[str | Path] | None = None,
    output_csv: str | Path = "data/manifests/train_real_plus_aug.csv",
    classes_path: str | Path = "classes.txt",
    project_root: str | Path | None = None,
) -> CombinedSummary:
    """中文说明：合并原始 train real 和 augmented_real；严禁加入 val_real。"""
    real_train_csv = Path(real_train_csv)
    aug_csv = Path(aug_csv)
    output_csv = Path(output_csv)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    classes = load_classes(classes_path)
    real_rows = _read_csv(real_train_csv, ["path", "label", "source"])
    aug_rows = _read_csv(aug_csv, ["path", "label", "source", "parent_path", "aug_id"])
    student_rows: list[dict[str, str]] = []
    augmented_student_rows: list[dict[str, str]] = []
    for extra_path in _normalize_extra_csv(extra_csv):
        rows = _read_extra_csv(extra_path)
        for row_number, row in enumerate(rows, start=2):
            source = row["source"].strip()
            if source == "real_student_images":
                _validate_row(row, row_number, classes, project_root, "real_student_images")
                student_rows.append(row)
            elif source == "augmented_student":
                _validate_row(row, row_number, classes, project_root, "augmented_student")
                augmented_student_rows.append(row)
            else:
                raise ValueError(f"{extra_path} row {row_number}: unsupported extra source={source}")

    combined_rows: list[list[str]] = []
    for row_number, row in enumerate(real_rows, start=2):
        _validate_row(row, row_number, classes, project_root, "real")
        combined_rows.append([row["path"].replace("\\", "/"), row["label"].strip(), "real"])
    for row_number, row in enumerate(aug_rows, start=2):
        _validate_row(row, row_number, classes, project_root, "augmented_real")
        combined_rows.append([row["path"].replace("\\", "/"), row["label"].strip(), "augmented_real"])
    for row_number, row in enumerate(student_rows, start=2):
        combined_rows.append([row["path"].replace("\\", "/"), row["label"].strip(), "real_student_images"])
    for row_number, row in enumerate(augmented_student_rows, start=2):
        combined_rows.append([row["path"].replace("\\", "/"), row["label"].strip(), "augmented_student"])

    _write_three_column_manifest(combined_rows, output_csv)

    return CombinedSummary(
        real_train_images=len(real_rows),
        augmented_images=len(aug_rows),
        student_images=len(student_rows),
        augmented_student_images=len(augmented_student_rows),
        combined_train_images=len(combined_rows),
        output_csv=output_csv,
    )


def build_full_plus_hardcase_manifest(
    base_csv: str | Path = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv",
    hardcase_aug_csv: str | Path = "data/manifests/augmented_hardcase_manifest.csv",
    output_csv: str | Path = "data/manifests/train_full_plus_hardcase_aug.csv",
    classes_path: str | Path = "classes.txt",
    project_root: str | Path | None = None,
) -> FullHardcaseCombinedSummary:
    base_csv = Path(base_csv)
    hardcase_aug_csv = Path(hardcase_aug_csv)
    output_csv = Path(output_csv)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    classes = load_classes(classes_path)
    base_rows = _read_csv(base_csv, ["path", "label", "source"])
    hardcase_rows = _read_csv(hardcase_aug_csv, ["path", "label", "source", "parent_path", "aug_id"])

    combined_rows: list[list[str]] = []
    for row_number, row in enumerate(base_rows, start=2):
        source = row["source"].strip()
        if source not in {"real", "augmented_real", "real_student_images", "augmented_student"}:
            raise ValueError(f"{base_csv} row {row_number}: unsupported base source={source}")
        _validate_row(row, row_number, classes, project_root, source)
        combined_rows.append([row["path"].replace("\\", "/"), row["label"].strip(), source])

    for row_number, row in enumerate(hardcase_rows, start=2):
        _validate_row(row, row_number, classes, project_root, "augmented_hardcase")
        combined_rows.append([row["path"].replace("\\", "/"), row["label"].strip(), "augmented_hardcase"])

    _write_three_column_manifest(combined_rows, output_csv)
    return FullHardcaseCombinedSummary(
        base_train_images=len(base_rows),
        hardcase_augmented_images=len(hardcase_rows),
        combined_train_images=len(combined_rows),
        output_csv=output_csv,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="combine real train manifest and augmented train manifest")
    parser.add_argument("--real_train_csv", default="splits/train_real.csv")
    parser.add_argument("--aug_csv", default="data/manifests/augmented_train_manifest.csv")
    parser.add_argument("--extra_csv", nargs="*")
    parser.add_argument("--output_csv", default="data/manifests/train_real_plus_aug.csv")
    parser.add_argument("--classes", default="classes.txt")
    parser.add_argument("--base_csv")
    parser.add_argument("--hardcase_aug_csv")
    args = parser.parse_args()
    if args.base_csv or args.hardcase_aug_csv:
        if not args.base_csv or not args.hardcase_aug_csv:
            raise ValueError("--base_csv and --hardcase_aug_csv must be provided together")
        summary = build_full_plus_hardcase_manifest(args.base_csv, args.hardcase_aug_csv, args.output_csv, args.classes)
        print(f"base_train_images={summary.base_train_images}")
        print(f"hardcase_augmented_images={summary.hardcase_augmented_images}")
        print(f"combined_train_images={summary.combined_train_images}")
        print(f"output_csv={summary.output_csv.as_posix()}")
    else:
        summary = build_combined_train_manifest(args.real_train_csv, args.aug_csv, args.extra_csv, args.output_csv, args.classes)
        print(f"real_train_images={summary.real_train_images}")
        print(f"augmented_real_images={summary.augmented_images}")
        print(f"student_images={summary.student_images}")
        print(f"augmented_student_images={summary.augmented_student_images}")
        print(f"combined_train_images={summary.combined_train_images}")
        print(f"output_csv={summary.output_csv.as_posix()}")


if __name__ == "__main__":
    main()
