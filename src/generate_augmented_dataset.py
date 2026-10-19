import argparse
import csv
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment import build_offline_augment_transform
from src.class_filter import is_valid_class, load_classes


@dataclass(frozen=True)
class AugmentSummary:
    original_train_images: int
    num_aug_per_image: int
    generated_augmented_images: int
    output_csv: Path
    source: str = "augmented_real"
    max_augmented_images: int | None = None


def _expected_input_source(output_source: str) -> str:
    if output_source == "augmented_real":
        return "real"
    if output_source == "augmented_student":
        return "real_student_images"
    raise ValueError(f"unsupported output source: {output_source}")


def _read_train_rows(input_csv: Path, output_source: str) -> list[dict[str, str]]:
    if "val" in input_csv.name.lower():
        raise ValueError("input_csv must be a training CSV; validation CSV is forbidden")
    expected_source = _expected_input_source(output_source)
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label", "source"]:
            raise ValueError("input_csv columns must be path,label,source")
        rows = [dict(row) for row in reader]
    for row_number, row in enumerate(rows, start=2):
        if row["source"].strip() != expected_source:
            raise ValueError(f"row {row_number}: source must be {expected_source} for {output_source}, got {row['source']}")
    return rows


def _read_student_batch_rows(input_csv: Path) -> list[dict[str, str]]:
    if "val" in input_csv.name.lower():
        raise ValueError("input_csv must be a student training CSV; validation CSV is forbidden")
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        allowed_fields = (
            ["path", "label", "source"],
            ["path", "label", "source", "batch_id", "original_name", "sha256"],
        )
        if reader.fieldnames not in allowed_fields:
            raise ValueError("student batch input_csv columns must be path,label,source or path,label,source,batch_id,original_name,sha256")
        rows = [dict(row) for row in reader]
    for row_number, row in enumerate(rows, start=2):
        if row["source"].strip() != "real_student_images":
            raise ValueError(f"row {row_number}: source must be real_student_images for augmented_student, got {row['source']}")
    return rows


def _count_manifest_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _resolve_project_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def _relative_posix(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def generate_augmented_dataset(
    input_csv: str | Path = "splits/train_real.csv",
    output_dir: str | Path = "data/augmented_real",
    output_csv: str | Path = "data/manifests/augmented_train_manifest.csv",
    num_aug: int = 10,
    image_size: int = 224,
    aug_strength: str = "medium",
    overwrite: bool = False,
    classes_path: str | Path = "classes.txt",
    project_root: str | Path | None = None,
    source: str = "augmented_real",
) -> AugmentSummary:
    """中文说明：只根据 train_real.csv 生成离线增强图，绝不读取验证集。"""
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    classes = load_classes(classes_path)
    rows = _read_train_rows(input_csv, source)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    transform = build_offline_augment_transform(image_size=image_size, aug_strength=aug_strength)
    output_rows: list[list[object]] = []
    for row_number, row in enumerate(rows, start=2):
        label = row["label"].strip()
        if not is_valid_class(label, classes):
            raise ValueError(f"row {row_number}: invalid label: {label}")
        parent_path = row["path"].strip()
        image_path = _resolve_project_path(project_root, parent_path)
        if not image_path.exists():
            raise FileNotFoundError(f"row {row_number}: image not found: {image_path}")
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        stem = image_path.stem
        for aug_id in range(num_aug):
            augmented = transform(image)
            out_path = output_dir / label / f"{stem}_aug{aug_id:02d}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            augmented.save(out_path, quality=92)
            output_rows.append(
                [
                    _relative_posix(project_root, out_path),
                    label,
                    source,
                    parent_path.replace("\\", "/"),
                    aug_id,
                ]
            )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source", "parent_path", "aug_id"])
        writer.writerows(output_rows)

    return AugmentSummary(
        original_train_images=len(rows),
        num_aug_per_image=num_aug,
        generated_augmented_images=len(output_rows),
        output_csv=output_csv,
        source=source,
    )


def generate_student_batch_augmented_dataset(
    input_csv: str | Path,
    base_manifest_csv: str | Path = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv",
    output_dir: str | Path = "data/augmented_student_batch",
    output_csv: str | Path = "data/manifests/augmented_student_batch.csv",
    default_num_aug: int = 8,
    low_sample_num_aug: int = 10,
    low_sample_threshold: int = 15,
    max_ratio: float = 0.30,
    image_size: int = 224,
    aug_strength: str = "light",
    overwrite: bool = False,
    classes_path: str | Path = "classes.txt",
    project_root: str | Path | None = None,
) -> AugmentSummary:
    input_csv = Path(input_csv)
    base_manifest_csv = Path(base_manifest_csv)
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    classes = load_classes(classes_path)
    rows = _read_student_batch_rows(input_csv)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    class_counts = Counter(row["label"].strip() for row in rows)
    base_count = _count_manifest_rows(base_manifest_csv)
    max_augmented_images = int(base_count * max_ratio)
    transform = build_offline_augment_transform(image_size=image_size, aug_strength=aug_strength)
    output_rows: list[list[object]] = []
    for row_number, row in enumerate(rows, start=2):
        label = row["label"].strip()
        if not is_valid_class(label, classes):
            raise ValueError(f"row {row_number}: invalid label: {label}")
        parent_path = row["path"].strip()
        image_path = _resolve_project_path(project_root, parent_path)
        if not image_path.exists():
            raise FileNotFoundError(f"row {row_number}: image not found: {image_path}")
        requested_aug = low_sample_num_aug if class_counts[label] < low_sample_threshold else default_num_aug
        remaining = max_augmented_images - len(output_rows)
        if remaining <= 0:
            break
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        stem = image_path.stem
        for aug_id in range(min(requested_aug, remaining)):
            augmented = transform(image)
            out_path = output_dir / label / f"{stem}_aug{aug_id:02d}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            augmented.save(out_path, quality=92)
            output_rows.append(
                [
                    _relative_posix(project_root, out_path),
                    label,
                    "augmented_student",
                    parent_path.replace("\\", "/"),
                    aug_id,
                ]
            )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source", "parent_path", "aug_id"])
        writer.writerows(output_rows)

    return AugmentSummary(
        original_train_images=len(rows),
        num_aug_per_image=default_num_aug,
        generated_augmented_images=len(output_rows),
        output_csv=output_csv,
        source="augmented_student",
        max_augmented_images=max_augmented_images,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="generate offline augmented images from train_real.csv only")
    parser.add_argument("--input_csv", default="splits/train_real.csv")
    parser.add_argument("--output_dir", default="data/augmented_real")
    parser.add_argument("--output_csv", default="data/manifests/augmented_train_manifest.csv")
    parser.add_argument("--num_aug", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--aug_strength", default="medium", choices=["light", "medium", "strong"])
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    parser.add_argument("--classes", default="classes.txt")
    parser.add_argument("--source", default="augmented_real", choices=["augmented_real", "augmented_student"])
    parser.add_argument("--student_batch_base_csv")
    parser.add_argument("--adaptive_student_batch", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.adaptive_student_batch:
        summary = generate_student_batch_augmented_dataset(
            input_csv=args.input_csv,
            base_manifest_csv=args.student_batch_base_csv or "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv",
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            default_num_aug=args.num_aug,
            image_size=args.image_size,
            aug_strength=args.aug_strength,
            overwrite=bool(args.overwrite),
            classes_path=args.classes,
        )
    else:
        summary = generate_augmented_dataset(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            num_aug=args.num_aug,
            image_size=args.image_size,
            aug_strength=args.aug_strength,
            overwrite=bool(args.overwrite),
            classes_path=args.classes,
            source=args.source,
        )
    print(f"original_train_images={summary.original_train_images}")
    print(f"num_aug_per_image={summary.num_aug_per_image}")
    print(f"generated_augmented_images={summary.generated_augmented_images}")
    if summary.max_augmented_images is not None:
        print(f"max_augmented_images={summary.max_augmented_images}")
    print(f"source={summary.source}")
    print(f"output_csv={summary.output_csv.as_posix()}")


if __name__ == "__main__":
    main()
