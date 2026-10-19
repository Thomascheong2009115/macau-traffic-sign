import argparse
import csv
import math
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


SOURCE = "augmented_hardcase"
FORBIDDEN_AUG_CLASSES = {"107", "203"}
NEED_COLLECTION_CLASSES = {"119", "314"}
HIGH_PRIORITY_CLASSES = {"105", "110", "113", "138", "210", "204", "213", "201", "208"}
MEDIUM_PRIORITY_CLASSES = {"312", "317"}
LOW_PRIORITY_CLASSES = {"121"}
DEFAULT_TARGET_CLASSES = sorted(HIGH_PRIORITY_CLASSES | MEDIUM_PRIORITY_CLASSES | LOW_PRIORITY_CLASSES | NEED_COLLECTION_CLASSES)


@dataclass(frozen=True)
class HardcaseAugSummary:
    input_images: int
    base_train_images: int
    planned_augmented_images: int
    max_augmented_images: int
    generated_augmented_images: int
    counts: dict[str, int]
    skipped_forbidden: dict[str, int]
    need_collection: list[str]
    output_csv: Path
    summary_csv: Path


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


def aug_count_for_current_count(current_count: int) -> int:
    if current_count <= 0:
        return 0
    if current_count <= 15:
        return 6
    if current_count <= 50:
        return 4
    if current_count <= 120:
        return 2
    return 1


def _read_target_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label", "source"]:
            raise ValueError("input_csv columns must be path,label,source")
        return [dict(row) for row in reader]


def _read_base_manifest_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label", "source"]:
            raise ValueError("base_manifest_csv columns must be path,label,source")
        return sum(1 for _ in reader)


def _write_need_collection(path: Path, labels: list[str], present_counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "status", "reason"])
        writer.writeheader()
        for label in labels:
            if present_counts[label] == 0:
                writer.writerow({"label": label, "status": "need_collection", "reason": "no current training samples"})


def _write_summary(
    path: Path,
    labels: list[str],
    present_counts: Counter[str],
    planned_counts: Counter[str],
    quota_counts: Counter[str],
    generated_counts: Counter[str],
    base_train_images: int,
    max_augmented_images: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "current_train_count",
                "num_aug_per_image",
                "planned_aug_count",
                "quota_aug_count",
                "generated_aug_count",
                "status",
                "reason",
                "base_train_images",
                "max_augmented_images",
            ],
        )
        writer.writeheader()
        for label in labels:
            current_count = present_counts[label]
            num_aug = aug_count_for_current_count(current_count)
            status = "generated"
            reason = ""
            if label in FORBIDDEN_AUG_CLASSES:
                status = "skipped_forbidden"
                reason = "over-predicted class"
                num_aug = 0
            elif label in NEED_COLLECTION_CLASSES and current_count == 0:
                status = "need_collection"
                reason = "no current training samples"
            elif generated_counts[label] < planned_counts[label]:
                status = "capped"
                reason = "global 25 percent cap"
            writer.writerow(
                {
                    "label": label,
                    "current_train_count": current_count,
                    "num_aug_per_image": num_aug,
                    "planned_aug_count": planned_counts[label],
                    "quota_aug_count": quota_counts[label],
                    "generated_aug_count": generated_counts[label],
                    "status": status,
                    "reason": reason,
                    "base_train_images": base_train_images,
                    "max_augmented_images": max_augmented_images,
                }
            )


def _cap_class_quotas(planned_counts: Counter[str], max_augmented_images: int) -> Counter[str]:
    total = sum(planned_counts.values())
    if total <= max_augmented_images:
        return Counter(planned_counts)
    if max_augmented_images <= 0:
        return Counter()

    quotas: Counter[str] = Counter()
    remainders: list[tuple[float, str]] = []
    for label, planned in planned_counts.items():
        exact = planned * max_augmented_images / total
        quota = int(math.floor(exact))
        if planned > 0 and quota == 0:
            quota = 1
        quotas[label] = min(quota, planned)
        remainders.append((exact - math.floor(exact), label))

    while sum(quotas.values()) > max_augmented_images:
        for _remainder, label in sorted(remainders, key=lambda item: (item[0], item[1])):
            if quotas[label] > 0:
                quotas[label] -= 1
                break

    while sum(quotas.values()) < max_augmented_images:
        changed = False
        for _remainder, label in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if quotas[label] < planned_counts[label]:
                quotas[label] += 1
                changed = True
                break
        if not changed:
            break
    return quotas


def generate_hardcase_aug(
    input_csv: str | Path = "data/manifests/hardcase_target_classes.csv",
    output_dir: str | Path = "data/augmented_hardcase",
    output_csv: str | Path = "data/manifests/augmented_hardcase_manifest.csv",
    base_manifest_csv: str | Path = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv",
    summary_csv: str | Path = "output/hardcase_aug_summary.csv",
    image_size: int = 224,
    aug_strength: str = "medium",
    overwrite: bool = False,
    classes_path: str | Path = "classes.txt",
    project_root: str | Path | None = None,
    need_collection_csv: str | Path = "data/manifests/augmented_hardcase_need_collection.csv",
) -> HardcaseAugSummary:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)
    base_manifest_csv = Path(base_manifest_csv)
    summary_csv = Path(summary_csv)
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    classes = load_classes(classes_path)
    rows = _read_target_rows(input_csv)
    base_train_images = _read_base_manifest_count(base_manifest_csv)
    max_augmented_images = int(base_train_images * 0.25)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    transform_strength = "light" if aug_strength == "conservative" else aug_strength
    transform = build_offline_augment_transform(image_size=image_size, aug_strength=transform_strength)
    output_rows: list[list[object]] = []
    generated_counts: Counter[str] = Counter()
    skipped_forbidden: Counter[str] = Counter()
    present_counts: Counter[str] = Counter(row["label"].strip() for row in rows)
    planned_counts: Counter[str] = Counter()
    labels_for_summary = sorted(set(DEFAULT_TARGET_CLASSES) | set(present_counts) | FORBIDDEN_AUG_CLASSES)

    candidates: list[tuple[int, int, dict[str, str], int]] = []
    for row_number, row in enumerate(rows, start=2):
        label = row["label"].strip()
        if label in FORBIDDEN_AUG_CLASSES:
            skipped_forbidden[label] += 1
            continue
        if label in NEED_COLLECTION_CLASSES:
            continue
        if label not in DEFAULT_TARGET_CLASSES:
            continue
        if not is_valid_class(label, classes):
            raise ValueError(f"row {row_number}: invalid label: {label}")
        current_count = present_counts[label]
        per_image_aug_count = aug_count_for_current_count(current_count)
        planned_counts[label] += per_image_aug_count
        if per_image_aug_count <= 0:
            continue
        candidates.append((current_count, row_number, row, per_image_aug_count))

    planned_augmented_images = sum(planned_counts.values())
    quota_counts = _cap_class_quotas(planned_counts, max_augmented_images)
    for _current_count, row_number, row, per_image_aug_count in sorted(candidates, key=lambda item: (item[0], item[2]["label"].strip(), item[1])):
        if len(output_rows) >= max_augmented_images:
            break
        label = row["label"].strip()
        if generated_counts[label] >= quota_counts[label]:
            continue
        parent_path = row["path"].strip().replace("\\", "/")
        image_path = _resolve_project_path(project_root, parent_path)
        if not image_path.exists():
            raise FileNotFoundError(f"row {row_number}: image not found: {image_path}")
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        stem = image_path.stem
        for aug_id in range(per_image_aug_count):
            if len(output_rows) >= max_augmented_images:
                break
            if generated_counts[label] >= quota_counts[label]:
                break
            augmented = transform(image)
            out_path = output_dir / label / f"{stem}_hc_aug{aug_id:03d}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            augmented.save(out_path, quality=92)
            output_rows.append(
                [
                    _relative_posix(project_root, out_path),
                    label,
                    SOURCE,
                    parent_path,
                    aug_id,
                ]
            )
            generated_counts[label] += 1

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source", "parent_path", "aug_id"])
        writer.writerows(output_rows)

    need_collection = sorted(label for label in NEED_COLLECTION_CLASSES if present_counts[label] == 0)
    _write_need_collection(Path(need_collection_csv), need_collection, present_counts)
    _write_summary(
        summary_csv,
        labels_for_summary,
        present_counts,
        planned_counts,
        quota_counts,
        generated_counts,
        base_train_images,
        max_augmented_images,
    )

    return HardcaseAugSummary(
        input_images=len(rows),
        base_train_images=base_train_images,
        planned_augmented_images=planned_augmented_images,
        max_augmented_images=max_augmented_images,
        generated_augmented_images=len(output_rows),
        counts=dict(generated_counts),
        skipped_forbidden=dict(skipped_forbidden),
        need_collection=need_collection,
        output_csv=output_csv,
        summary_csv=summary_csv,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="generate conservative hard-case augmentations")
    parser.add_argument("--input_csv", default="data/manifests/hardcase_target_classes.csv")
    parser.add_argument("--output_dir", default="data/augmented_hardcase")
    parser.add_argument("--output_csv", default="data/manifests/augmented_hardcase_manifest.csv")
    parser.add_argument("--base_manifest_csv", default="data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv")
    parser.add_argument("--summary_csv", default="output/hardcase_aug_summary.csv")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--aug_strength", default="medium", choices=["conservative", "medium"])
    parser.add_argument("--overwrite", type=int, choices=[0, 1], default=0)
    parser.add_argument("--classes", default="classes.txt")
    parser.add_argument("--need_collection_csv", default="data/manifests/augmented_hardcase_need_collection.csv")
    args = parser.parse_args()
    summary = generate_hardcase_aug(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        base_manifest_csv=args.base_manifest_csv,
        summary_csv=args.summary_csv,
        image_size=args.image_size,
        aug_strength=args.aug_strength,
        overwrite=bool(args.overwrite),
        classes_path=args.classes,
        need_collection_csv=args.need_collection_csv,
    )
    print(f"input_images={summary.input_images}")
    print(f"base_train_images={summary.base_train_images}")
    print(f"planned_augmented_images={summary.planned_augmented_images}")
    print(f"max_augmented_images={summary.max_augmented_images}")
    print(f"generated_augmented_images={summary.generated_augmented_images}")
    for label, count in sorted(summary.counts.items()):
        print(f"{label}_generated={count}")
    for label in summary.need_collection:
        print(f"{label}_status=need_collection")
    for label, count in sorted(summary.skipped_forbidden.items()):
        print(f"{label}_skipped_forbidden={count}")
    print(f"output_csv={summary.output_csv.as_posix()}")
    print(f"summary_csv={summary.summary_csv.as_posix()}")


if __name__ == "__main__":
    main()
