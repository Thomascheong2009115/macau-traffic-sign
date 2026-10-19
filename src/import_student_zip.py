import argparse
import csv
import hashlib
import re
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.class_filter import is_valid_class, load_classes


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_CLASS_RE = re.compile(r"(?:^|_)(\d{3})(?:_|\.|$)")


@dataclass(frozen=True)
class StudentImportSummary:
    batch_id: str
    extract_dir: Path
    manifest_csv: Path
    report_csv: Path
    total_images_found: int
    imported_images: int
    duplicate_skipped: int
    invalid_class_skipped: int
    csv_rows: int
    images_missing_csv: int
    classes_with_new_images: dict[str, int]


def _safe_extract(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = (output_dir / info.filename).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"unsafe zip member path: {info.filename}")
            archive.extract(info, output_dir)


def _normal_name(path_text: str) -> str:
    return Path(str(path_text).strip().replace("\\", "/")).name.strip()


def _read_csv_labels(csv_paths: list[Path]) -> tuple[dict[str, str], int]:
    labels: dict[str, str] = {}
    row_count = 0
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            field_map = {field.strip().lower(): field for field in reader.fieldnames}
            img_field = field_map.get("img") or field_map.get("image") or field_map.get("filename") or field_map.get("file")
            class_field = field_map.get("class") or field_map.get("label")
            if not img_field:
                continue
            for row in reader:
                row_count += 1
                image_name = _normal_name(row.get(img_field, ""))
                if not image_name:
                    continue
                label = row.get(class_field, "").strip() if class_field else ""
                labels.setdefault(image_name.lower(), label)
    return labels, row_count


def _class_from_filename(path: Path) -> str:
    match = FILENAME_CLASS_RE.search(path.name)
    return match.group(1) if match else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _existing_student_identity(student_root: Path) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    hashes: set[str] = set()
    if not student_root.exists():
        return names, hashes
    for image_path in student_root.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            names.add(image_path.name.lower())
            hashes.add(_sha256(image_path))
    return names, hashes


def _write_report(path: Path, summary_rows: list[tuple[str, object]], class_counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        for key, value in summary_rows:
            writer.writerow(["summary", key, value])
        for label, count in class_counts.most_common():
            writer.writerow(["class_count", label, count])


def import_student_zip(
    zip_path: str | Path,
    project_root: str | Path | None = None,
    timestamp: str | None = None,
    classes_path: str | Path = "classes.txt",
) -> StudentImportSummary:
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    zip_path = Path(zip_path)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"student_batch_{timestamp}"
    extract_dir = project_root / "data" / "imports" / batch_id
    manifest_csv = project_root / "data" / "manifests" / f"{batch_id}.csv"
    report_csv = project_root / "output" / f"student_import_report_{timestamp}.csv"
    student_root = project_root / "data" / "real_student_images"
    classes = load_classes(project_root / classes_path if not Path(classes_path).is_absolute() else classes_path)

    _safe_extract(zip_path, extract_dir)
    csv_paths = sorted(path for path in extract_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".csv")
    image_paths = sorted(path for path in extract_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    csv_labels, csv_rows = _read_csv_labels(csv_paths)
    existing_names, existing_hashes = _existing_student_identity(student_root)
    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    output_rows: list[list[str]] = []
    class_counts: Counter[str] = Counter()
    duplicate_skipped = 0
    invalid_class_skipped = 0
    images_missing_csv = 0

    for image_path in image_paths:
        original_name = image_path.name.strip()
        csv_label = csv_labels.get(original_name.lower(), "").strip()
        label = csv_label or _class_from_filename(image_path)
        if not csv_label:
            images_missing_csv += 1
        if not is_valid_class(label, classes):
            invalid_class_skipped += 1
            continue
        name_key = original_name.lower()
        if name_key in existing_names or name_key in seen_names:
            duplicate_skipped += 1
            continue
        digest = _sha256(image_path)
        if digest in existing_hashes or digest in seen_hashes:
            duplicate_skipped += 1
            continue
        target_dir = student_root / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / original_name
        shutil.copy2(image_path, target)
        existing_names.add(name_key)
        seen_names.add(name_key)
        seen_hashes.add(digest)
        class_counts[label] += 1
        output_rows.append(
            [
                _relative_posix(project_root, target),
                label,
                "real_student_images",
                batch_id,
                original_name,
                digest,
            ]
        )

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source", "batch_id", "original_name", "sha256"])
        writer.writerows(output_rows)

    summary_rows = [
        ("total_images_found", len(image_paths)),
        ("imported_images", len(output_rows)),
        ("duplicate_skipped", duplicate_skipped),
        ("invalid_class_skipped", invalid_class_skipped),
        ("csv_rows", csv_rows),
        ("images_missing_csv", images_missing_csv),
        ("classes_with_new_images", len(class_counts)),
    ]
    _write_report(report_csv, summary_rows, class_counts)
    return StudentImportSummary(
        batch_id=batch_id,
        extract_dir=extract_dir,
        manifest_csv=manifest_csv,
        report_csv=report_csv,
        total_images_found=len(image_paths),
        imported_images=len(output_rows),
        duplicate_skipped=duplicate_skipped,
        invalid_class_skipped=invalid_class_skipped,
        csv_rows=csv_rows,
        images_missing_csv=images_missing_csv,
        classes_with_new_images=dict(class_counts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="import a new student traffic-sign image zip into real_student_images")
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--classes", default="classes.txt")
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    summary = import_student_zip(args.zip_path, classes_path=args.classes, timestamp=args.timestamp)
    print(f"python_import_batch={summary.batch_id}")
    print(f"total_images_found={summary.total_images_found}")
    print(f"imported_images={summary.imported_images}")
    print(f"duplicate_skipped={summary.duplicate_skipped}")
    print(f"invalid_class_skipped={summary.invalid_class_skipped}")
    print(f"csv_rows={summary.csv_rows}")
    print(f"images_missing_csv={summary.images_missing_csv}")
    print(f"classes_with_new_images={len(summary.classes_with_new_images)}")
    print(f"manifest_csv={summary.manifest_csv.as_posix()}")
    print(f"report_csv={summary.report_csv.as_posix()}")


if __name__ == "__main__":
    main()
