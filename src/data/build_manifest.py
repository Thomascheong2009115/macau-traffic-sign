"""Build a clean manifest CSV from raw image directories.

Scans student data directories, assigns source labels, computes SHA256.
Excludes any path containing teacher markers (real_images/, real_crops/, augmented_real/).
"""
import csv
import hashlib
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.image_io import validate_image

# Allowed source tags
SOURCE_STUDENT_REAL = "student_real"
SOURCE_SD_AUG = "sd_aug"

# Path markers that indicate teacher data (must not appear in image path)
TEACHER_PATH_MARKERS = [
    "real_images",
    "real_crops",
    "augmented_real",
]

# Directories to scan for student real data (images organized by class subdirectory)
STUDENT_REAL_DIRS = [
    "data/real_student_images",
]

# Directory to scan for student batch imports (with per-student CSVs)
STUDENT_IMPORT_DIR = "data/imports/student_batch_20260607_foreground_probe/students"

# File extensions to scan
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _sha256_hex(filepath: Path) -> str:
    """Compute SHA-256 hash of file for dedup."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_teacher_path(path: Path) -> bool:
    """Check if a path contains teacher data markers."""
    path_str = str(path).replace("\\", "/").lower()
    return any(marker.lower() in path_str for marker in TEACHER_PATH_MARKERS)


def _scan_directory(root: Path, project_root: Path) -> list[dict]:
    """Scan a directory recursively for images."""
    results = []
    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix not in IMAGE_EXTS and filepath.suffix.lower() not in IMAGE_EXTS:
            continue

        is_valid, reason = validate_image(filepath)
        if not is_valid:
            results.append({
                "path": str(filepath),
                "label": "",
                "source": "rejected",
                "parent_path": str(filepath.parent),
                "sha256": "",
                "reject_reason": reason,
            })
            continue

        # Extract label from parent directory name
        parent_dir = filepath.parent.name
        label = parent_dir if parent_dir.isdigit() and len(parent_dir) == 3 else ""

        if not label:
            results.append({
                "path": str(filepath),
                "label": "",
                "source": "rejected",
                "parent_path": str(filepath.parent),
                "sha256": "",
                "reject_reason": "no_label_from_parent",
            })
            continue

        sha = _sha256_hex(filepath)
        results.append({
            "path": str(filepath),
            "label": label,
            "source": SOURCE_STUDENT_REAL,
            "parent_path": str(filepath.parent),
            "sha256": sha,
            "reject_reason": "",
        })
    return results


def _load_label_csvs(import_dir: Path) -> dict[str, str]:
    """Load label mappings from per-student CSV files in import directory.

    CSV format: img,width,height,x1,y1,x2,y2,class
    Returns dict of {filename: label}
    """
    label_map: dict[str, str] = {}
    for csv_path in sorted(import_dir.rglob("*.csv")):
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img = row.get("img", "").strip()
                    class_label = row.get("class", "").strip()
                    if img and class_label and class_label.isdigit() and len(class_label) == 3:
                        label_map[img] = class_label
        except Exception:
            pass  # Skip malformed CSVs
    return label_map


def _scan_import_directory(import_dir: Path, label_map: dict[str, str], project_root: Path) -> list[dict]:
    """Scan import directory for images, using label_map for class assignment."""
    results = []
    for filepath in sorted(import_dir.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        is_valid, reason = validate_image(filepath)
        if not is_valid:
            results.append({
                "path": str(filepath),
                "label": "",
                "source": "rejected",
                "parent_path": str(filepath.parent),
                "sha256": "",
                "reject_reason": reason,
            })
            continue

        filename = filepath.name
        label = label_map.get(filename, "")
        if not label:
            results.append({
                "path": str(filepath),
                "label": "",
                "source": "rejected",
                "parent_path": str(filepath.parent),
                "sha256": "",
                "reject_reason": "no_label_in_csv",
            })
            continue

        sha = _sha256_hex(filepath)
        results.append({
            "path": str(filepath),
            "label": label,
            "source": SOURCE_STUDENT_REAL,
            "parent_path": str(filepath.parent),
            "sha256": sha,
            "reject_reason": "",
        })
    return results


def build_manifest(
    output_path: str | Path = "outputs/manifests/raw_manifest.csv",
    project_root: str | Path | None = None,
) -> Path:
    """Scan all student data directories and write a raw manifest CSV.

    Returns path to the written manifest.
    """
    project_root = Path(project_root) if project_root else Path.cwd()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict] = []

    # Scan student real images
    for rel_dir in STUDENT_REAL_DIRS:
        scan_dir = project_root / rel_dir
        if scan_dir.exists():
            all_entries.extend(_scan_directory(scan_dir, project_root))

    # Scan student imports (batch data) using per-student label CSVs
    import_dir = project_root / STUDENT_IMPORT_DIR
    if import_dir.exists():
        label_map = _load_label_csvs(import_dir)
        print(f"  Loaded {len(label_map)} label mappings from import CSVs")
        all_entries.extend(_scan_import_directory(import_dir, label_map, project_root))

    # Remove duplicates by SHA
    seen_sha: set[str] = set()
    deduped: list[dict] = []
    for entry in all_entries:
        sha = entry.get("sha256", "")
        if sha and sha in seen_sha:
            entry["source"] = "rejected"
            entry["reject_reason"] = "duplicate_hash"
        if sha:
            seen_sha.add(sha)
        deduped.append(entry)

    # Write manifest
    fieldnames = ["path", "label", "source", "parent_path", "sha256", "reject_reason"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    accepted = sum(1 for e in deduped if e["source"] == SOURCE_STUDENT_REAL)
    rejected = sum(1 for e in deduped if e["source"] == "rejected")
    print(f"build_manifest: {accepted} accepted, {rejected} rejected → {output_path}")
    return output_path


if __name__ == "__main__":
    build_manifest()
