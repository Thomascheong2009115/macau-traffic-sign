"""Audit dataset: validate manifest, check class distribution, find issues.

Input: raw manifest CSV from build_manifest.py
Output:
  - outputs/audits/class_distribution.csv
  - outputs/manifests/rejected_files.csv
  - outputs/manifests/clean_manifest.csv (only valid entries)
"""
import csv
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.class_filter import is_valid_class, load_classes

ALLOWED_SOURCES = {"student_real", "sd_aug"}


def audit_dataset(
    manifest_path: str | Path = "outputs/manifests/raw_manifest.csv",
    classes_path: str | Path = "classes.txt",
    output_dir: str | Path = "outputs/audits",
    clean_manifest_path: str | Path = "outputs/manifests/clean_manifest.csv",
    rejected_path: str | Path = "outputs/manifests/rejected_files.csv",
) -> dict:
    """Audit manifest and produce clean/clean + rejected CSVs.

    Returns audit summary dict.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    clean_manifest_path = Path(clean_manifest_path)
    rejected_path = Path(rejected_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(classes_path)
    valid_labels = set(classes)

    clean: list[dict] = []
    rejected: list[dict] = []
    class_counts: Counter = Counter()
    source_counts: Counter = Counter()

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip()
            source = row.get("source", "").strip()
            path = row.get("path", "")
            reject_reason = row.get("reject_reason", "").strip()
            sha = row.get("sha256", "").strip()

            # Already rejected by build_manifest
            if source == "rejected" or reject_reason:
                rejected.append(dict(row))
                continue

            # Check source is allowed
            if source not in ALLOWED_SOURCES:
                row["reject_reason"] = f"invalid_source_{source}"
                rejected.append(dict(row))
                continue

            # Check label is valid
            if label not in valid_labels:
                row["reject_reason"] = f"invalid_label_{label}"
                rejected.append(dict(row))
                continue

            # Check teacher markers still not present
            path_lower = path.lower().replace("\\", "/")
            teacher_markers = ["real_images", "real_crops", "augmented_real"]
            if any(m in path_lower for m in teacher_markers):
                row["reject_reason"] = "teacher_data_marker"
                rejected.append(dict(row))
                continue

            # Accepted
            clean.append(dict(row))
            class_counts[label] += 1
            source_counts[source] += 1

    # Write clean manifest
    clean_fields = ["path", "label", "source", "parent_path", "sha256"]
    with clean_manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=clean_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean)

    # Write rejected files
    reject_fields = ["path", "label", "source", "parent_path", "sha256", "reject_reason"]
    with rejected_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reject_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rejected)

    # Write class distribution
    class_dist_path = output_dir / "class_distribution.csv"
    with class_dist_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count", "source"])
        for label in sorted(classes):
            writer.writerow([label, class_counts.get(label, 0), "student_real"])
        # Also note which classes are present/absent
        present = [c for c in classes if class_counts.get(c, 0) > 0]
        absent = [c for c in classes if class_counts.get(c, 0) == 0]
        writer.writerow([])
        writer.writerow([f"total_present_classes", len(present)])
        writer.writerow([f"total_absent_classes", len(absent)])
        if absent:
            writer.writerow(["absent_classes", ", ".join(absent)])

    summary = {
        "total_images": len(clean) + len(rejected),
        "accepted": len(clean),
        "rejected": len(rejected),
        "classes_present": len([c for c in classes if class_counts.get(c, 0) > 0]),
        "classes_total": len(classes),
        "classes_absent": len([c for c in classes if class_counts.get(c, 0) == 0]),
        "class_counts": dict(class_counts),
        "source_counts": dict(source_counts),
    }

    print(f"audit_dataset: {summary['accepted']} accepted, {summary['rejected']} rejected")
    print(f"  classes present: {summary['classes_present']}/{summary['classes_total']}")
    print(f"  clean manifest: {clean_manifest_path}")
    print(f"  rejected: {rejected_path}")
    print(f"  class dist: {class_dist_path}")

    return summary


if __name__ == "__main__":
    manifest = Path("outputs/manifests/raw_manifest.csv")
    if not manifest.exists():
        print("ERROR: raw_manifest.csv not found. Run build_manifest.py first.")
        import sys as _s
        _s.exit(1)
    audit_dataset(manifest)
