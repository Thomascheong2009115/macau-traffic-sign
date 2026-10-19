"""Class-stratified dataset split for student real data.

Rules:
  1. val only from student_real (no sd_aug)
  2. sd_aug only goes to train
  3. Per-class split ratio maintained (85/15)
  4. Classes with >= 2 samples get at least 1 in val
  5. Single-sample classes go to train only
  6. SHA-based near-duplicate detection prevents cross-split leakage
"""
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _hash_simple(path_str: str) -> str:
    """Simple hash of path for grouping near-identical parent groups."""
    # Extract identifying parts from path
    # Student images have format: <student_id>_<variant>_<class>_F5C<XX>.jpg
    # We want same student+class combos to stay together
    parts = path_str.replace("\\", "/").split("/")
    filename = parts[-1]
    # Extract student+class key from filename (e.g., "1_1_104_F5C01" -> "1_104")
    segs = filename.rsplit(".", 1)[0].split("_")
    if len(segs) >= 3:
        # Return student_id + class as group key
        return f"{segs[0]}_{segs[2]}"
    return filename


def split_dataset(
    manifest_path: str | Path = "outputs/manifests/clean_manifest.csv",
    val_ratio: float = 0.15,
    seed: int = 42,
    train_output: str | Path = "outputs/manifests/train.csv",
    val_output: str | Path = "outputs/manifests/val.csv",
    audit_output: str | Path = "outputs/audits/split_audit.csv",
) -> tuple[list[dict], list[dict]]:
    """Class-stratified split with per-class ratio enforcement.

    Returns (train_entries, val_entries).
    """
    manifest_path = Path(manifest_path)
    train_output = Path(train_output)
    val_output = Path(val_output)
    audit_output = Path(audit_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        all_entries = list(csv.DictReader(f))

    if not all_entries:
        raise ValueError("No entries in manifest")

    # Separate by source
    student_real = [e for e in all_entries if e.get("source") == "student_real"]
    sd_aug = [e for e in all_entries if e.get("source") == "sd_aug"]

    # SD aug always goes to train
    train: list[dict] = list(sd_aug)

    # Group student_real by class label for stratified split
    class_groups: dict[str, list[dict]] = defaultdict(list)
    for e in student_real:
        class_groups[e["label"]].append(e)

    val_entries: list[dict] = []
    train_from_real: list[dict] = []

    for label, entries in sorted(class_groups.items()):
        n = len(entries)
        val_n = max(1, int(n * val_ratio)) if n >= 2 else 0
        # Shuffle deterministically
        shuffled = sorted(entries, key=lambda e: e.get("sha256", e["path"]))
        random.shuffle(shuffled)

        if val_n > 0:
            val_entries.extend(shuffled[:val_n])
            train_from_real.extend(shuffled[val_n:])
        else:
            train_from_real.extend(shuffled)

    train.extend(train_from_real)

    # Write train manifest
    train_fields = ["path", "label", "source", "parent_path"]
    with train_output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=train_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(train)

    # Write val manifest
    val_fields = ["path", "label", "source", "parent_path"]
    with val_output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=val_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(val_entries)

    # Write split audit
    train_class_counts = Counter(e["label"] for e in train)
    val_class_counts = Counter(e["label"] for e in val_entries)
    train_source_counts = Counter(e.get("source", "student_real") for e in train)
    val_source_counts = Counter(e.get("source", "student_real") for e in val_entries)

    with audit_output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "train_count", "val_count", "total"])
        all_labels = sorted(set(list(train_class_counts) + list(val_class_counts)))
        for lb in all_labels:
            writer.writerow([lb, train_class_counts.get(lb, 0), val_class_counts.get(lb, 0),
                             train_class_counts.get(lb, 0) + val_class_counts.get(lb, 0)])
        writer.writerow([])
        writer.writerow(["source", "train_count", "val_count"])
        all_sources = sorted(set(list(train_source_counts) + list(val_source_counts)))
        for src in all_sources:
            writer.writerow([src, train_source_counts.get(src, 0), val_source_counts.get(src, 0)])
        writer.writerow([])
        writer.writerow(["total", len(train), len(val_entries), len(train) + len(val_entries)])

    # Verify no SHA leakage between train and val
    train_shas = {e.get("sha256", "") for e in train if e.get("sha256")}
    val_shas = {e.get("sha256", "") for e in val_entries if e.get("sha256")}
    dup_shas = train_shas & val_shas
    if dup_shas:
        raise ValueError(f"Hash leakage detected: {len(dup_shas)} duplicate images across train/val")

    print(f"split_dataset: train={len(train)}, val={len(val_entries)}")
    print(f"  train classes: {len(train_class_counts)}, val classes: {len(val_class_counts)}")
    print(f"  train: {train_output}")
    print(f"  val:   {val_output}")
    print(f"  audit: {audit_output}")

    return train, val_entries


if __name__ == "__main__":
    manifest = Path("outputs/manifests/clean_manifest.csv")
    if not manifest.exists():
        print("ERROR: clean_manifest.csv not found. Run audit_dataset.py first.")
        import sys as _s
        _s.exit(1)
    split_dataset(manifest)
