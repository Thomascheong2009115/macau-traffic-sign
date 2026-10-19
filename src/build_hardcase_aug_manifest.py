import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_TARGET_CLASSES = [
    "105",
    "110",
    "113",
    "138",
    "210",
    "204",
    "213",
    "201",
    "208",
    "119",
    "121",
    "312",
    "314",
    "317",
]


@dataclass(frozen=True)
class HardcaseAugManifestSummary:
    counts: dict[str, int]
    total: int
    output_csv: Path


def parse_target_classes(value: str) -> list[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise ValueError("target classes must not be empty")
    if any(not (label.isdigit() and len(label) == 3) for label in classes):
        raise ValueError("target classes must be comma-separated three-digit labels")
    if len(set(classes)) != len(classes):
        raise ValueError("target classes must be unique")
    return classes


def build_hardcase_aug_manifest(
    input_csv: str | Path = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv",
    output_csv: str | Path = "data/manifests/hardcase_target_classes.csv",
    target_classes: list[str] | None = None,
) -> HardcaseAugManifestSummary:
    targets = list(target_classes) if target_classes is not None else DEFAULT_TARGET_CLASSES
    target_set = set(targets)
    rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter({label: 0 for label in targets})

    with Path(input_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "label", "source"]:
            raise ValueError("input_csv columns must be path,label,source")
        for row in reader:
            label = row["label"].strip()
            if label not in target_set:
                continue
            out_row = {
                "path": row["path"].strip().replace("\\", "/"),
                "label": label,
                "source": row["source"].strip(),
            }
            rows.append(out_row)
            counts[label] += 1

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    for label in targets:
        print(f"{label}_count={counts[label]}")
    print(f"total={len(rows)}")
    print(f"output_csv={output_csv.as_posix()}")
    return HardcaseAugManifestSummary(counts=dict(counts), total=len(rows), output_csv=output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="build hard-case target-class manifest without generating images")
    parser.add_argument("--input_csv", default="data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv")
    parser.add_argument("--output_csv", default="data/manifests/hardcase_target_classes.csv")
    parser.add_argument("--target_classes", default=",".join(DEFAULT_TARGET_CLASSES))
    args = parser.parse_args()
    build_hardcase_aug_manifest(args.input_csv, args.output_csv, parse_target_classes(args.target_classes))


if __name__ == "__main__":
    main()
