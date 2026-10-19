import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DATA_PLAN_FIELDS = [
    "error_group",
    "true_class",
    "pred_class",
    "current_train_count",
    "current_aug_count",
    "priority",
    "recommended_aug_count",
    "recommended_collect_count",
]

COLLECT_PLAN_FIELDS = [
    "true_class",
    "error_group",
    "hard_case_count",
    "top_pred_classes",
    "current_train_count",
    "current_aug_count",
    "priority",
    "recommended_collect_count",
    "note",
]

PRIORITY_BY_GROUP = {
    "red_forbidden": "high",
    "blue_direction": "high",
    "warning_triangle": "medium",
    "other": "low",
}

PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
TRAIN_SOURCES = {"real", "real_student_images"}
AUG_SOURCES = {"augmented_real", "augmented_student"}


def _field_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header")
    return {name.strip().lower(): name for name in fieldnames}


def _get(row: dict[str, str], mapping: dict[str, str], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        key = candidate.lower()
        if key in mapping:
            return row.get(mapping[key], "").strip()
    raise ValueError(f"missing required column: one of {', '.join(candidates)}")


def _read_hard_cases(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            error_group = _get(row, mapping, ("error_group",))
            priority = row.get(mapping.get("priority", ""), "").strip() if "priority" in mapping else ""
            rows.append(
                {
                    "true_class": _get(row, mapping, ("true_class", "class", "label")),
                    "pred_class": _get(row, mapping, ("pred_class", "prediction", "pred")),
                    "error_group": error_group,
                    "priority": priority or PRIORITY_BY_GROUP.get(error_group, "low"),
                }
            )
    return rows


def _count_inventory(train_manifest: str | Path) -> tuple[Counter[str], Counter[str]]:
    train_counts: Counter[str] = Counter()
    aug_counts: Counter[str] = Counter()
    with Path(train_manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            label = _get(row, mapping, ("label", "class"))
            source = _get(row, mapping, ("source",))
            if source in TRAIN_SOURCES:
                train_counts[label] += 1
            elif source in AUG_SOURCES:
                aug_counts[label] += 1
    return train_counts, aug_counts


def _recommended_aug_count(priority: str, hard_count: int, true_class: str, pred_class: str) -> int:
    # Over-predicted classes are not direct augmentation targets. This function is called
    # only for the true class, and scales by observed mistakes for that true/pred pair.
    base = {"high": 60, "medium": 35, "low": 20}.get(priority, 20)
    if pred_class in {"107", "203"}:
        base += 20
    return base * max(1, hard_count)


def _recommended_collect_count(priority: str, train_count: int, hard_count: int) -> int:
    if train_count <= 1:
        base = 18
    elif train_count <= 2:
        base = 14
    elif train_count <= 5:
        base = 10
    elif train_count <= 10:
        base = 6
    else:
        base = 3
    multiplier = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(priority, 0.4)
    return max(1, int(round(base * multiplier))) + max(0, hard_count - 1)


def _note_for_group(error_group: str, pred_classes: Iterable[str]) -> str:
    preds = set(pred_classes)
    if error_group == "red_forbidden":
        return "focus true classes mispredicted as 107; do not blindly add 107"
    if error_group == "blue_direction":
        if "203" in preds:
            return "不要破坏 203; focus true classes mispredicted as 203"
        return "不要破坏 203; keep blue-direction augmentation source-balanced"
    if error_group == "warning_triangle":
        return "medium priority 3xx shape and crop robustness collection"
    return "low priority after red and blue-direction hard cases"


def build_hardcase_plan(
    hard_cases_csv: str | Path = "data/manifests/hard_cases_from_current_test.csv",
    train_manifest: str | Path = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv",
    data_plan_csv: str | Path = "output/hardcase_data_plan.csv",
    collect_plan_csv: str | Path = "output/hardcase_collect_plan.csv",
) -> list[dict[str, str]]:
    hard_cases = _read_hard_cases(hard_cases_csv)
    train_counts, aug_counts = _count_inventory(train_manifest)

    pair_counts: Counter[tuple[str, str, str]] = Counter()
    pair_priority: dict[tuple[str, str, str], str] = {}
    class_counts: Counter[str] = Counter()
    class_groups: dict[str, Counter[str]] = defaultdict(Counter)
    class_preds: dict[str, Counter[str]] = defaultdict(Counter)

    for row in hard_cases:
        key = (row["error_group"], row["true_class"], row["pred_class"])
        pair_counts[key] += 1
        pair_priority[key] = PRIORITY_BY_GROUP.get(row["error_group"], row["priority"])
        class_counts[row["true_class"]] += 1
        class_groups[row["true_class"]][row["error_group"]] += 1
        class_preds[row["true_class"]][row["pred_class"]] += 1

    data_rows: list[dict[str, str]] = []
    for (error_group, true_class, pred_class), hard_count in sorted(
        pair_counts.items(),
        key=lambda item: (
            PRIORITY_RANK.get(pair_priority[item[0]], 9),
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    ):
        priority = pair_priority[(error_group, true_class, pred_class)]
        current_train_count = train_counts[true_class]
        current_aug_count = aug_counts[true_class]
        data_rows.append(
            {
                "error_group": error_group,
                "true_class": true_class,
                "pred_class": pred_class,
                "current_train_count": str(current_train_count),
                "current_aug_count": str(current_aug_count),
                "priority": priority,
                "recommended_aug_count": str(_recommended_aug_count(priority, hard_count, true_class, pred_class)),
                "recommended_collect_count": str(_recommended_collect_count(priority, current_train_count, hard_count)),
            }
        )

    collect_rows: list[dict[str, str]] = []
    for true_class, hard_count in sorted(
        class_counts.items(),
        key=lambda item: (
            PRIORITY_RANK.get(PRIORITY_BY_GROUP.get(class_groups[item[0]].most_common(1)[0][0], "low"), 9),
            item[0],
        ),
    ):
        top_group = class_groups[true_class].most_common(1)[0][0]
        priority = PRIORITY_BY_GROUP.get(top_group, "low")
        pred_summary = "; ".join(f"{label} x{count}" for label, count in class_preds[true_class].most_common())
        collect_rows.append(
            {
                "true_class": true_class,
                "error_group": top_group,
                "hard_case_count": str(hard_count),
                "top_pred_classes": pred_summary,
                "current_train_count": str(train_counts[true_class]),
                "current_aug_count": str(aug_counts[true_class]),
                "priority": priority,
                "recommended_collect_count": str(_recommended_collect_count(priority, train_counts[true_class], hard_count)),
                "note": _note_for_group(top_group, class_preds[true_class]),
            }
        )

    data_plan_csv = Path(data_plan_csv)
    data_plan_csv.parent.mkdir(parents=True, exist_ok=True)
    with data_plan_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATA_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(data_rows)

    collect_plan_csv = Path(collect_plan_csv)
    collect_plan_csv.parent.mkdir(parents=True, exist_ok=True)
    with collect_plan_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLLECT_PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(collect_rows)

    return data_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="build hard-case data augmentation and collection plans")
    parser.add_argument("--hard_cases_csv", default="data/manifests/hard_cases_from_current_test.csv")
    parser.add_argument("--train_manifest", default="data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv")
    parser.add_argument("--data_plan_csv", default="output/hardcase_data_plan.csv")
    parser.add_argument("--collect_plan_csv", default="output/hardcase_collect_plan.csv")
    args = parser.parse_args()
    rows = build_hardcase_plan(args.hard_cases_csv, args.train_manifest, args.data_plan_csv, args.collect_plan_csv)
    print(f"data_plan_rows={len(rows)}")
    print(f"data_plan_csv={Path(args.data_plan_csv).as_posix()}")
    print(f"collect_plan_csv={Path(args.collect_plan_csv).as_posix()}")


if __name__ == "__main__":
    main()
