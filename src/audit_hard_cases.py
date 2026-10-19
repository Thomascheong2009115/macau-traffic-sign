import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


RED_CLASSES = {"105", "107", "110", "113", "138"}
BLUE_CLASSES = {"203", "204", "209", "210", "213"}
RED_EXPERT_NAME = "expert_red_105_107_110_113_138"

RED_AUDIT_FIELDS = [
    "expert_used_count",
    "changed_count",
    "target_group_trigger_count",
    "target_group_correct_before",
    "target_group_correct_after",
    "img",
    "true_class",
    "main_pred",
    "expert_pred",
    "final_class",
    "result",
]

HARD_CASE_FIELDS = ["img", "true_class", "pred_class", "error_group", "priority"]
SUMMARY_FIELDS = ["error_group", "count", "top_confusions", "recommended_action"]


def _field_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header")
    return {name.strip().lower(): name for name in fieldnames}


def _get(row: dict[str, str], mapping: dict[str, str], candidates: Iterable[str], default: str = "") -> str:
    for candidate in candidates:
        key = candidate.lower()
        if key in mapping:
            return row.get(mapping[key], "").strip()
    if default != "":
        return default
    raise ValueError(f"missing required column: one of {', '.join(candidates)}")


def _read_label_csv(path: str | Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, mapping, ("img", "IMG", "image"))
            label = _get(row, mapping, ("class", "Class", "CLASS", "label", "true_class"))
            if img:
                labels[img] = label
    return labels


def _read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, mapping, ("img", "IMG", "image"))
            pred = _get(row, mapping, ("final_class", "class", "Class", "top1_class", "main_top1_class", "pred"))
            rows.append({"img": img, "pred_class": pred})
    return rows


def _read_debug_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, mapping, ("img", "IMG", "image"))
            main_pred = _get(row, mapping, ("main_top1_class", "top1_class", "class", "Class", "pred"))
            final_class = _get(row, mapping, ("final_class", "class", "Class", "top1_class", "main_top1_class"))
            expert_used = _get(row, mapping, ("expert_used",), default="0")
            expert_pred = _get(row, mapping, ("expert_pred",), default="")
            rows.append(
                {
                    "img": img,
                    "main_pred": main_pred,
                    "final_class": final_class,
                    "expert_used": expert_used,
                    "expert_pred": expert_pred,
                }
            )
    return rows


def _strip_expert_prefix(value: str, expert_name: str) -> str:
    prefix = f"{expert_name}:"
    return value[len(prefix) :] if value.startswith(prefix) else value


def _result(true_class: str, before: str, after: str) -> str:
    before_correct = before == true_class
    after_correct = after == true_class
    if before == after:
        return "unchanged"
    if not before_correct and after_correct:
        return "rescued"
    if before_correct and not after_correct:
        return "damaged"
    return "neutral"


def audit_red_expert_trigger(
    debug_csv: str | Path = "output/debug_expert_red_105_107_110_113_138.csv",
    gt_csv: str | Path = "input/ground_truth.csv",
    output_csv: str | Path = "output/audit_red_expert_trigger.csv",
    expert_name: str = RED_EXPERT_NAME,
    target_classes: set[str] | None = None,
) -> dict[str, int]:
    target = target_classes or RED_CLASSES
    truth = _read_label_csv(gt_csv)
    rows = _read_debug_rows(debug_csv)

    metrics = {
        "expert_used_count": 0,
        "changed_count": 0,
        "target_group_trigger_count": 0,
        "target_group_correct_before": 0,
        "target_group_correct_after": 0,
    }
    detail_rows: list[dict[str, str | int]] = []

    for row in rows:
        img = row["img"]
        true_class = truth.get(img, "")
        if true_class in target:
            metrics["target_group_correct_before"] += int(row["main_pred"] == true_class)
            metrics["target_group_correct_after"] += int(row["final_class"] == true_class)

        expert_pred_raw = row["expert_pred"]
        is_red_trigger = row["expert_used"] == "1" and expert_pred_raw.startswith(f"{expert_name}:")
        if not is_red_trigger:
            continue

        expert_pred = _strip_expert_prefix(expert_pred_raw, expert_name)
        metrics["expert_used_count"] += 1
        if row["main_pred"] != row["final_class"]:
            metrics["changed_count"] += 1
        involved = {true_class, row["main_pred"], expert_pred, row["final_class"]}
        if involved & target:
            metrics["target_group_trigger_count"] += 1
        detail_rows.append(
            {
                **metrics,
                "img": img,
                "true_class": true_class,
                "main_pred": row["main_pred"],
                "expert_pred": expert_pred,
                "final_class": row["final_class"],
                "result": _result(true_class, row["main_pred"], row["final_class"]),
            }
        )

    # Re-apply final aggregate values to every detail row for convenient filtering.
    for detail in detail_rows:
        for key, value in metrics.items():
            detail[key] = value

    if not detail_rows:
        detail_rows.append({**metrics, "img": "", "true_class": "", "main_pred": "", "expert_pred": "", "final_class": "", "result": ""})

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RED_AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)
    return metrics


def classify_error_group(true_class: str, pred_class: str) -> str:
    involved = {true_class, pred_class}
    if involved & RED_CLASSES:
        return "red_forbidden"
    if involved & BLUE_CLASSES:
        return "blue_direction"
    if true_class.startswith("3") or pred_class.startswith("3"):
        return "warning_triangle"
    return "other"


def _priority(error_group: str) -> str:
    if error_group in {"red_forbidden", "blue_direction"}:
        return "high"
    if error_group == "warning_triangle":
        return "medium"
    return "low"


def _recommended_action(error_group: str) -> str:
    if error_group == "red_forbidden":
        return "Prepare hard-case augmentation for 105/107/110/113/138 and reduce 107 over-prediction."
    if error_group == "blue_direction":
        return "Do not use blue expert; audit 203-safe 203/204/209/210/213 triggers before retraining."
    if error_group == "warning_triangle":
        return "Keep 3xx warning triangle cases as medium-priority augmentation candidates."
    return "Review after high-priority red and blue hard cases."


def generate_hard_cases(
    gt_csv: str | Path = "input/ground_truth.csv",
    pred_csv: str | Path = "output/submission.csv",
    output_csv: str | Path = "data/manifests/hard_cases_from_current_test.csv",
    summary_csv: str | Path = "output/hard_case_summary.csv",
) -> dict[str, int]:
    truth = _read_label_csv(gt_csv)
    predictions = {row["img"]: row["pred_class"] for row in _read_prediction_rows(pred_csv)}

    hard_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    confusions: dict[str, Counter[str]] = defaultdict(Counter)

    for img, true_class in truth.items():
        pred_class = predictions.get(img, "")
        if not pred_class or pred_class == true_class:
            continue
        error_group = classify_error_group(true_class, pred_class)
        counts[error_group] += 1
        confusions[error_group][f"{true_class}->{pred_class}"] += 1
        hard_rows.append(
            {
                "img": img,
                "true_class": true_class,
                "pred_class": pred_class,
                "error_group": error_group,
                "priority": _priority(error_group),
            }
        )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HARD_CASE_FIELDS)
        writer.writeheader()
        writer.writerows(hard_rows)

    summary_rows: list[dict[str, str | int]] = []
    for group in ("red_forbidden", "blue_direction", "warning_triangle", "other"):
        if counts[group] <= 0:
            continue
        top_confusions = "; ".join(f"{pair} x{count}" for pair, count in confusions[group].most_common(5))
        summary_rows.append(
            {
                "error_group": group,
                "count": counts[group],
                "top_confusions": top_confusions,
                "recommended_action": _recommended_action(group),
            }
        )

    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="audit weak experts and generate current-test hard cases")
    parser.add_argument("--red_debug_csv", default="output/debug_expert_red_105_107_110_113_138.csv")
    parser.add_argument("--gt_csv", default="input/ground_truth.csv")
    parser.add_argument("--current_pred_csv", default="output/submission.csv")
    parser.add_argument("--red_audit_csv", default="output/audit_red_expert_trigger.csv")
    parser.add_argument("--hard_cases_csv", default="data/manifests/hard_cases_from_current_test.csv")
    parser.add_argument("--summary_csv", default="output/hard_case_summary.csv")
    args = parser.parse_args()

    red_metrics = audit_red_expert_trigger(args.red_debug_csv, args.gt_csv, args.red_audit_csv)
    hard_counts = generate_hard_cases(args.gt_csv, args.current_pred_csv, args.hard_cases_csv, args.summary_csv)

    for key, value in red_metrics.items():
        print(f"{key}={value}")
    for key in ("red_forbidden", "blue_direction", "warning_triangle", "other"):
        print(f"{key}_hard_cases={hard_counts.get(key, 0)}")


if __name__ == "__main__":
    main()
