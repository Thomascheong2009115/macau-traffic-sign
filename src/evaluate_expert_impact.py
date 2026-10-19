import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth


SUMMARY_FIELDS = [
    "baseline_correct",
    "expert_correct",
    "net_gain",
    "changed_count",
    "rescued_count",
    "damaged_count",
    "203_correct_before",
    "203_correct_after",
    "202_damaged_count",
    "204_damaged_count",
]
DETAIL_FIELDS = ["img", "true_class", "old_pred", "new_pred", "old_correct", "new_correct", "result"]


def _field_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header")
    return {name.strip().lower(): name for name in fieldnames}


def _get(row: dict[str, str], mapping: dict[str, str], candidates: Iterable[str], default: str = "") -> str:
    for candidate in candidates:
        key = candidate.lower()
        if key in mapping:
            return row.get(mapping[key], "").strip()
    return default


def _read_debug_predictions(path: str | Path, final: bool = False) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, mapping, ("img", "IMG", "image"))
            if final:
                pred = _get(row, mapping, ("final_class", "class", "top1_class", "top1", "pred", "prediction", "Class"))
            else:
                pred = _get(row, mapping, ("top1_class", "top1", "main_top1_class", "final_class", "pred", "prediction", "Class"))
            if img:
                predictions[img] = pred
    return predictions


def evaluate_expert_impact(
    baseline_debug: str | Path,
    expert_debug: str | Path,
    gt_csv: str | Path,
    output_csv: str | Path = "output/expert_impact.csv",
    detail_csv: str | Path | None = None,
) -> dict[str, int]:
    truth = read_ground_truth(gt_csv)
    baseline = _read_debug_predictions(baseline_debug, final=False)
    expert = _read_debug_predictions(expert_debug, final=True)

    metrics = {field: 0 for field in SUMMARY_FIELDS}
    detail_rows: list[dict[str, str]] = []
    for img, true_label in truth.items():
        if img not in baseline or img not in expert:
            continue
        old_pred = baseline[img]
        new_pred = expert[img]
        old_correct = old_pred == true_label
        new_correct = new_pred == true_label
        metrics["baseline_correct"] += int(old_correct)
        metrics["expert_correct"] += int(new_correct)
        if true_label == "203":
            metrics["203_correct_before"] += int(old_correct)
            metrics["203_correct_after"] += int(new_correct)
        result = "unchanged"
        if old_pred != new_pred:
            metrics["changed_count"] += 1
            result = "neutral"
            if not old_correct and new_correct:
                metrics["rescued_count"] += 1
                result = "rescued"
            elif old_correct and not new_correct:
                metrics["damaged_count"] += 1
                result = "damaged"
                if true_label == "202":
                    metrics["202_damaged_count"] += 1
                elif true_label == "204":
                    metrics["204_damaged_count"] += 1
        detail_rows.append(
            {
                "img": img,
                "true_class": true_label,
                "old_pred": old_pred,
                "new_pred": new_pred,
                "old_correct": str(old_correct),
                "new_correct": str(new_correct),
                "result": result,
            }
        )
    metrics["net_gain"] = metrics["rescued_count"] - metrics["damaged_count"]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(metrics)

    detail_path = Path(detail_csv) if detail_csv is not None else output_csv.with_name(output_csv.stem + "_details.csv")
    with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(detail_rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="compare baseline and 202/203/204 expert debug outputs")
    parser.add_argument("--baseline_debug", default="output/debug_full_reinforced.csv")
    parser.add_argument("--expert_debug", default="output/debug_expert_202_203_204.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_csv", default="output/expert_impact.csv")
    parser.add_argument("--detail_csv")
    args = parser.parse_args()
    metrics = evaluate_expert_impact(args.baseline_debug, args.expert_debug, args.gt, args.output_csv, args.detail_csv)
    for key in SUMMARY_FIELDS:
        print(f"{key}={metrics[key]}")


if __name__ == "__main__":
    main()
