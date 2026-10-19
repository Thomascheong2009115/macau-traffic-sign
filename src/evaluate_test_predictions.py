import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _normalized_field_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header")
    return {name.strip().lower(): name for name in fieldnames}


def _get(row: dict[str, str], field_map: dict[str, str], candidates: Iterable[str]) -> str:
    for candidate in candidates:
        key = candidate.lower()
        if key in field_map:
            return row.get(field_map[key], "").strip()
    raise ValueError(f"missing required column: one of {', '.join(candidates)}")


def read_ground_truth(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ground truth CSV not found: {path}. "
            "Provide a CSV with columns IMG,Class, or copy it to input/ground_truth.csv. "
            "The evaluation step cannot compute accuracy without true labels."
        )
    truth: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        field_map = _normalized_field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, field_map, ("IMG", "image", "img"))
            label = _get(row, field_map, ("Class", "CLASS", "class", "label", "true", "true_class"))
            if img:
                truth[img] = label
    return truth


def read_predictions(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        field_map = _normalized_field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, field_map, ("IMG", "image", "img"))
            top1 = _get(row, field_map, ("final_class", "ensemble_top1", "ensemble_class", "top1_class", "top1", "pred", "prediction", "Class"))
            top2 = ""
            for name in ("ensemble_top2", "main_top2_class", "top2_class", "top2"):
                if name in field_map:
                    top2 = row.get(field_map[name], "").strip()
                    break
            top3 = ""
            for name in ("ensemble_top3", "main_top3_class", "top3_class", "top3"):
                if name in field_map:
                    top3 = row.get(field_map[name], "").strip()
                    break
            confidence = ""
            for name in ("final_conf", "main_top1_conf", "top1_conf", "top1_confidence", "confidence", "conf"):
                if name in field_map:
                    confidence = row.get(field_map[name], "").strip()
                    break
            rows.append({"IMG": img, "top1": top1, "top2": top2, "top3": top3, "confidence": confidence})
    return rows


def evaluate_predictions(
    pred_csv: str | Path,
    gt_csv: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, float | int]:
    pred_csv = Path(pred_csv)
    gt_csv = Path(gt_csv)
    output_dir = Path(output_dir) if output_dir is not None else pred_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = read_ground_truth(gt_csv)
    predictions = read_predictions(pred_csv)
    total = 0
    correct = 0
    top3_contains_true_count = 0
    wrong_rows: list[list[str]] = []
    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()
    confusion_pairs: Counter[tuple[str, str]] = Counter()

    for pred in predictions:
        img = pred["IMG"]
        if img not in truth:
            continue
        true_label = truth[img]
        top_labels = [pred["top1"], pred["top2"], pred["top3"]]
        total += 1
        per_class_total[true_label] += 1
        is_correct = pred["top1"] == true_label
        if is_correct:
            correct += 1
            per_class_correct[true_label] += 1
        else:
            confusion_pairs[(true_label, pred["top1"])] += 1
            wrong_rows.append(
                [
                    img,
                    true_label,
                    pred["top1"],
                    pred["confidence"],
                    pred["top1"],
                    pred["top2"],
                    pred["top3"],
                    str(true_label in top_labels),
                ]
            )
        if true_label in top_labels:
            top3_contains_true_count += 1

    accuracy = correct / total if total else 0.0
    with (output_dir / "wrong_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["IMG", "true_class", "pred_class", "confidence", "top1", "top2", "top3", "top3_contains_true"])
        writer.writerows(wrong_rows)

    with (output_dir / "per_class_accuracy.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class", "total", "correct", "accuracy"])
        for label in sorted(per_class_total):
            class_total = per_class_total[label]
            class_correct = per_class_correct[label]
            writer.writerow([label, class_total, class_correct, class_correct / class_total if class_total else 0.0])

    with (output_dir / "confusion_pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "pred_class", "count"])
        for (true_label, pred_label), count in confusion_pairs.most_common():
            writer.writerow([true_label, pred_label, count])

    metrics = {
        "test_accuracy": accuracy,
        "top3_contains_true_count": top3_contains_true_count,
        "total": total,
        "correct": correct,
    }
    with (output_dir / "test_metrics.txt").open("w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            handle.write(f"{key}={value}\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate external test predictions")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    try:
        metrics = evaluate_predictions(args.pred, args.gt, args.output_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    print(f"test_accuracy={metrics['test_accuracy']:.4f}")
    print(f"top3_contains_true_count={metrics['top3_contains_true_count']}")
    print(f"total={metrics['total']}")
    print(f"correct={metrics['correct']}")


if __name__ == "__main__":
    main()
