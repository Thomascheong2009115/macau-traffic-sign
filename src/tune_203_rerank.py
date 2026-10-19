import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth


TARGET_CLASSES = {"202", "203", "204"}
MAX_CONF_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MARGIN_THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
CANDIDATE_RANKS = [2, 3, 5]


@dataclass(frozen=True)
class PredictionRow:
    img: str
    top_labels: list[str]
    top_confidences: list[float]

    @property
    def old_pred(self) -> str:
        return self.top_labels[0] if self.top_labels else ""

    @property
    def old_conf(self) -> float:
        return self.top_confidences[0] if self.top_confidences else 0.0


@dataclass(frozen=True)
class Rule:
    max_conf_threshold: float
    margin_threshold: float
    candidate_rank: int

    @property
    def description(self) -> str:
        return (
            "if top1 in {202,204} and 203 in top"
            f"{self.candidate_rank} and top1_conf <= {self.max_conf_threshold:.2f} "
            f"and top1_conf - conf_203 <= {self.margin_threshold:.2f}, set pred=203"
        )


def _normalized_field_map(fieldnames: Iterable[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header")
    return {name.strip().lower(): name for name in fieldnames}


def _get(row: dict[str, str], field_map: dict[str, str], candidates: Iterable[str], default: str = "") -> str:
    for candidate in candidates:
        key = candidate.lower()
        if key in field_map:
            return row.get(field_map[key], "").strip()
    return default


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_debug_predictions(path: str | Path, max_rank: int = 5) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        field_map = _normalized_field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, field_map, ("IMG", "image", "img"))
            labels: list[str] = []
            confidences: list[float] = []
            for rank in range(1, max_rank + 1):
                label = _get(row, field_map, (f"top{rank}",))
                confidence = _get(row, field_map, (f"top{rank}_confidence", f"top{rank}_conf", f"conf{rank}"))
                if label:
                    labels.append(label)
                    confidences.append(_float_or_zero(confidence))
            if img and labels:
                rows.append(PredictionRow(img=img, top_labels=labels, top_confidences=confidences))
    return rows


def _target_involved(prediction: PredictionRow, true_label: str) -> bool:
    labels = set(prediction.top_labels)
    labels.add(prediction.old_pred)
    labels.add(true_label)
    return bool(labels & TARGET_CLASSES)


def _candidate_203_confidence(prediction: PredictionRow, candidate_rank: int) -> float | None:
    search_limit = min(candidate_rank, len(prediction.top_labels))
    for index in range(1, search_limit):
        if prediction.top_labels[index] == "203":
            return prediction.top_confidences[index]
    return None


def apply_rule(prediction: PredictionRow, rule: Rule) -> tuple[str, float | None]:
    if prediction.old_pred not in {"202", "204"}:
        return prediction.old_pred, None
    conf_203 = _candidate_203_confidence(prediction, rule.candidate_rank)
    if conf_203 is None:
        return prediction.old_pred, None
    if prediction.old_conf <= rule.max_conf_threshold and prediction.old_conf - conf_203 <= rule.margin_threshold:
        return "203", conf_203
    return prediction.old_pred, None


def evaluate_rule(predictions: list[PredictionRow], truth: dict[str, str], rule: Rule) -> tuple[dict[str, object], list[list[str]]]:
    total = 0
    correct = 0
    changed_count = 0
    rescued_count = 0
    damaged_count = 0
    correct_203 = 0
    damaged_202 = 0
    damaged_204 = 0
    preview_rows: list[list[str]] = []

    for prediction in predictions:
        if prediction.img not in truth:
            continue
        true_label = truth[prediction.img]
        total += 1
        old_pred = prediction.old_pred
        new_pred = old_pred
        conf_203: float | None = None
        if _target_involved(prediction, true_label):
            new_pred, conf_203 = apply_rule(prediction, rule)

        old_correct = old_pred == true_label
        new_correct = new_pred == true_label
        if new_correct:
            correct += 1
        if true_label == "203" and new_correct:
            correct_203 += 1
        if new_pred != old_pred:
            changed_count += 1
            status = "neutral"
            if not old_correct and new_correct:
                rescued_count += 1
                status = "rescued"
            elif old_correct and not new_correct:
                damaged_count += 1
                status = "damaged"
                if true_label == "202":
                    damaged_202 += 1
                elif true_label == "204":
                    damaged_204 += 1
            preview_rows.append(
                [
                    prediction.img,
                    true_label,
                    old_pred,
                    new_pred,
                    f"{prediction.old_conf:.6f}",
                    f"{(conf_203 if conf_203 is not None else 0.0):.6f}",
                    status,
                ]
            )

    result = {
        "total_accuracy": correct / total if total else 0.0,
        "correct_count": correct,
        "changed_count": changed_count,
        "rescued_count": rescued_count,
        "damaged_count": damaged_count,
        "203_correct_count": correct_203,
        "202_damaged_count": damaged_202,
        "204_damaged_count": damaged_204,
        "max_conf_threshold": rule.max_conf_threshold,
        "margin_threshold": rule.margin_threshold,
        "candidate_rank": rule.candidate_rank,
        "rule_description": rule.description,
    }
    return result, preview_rows


def _result_sort_key(result: dict[str, object]) -> tuple[int, int, int, int, float, float]:
    return (
        int(result["correct_count"]),
        int(result["rescued_count"]),
        -int(result["damaged_count"]),
        -int(result["changed_count"]),
        -float(result["max_conf_threshold"]),
        -float(result["margin_threshold"]),
    )


def write_results(results: list[dict[str, object]], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "total_accuracy",
        "correct_count",
        "changed_count",
        "rescued_count",
        "damaged_count",
        "203_correct_count",
        "202_damaged_count",
        "204_damaged_count",
        "max_conf_threshold",
        "margin_threshold",
        "candidate_rank",
        "rule_description",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["total_accuracy"] = f"{float(row['total_accuracy']):.6f}"
            row["max_conf_threshold"] = f"{float(row['max_conf_threshold']):.2f}"
            row["margin_threshold"] = f"{float(row['margin_threshold']):.2f}"
            writer.writerow(row)


def write_preview(preview_rows: list[list[str]], preview_csv: str | Path) -> None:
    preview_csv = Path(preview_csv)
    preview_csv.parent.mkdir(parents=True, exist_ok=True)
    with preview_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["IMG", "true_class", "old_pred", "new_pred", "old_conf", "conf_203", "rescued_or_damaged"])
        writer.writerows(preview_rows)


def tune_203_rerank(
    debug_csv: str | Path,
    gt_csv: str | Path,
    output_csv: str | Path = "output/rerank_203_results.csv",
    preview_csv: str | Path = "output/rerank_203_best_preview.csv",
) -> dict[str, object]:
    predictions = read_debug_predictions(debug_csv, max_rank=max(CANDIDATE_RANKS))
    truth = read_ground_truth(gt_csv)

    scored: list[tuple[dict[str, object], list[list[str]]]] = []
    for max_conf_threshold in MAX_CONF_THRESHOLDS:
        for margin_threshold in MARGIN_THRESHOLDS:
            for candidate_rank in CANDIDATE_RANKS:
                rule = Rule(max_conf_threshold, margin_threshold, candidate_rank)
                scored.append(evaluate_rule(predictions, truth, rule))

    scored.sort(key=lambda item: _result_sort_key(item[0]), reverse=True)
    results = [item[0] for item in scored]
    best_result, best_preview_rows = scored[0]
    write_results(results, output_csv)
    write_preview(best_preview_rows, preview_csv)
    return best_result


def main() -> None:
    parser = argparse.ArgumentParser(description="search local 203 rerank rules from debug predictions")
    parser.add_argument("--debug_csv", default="output/debug_baseline.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_csv", default="output/rerank_203_results.csv")
    parser.add_argument("--preview_csv", default="output/rerank_203_best_preview.csv")
    args = parser.parse_args()

    best = tune_203_rerank(args.debug_csv, args.gt, args.output_csv, args.preview_csv)
    print(f"best_total_accuracy={float(best['total_accuracy']):.4f}")
    print(f"best_correct_count={best['correct_count']}")
    print(f"rescued_count={best['rescued_count']}")
    print(f"damaged_count={best['damaged_count']}")
    print(f"rule={best['rule_description']}")


if __name__ == "__main__":
    main()
