import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth


CONF_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MARGIN_THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
CANDIDATE_RANKS = [2, 3]


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
    rule_type: str
    conf_threshold: float
    margin_threshold: float
    candidate_rank: int
    focus_group: tuple[str, ...] = ()

    @property
    def description(self) -> str:
        base = (
            f"{self.rule_type}: if top1_conf <= {self.conf_threshold:.2f} "
            f"and candidate=top{self.candidate_rank} "
            f"and top1_conf - candidate_conf <= {self.margin_threshold:.2f}, "
            f"replace top1 with top{self.candidate_rank}"
        )
        if self.rule_type == "focus_group":
            return base + " when old/new are both in {" + ",".join(self.focus_group) + "}"
        return base


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
                label = _get(row, field_map, (f"top{rank}_class", f"top{rank}", f"pred{rank}"))
                confidence = _get(
                    row,
                    field_map,
                    (f"top{rank}_conf", f"top{rank}_confidence", f"confidence{rank}", f"conf{rank}"),
                )
                if label:
                    labels.append(label)
                    confidences.append(_float_or_zero(confidence))
            if img and labels:
                rows.append(PredictionRow(img=img, top_labels=labels, top_confidences=confidences))
    return rows


def parse_focus_group(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _candidate(prediction: PredictionRow, rank: int) -> tuple[str, float] | None:
    index = rank - 1
    if index <= 0 or index >= len(prediction.top_labels):
        return None
    return prediction.top_labels[index], prediction.top_confidences[index]


def apply_rule(prediction: PredictionRow, rule: Rule) -> tuple[str, float | None]:
    candidate = _candidate(prediction, rule.candidate_rank)
    if candidate is None:
        return prediction.old_pred, None
    candidate_label, candidate_conf = candidate
    if prediction.old_conf > rule.conf_threshold:
        return prediction.old_pred, None
    if prediction.old_conf - candidate_conf > rule.margin_threshold:
        return prediction.old_pred, None
    if rule.rule_type == "focus_group":
        focus = set(rule.focus_group)
        if prediction.old_pred not in focus or candidate_label not in focus:
            return prediction.old_pred, None
    return candidate_label, candidate_conf


def evaluate_rule(predictions: list[PredictionRow], truth: dict[str, str], rule: Rule) -> tuple[dict[str, object], list[list[str]]]:
    total = 0
    correct = 0
    changed_count = 0
    rescued_count = 0
    damaged_count = 0
    preview_rows: list[list[str]] = []

    for prediction in predictions:
        if prediction.img not in truth:
            continue
        true_label = truth[prediction.img]
        total += 1
        old_pred = prediction.old_pred
        new_pred, new_conf = apply_rule(prediction, rule)
        old_correct = old_pred == true_label
        new_correct = new_pred == true_label
        if new_correct:
            correct += 1
        if new_pred != old_pred:
            changed_count += 1
            result = "neutral"
            if not old_correct and new_correct:
                rescued_count += 1
                result = "rescued"
            elif old_correct and not new_correct:
                damaged_count += 1
                result = "damaged"
            preview_rows.append(
                [
                    prediction.img,
                    true_label,
                    old_pred,
                    new_pred,
                    f"{prediction.old_conf:.6f}",
                    f"{(new_conf if new_conf is not None else 0.0):.6f}",
                    result,
                ]
            )

    result_row = {
        "correct_count": correct,
        "accuracy": correct / total if total else 0.0,
        "changed_count": changed_count,
        "rescued_count": rescued_count,
        "damaged_count": damaged_count,
        "net_gain": rescued_count - damaged_count,
        "rule_type": rule.rule_type,
        "conf_threshold": rule.conf_threshold,
        "margin_threshold": rule.margin_threshold,
        "candidate_rank": rule.candidate_rank,
        "rule_description": rule.description,
    }
    return result_row, preview_rows


def write_wrong_sample_stats(predictions: list[PredictionRow], truth: dict[str, str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "count"])
        wrong = [row for row in predictions if row.img in truth and row.old_pred != truth[row.img]]
        writer.writerow(["wrong_total", len(wrong)])
        writer.writerow(["true_in_top2", sum(1 for row in wrong if len(row.top_labels) >= 2 and truth[row.img] in row.top_labels[:2])])
        writer.writerow(["true_in_top3", sum(1 for row in wrong if len(row.top_labels) >= 3 and truth[row.img] in row.top_labels[:3])])
        writer.writerow(["true_in_top5", sum(1 for row in wrong if truth[row.img] in row.top_labels[:5])])


def write_confusion_pairs(predictions: list[PredictionRow], truth: dict[str, str], output_path: str | Path) -> None:
    pairs = Counter((truth[row.img], row.old_pred) for row in predictions if row.img in truth and row.old_pred != truth[row.img])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "pred_class", "count"])
        for (true_label, pred_label), count in pairs.most_common():
            writer.writerow([true_label, pred_label, count])


def write_results(results: list[dict[str, object]], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "correct_count",
        "accuracy",
        "changed_count",
        "rescued_count",
        "damaged_count",
        "net_gain",
        "rule_type",
        "conf_threshold",
        "margin_threshold",
        "candidate_rank",
        "rule_description",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["accuracy"] = f"{float(row['accuracy']):.6f}"
            row["conf_threshold"] = f"{float(row['conf_threshold']):.2f}"
            row["margin_threshold"] = f"{float(row['margin_threshold']):.2f}"
            writer.writerow(row)


def write_preview(preview_rows: list[list[str]], preview_csv: str | Path) -> None:
    preview_csv = Path(preview_csv)
    preview_csv.parent.mkdir(parents=True, exist_ok=True)
    with preview_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["IMG", "true_class", "old_pred", "new_pred", "old_conf", "new_conf", "result"])
        writer.writerows(preview_rows)


def _sort_key(result: dict[str, object]) -> tuple[int, int, int, int, float, float]:
    return (
        int(result["correct_count"]),
        int(result["net_gain"]),
        int(result["rescued_count"]),
        -int(result["damaged_count"]),
        -float(result["conf_threshold"]),
        -float(result["margin_threshold"]),
    )


def _candidate_rules(focus_group: tuple[str, ...]) -> list[Rule]:
    rules: list[Rule] = []
    for conf_threshold in CONF_THRESHOLDS:
        for margin_threshold in MARGIN_THRESHOLDS:
            for candidate_rank in CANDIDATE_RANKS:
                rules.append(Rule("global", conf_threshold, margin_threshold, candidate_rank))
                if focus_group:
                    rules.append(Rule("focus_group", conf_threshold, margin_threshold, candidate_rank, focus_group))
    return rules


def tune_topk_rerank(
    debug_csv: str | Path = "output/debug_full_reinforced.csv",
    gt_csv: str | Path = "input/ground_truth.csv",
    output_csv: str | Path = "output/topk_rerank_results.csv",
    preview_csv: str | Path = "output/topk_rerank_best_preview.csv",
    focus_groups: str | None = None,
) -> dict[str, object]:
    predictions = read_debug_predictions(debug_csv, max_rank=5)
    truth = read_ground_truth(gt_csv)
    focus_group = parse_focus_group(focus_groups)

    scored: list[tuple[dict[str, object], list[list[str]]]] = []
    for rule in _candidate_rules(focus_group):
        scored.append(evaluate_rule(predictions, truth, rule))

    scored.sort(key=lambda item: _sort_key(item[0]), reverse=True)
    results = [item[0] for item in scored]
    best_result, best_preview_rows = scored[0]
    output_path = Path(output_csv)
    write_results(results, output_path)
    write_preview(best_preview_rows, preview_csv)
    write_wrong_sample_stats(predictions, truth, output_path.with_name("topk_rerank_wrong_stats.csv"))
    write_confusion_pairs(predictions, truth, output_path.with_name("topk_rerank_confusion_pairs.csv"))
    return best_result


def main() -> None:
    parser = argparse.ArgumentParser(description="search top-k rerank rules from debug predictions")
    parser.add_argument("--debug_csv", default="output/debug_full_reinforced.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_csv", default="output/topk_rerank_results.csv")
    parser.add_argument("--preview_csv", default="output/topk_rerank_best_preview.csv")
    parser.add_argument("--focus_groups", default="")
    args = parser.parse_args()

    best = tune_topk_rerank(args.debug_csv, args.gt, args.output_csv, args.preview_csv, args.focus_groups)
    print(f"best_correct_count={best['correct_count']}")
    print(f"best_accuracy={float(best['accuracy']):.4f}")
    print(f"changed_count={best['changed_count']}")
    print(f"rescued_count={best['rescued_count']}")
    print(f"damaged_count={best['damaged_count']}")
    print(f"net_gain={best['net_gain']}")
    print(f"rule={best['rule_description']}")
    if int(best["net_gain"]) > 0:
        print("recommendation=consider adopting this rule after manual review")
    else:
        print("recommendation=do not adopt; net_gain <= 0")


if __name__ == "__main__":
    main()
