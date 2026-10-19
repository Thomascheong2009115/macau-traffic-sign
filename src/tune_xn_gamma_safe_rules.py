import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth
from src.tune_model_ensemble import DebugPrediction, read_debug_prediction_scores


RED_GROUP = {"105", "107", "110", "113", "118", "138"}
BLUE_GROUP = {"201", "203", "204", "209", "210", "213"}
CONFIDENCE_THRESHOLDS = (0.45, 0.55, 0.65)
CONFIDENCE_MARGINS = (0.05, 0.10, 0.15)
RESULT_FIELDS = [
    "rule",
    "correct",
    "accuracy",
    "rescued",
    "damaged",
    "net_gain",
    "203_correct_after",
    "wrong_107",
    "wrong_203",
    "rule_description",
]
PREVIEW_FIELDS = [
    "IMG",
    "true_class",
    "baseline_pred",
    "candidate_pred",
    "baseline_correct",
    "candidate_correct",
    "result",
    "rule",
    "rule_description",
]


@dataclass(frozen=True)
class GammaPrediction:
    img: str
    pred: str
    confidence: float
    student_pred: str
    student_confidence: float
    noise_pred: str
    noise_confidence: float


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


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_gamma_predictions(path: str | Path) -> dict[str, GammaPrediction]:
    predictions: dict[str, GammaPrediction] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, mapping, ("img", "IMG", "image"))
            pred = _get(row, mapping, ("ensemble_class", "final_class", "class", "Class"))
            student_pred = _get(row, mapping, ("student_class",))
            noise_pred = _get(row, mapping, ("noise_class",))
            student_conf = _to_float(_get(row, mapping, ("student_conf",)))
            noise_conf = _to_float(_get(row, mapping, ("noise_conf",)))
            if pred == student_pred:
                confidence = student_conf
            elif pred == noise_pred:
                confidence = noise_conf
            else:
                confidence = max(student_conf, noise_conf)
            if img:
                predictions[img] = GammaPrediction(
                    img=img,
                    pred=pred,
                    confidence=confidence,
                    student_pred=student_pred,
                    student_confidence=student_conf,
                    noise_pred=noise_pred,
                    noise_confidence=noise_conf,
                )
    return predictions


def _ranked_alternatives(
    base_pred: str,
    sources: list[DebugPrediction],
    allowed: set[str] | None = None,
    top23_only: bool = True,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for source in sources:
        labels = sorted(source.scores.items(), key=lambda item: item[1], reverse=True)
        if top23_only:
            labels = labels[1:3]
        for label, score in labels:
            if label == base_pred:
                continue
            if allowed is not None and label not in allowed:
                continue
            scores[label] = max(scores.get(label, 0.0), score)
    return sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)


def _red_group_prediction(base: GammaPrediction, sources: list[DebugPrediction]) -> str:
    if base.pred not in RED_GROUP:
        return base.pred
    alternatives = _ranked_alternatives(base.pred, sources, RED_GROUP)
    return alternatives[0][0] if alternatives else base.pred


def _blue_protected_prediction(base: GammaPrediction, sources: list[DebugPrediction]) -> str:
    if base.pred == "203":
        return base.pred
    if base.pred not in BLUE_GROUP:
        return base.pred
    alternatives = _ranked_alternatives(base.pred, sources, BLUE_GROUP)
    return alternatives[0][0] if alternatives else base.pred


def _confidence_gated_prediction(base: GammaPrediction, sources: list[DebugPrediction], threshold: float, margin: float) -> str:
    if base.confidence >= threshold and base.student_confidence >= threshold:
        return base.pred
    alternatives = _ranked_alternatives(base.pred, sources, allowed=None)
    for label, score in alternatives:
        if base.confidence - score <= margin:
            return label
    return base.pred


def _score_rule(
    truth: dict[str, str],
    gamma_predictions: dict[str, GammaPrediction],
    student_predictions: dict[str, DebugPrediction],
    noise_predictions: dict[str, DebugPrediction],
    rule: str,
    description: str,
    predict: Callable[[GammaPrediction, list[DebugPrediction]], str],
) -> tuple[dict[str, int | float | str], list[dict[str, str]]]:
    correct = 0
    rescued = 0
    damaged = 0
    correct_203 = 0
    wrong_107 = 0
    wrong_203 = 0
    total = 0
    preview_rows: list[dict[str, str]] = []
    for img, true_label in truth.items():
        if img not in gamma_predictions or img not in student_predictions or img not in noise_predictions:
            continue
        total += 1
        base = gamma_predictions[img]
        candidate = predict(base, [student_predictions[img], noise_predictions[img]])
        baseline_correct = base.pred == true_label
        candidate_correct = candidate == true_label
        correct += int(candidate_correct)
        correct_203 += int(true_label == "203" and candidate == "203")
        wrong_107 += int(true_label == "107" and candidate != "107")
        wrong_203 += int(true_label == "203" and candidate != "203")
        result = "unchanged"
        if candidate != base.pred:
            result = "neutral"
            if not baseline_correct and candidate_correct:
                rescued += 1
                result = "rescued"
            elif baseline_correct and not candidate_correct:
                damaged += 1
                result = "damaged"
            preview_rows.append(
                {
                    "IMG": img,
                    "true_class": true_label,
                    "baseline_pred": base.pred,
                    "candidate_pred": candidate,
                    "baseline_correct": str(baseline_correct),
                    "candidate_correct": str(candidate_correct),
                    "result": result,
                    "rule": rule,
                    "rule_description": description,
                }
            )
    metrics: dict[str, int | float | str] = {
        "rule": rule,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "rescued": rescued,
        "damaged": damaged,
        "net_gain": rescued - damaged,
        "203_correct_after": correct_203,
        "wrong_107": wrong_107,
        "wrong_203": wrong_203,
        "rule_description": description,
    }
    return metrics, preview_rows


def _best_key(row: dict[str, int | float | str]) -> tuple[float, float, float, float, float]:
    return (
        float(row["correct"]),
        float(row["net_gain"]),
        -float(row["damaged"]),
        -float(row["wrong_203"]),
        -float(row["wrong_107"]),
    )


def _candidate_rules() -> list[tuple[str, str, Callable[[GammaPrediction, list[DebugPrediction]], str]]]:
    rules: list[tuple[str, str, Callable[[GammaPrediction, list[DebugPrediction]], str]]] = [
        ("red_group_rerank", "red group top2/top3 rerank within 105/107/110/113/118/138", _red_group_prediction),
        ("blue_protected_rerank", "blue group top2/top3 rerank within 201/203/204/209/210/213 with 203 protected", _blue_protected_prediction),
    ]
    for threshold in CONFIDENCE_THRESHOLDS:
        for margin in CONFIDENCE_MARGINS:
            rules.append(
                (
                    f"confidence_gated_t{threshold:.2f}_m{margin:.2f}",
                    f"low confidence top2/top3 rerank threshold={threshold:.2f} margin={margin:.2f}",
                    lambda base, sources, t=threshold, m=margin: _confidence_gated_prediction(base, sources, t, m),
                )
            )
    return rules


def tune_xn_gamma_safe_rules(
    gamma_debug_csv: str | Path = "output/debug_xn_gamma.csv",
    student_debug_csv: str | Path = "output/_xn_gamma_student_debug.csv",
    noise_debug_csv: str | Path = "output/_xn_gamma_noise_debug.csv",
    gt_csv: str | Path = "input/ground_truth.csv",
    output_csv: str | Path = "output/xn_gamma_safe_rule_results.csv",
    preview_csv: str | Path = "output/xn_gamma_safe_rule_best_preview.csv",
) -> dict[str, int | float | str]:
    truth = read_ground_truth(gt_csv)
    gamma_predictions = read_gamma_predictions(gamma_debug_csv)
    student_predictions = read_debug_prediction_scores(student_debug_csv)
    noise_predictions = read_debug_prediction_scores(noise_debug_csv)

    scored: list[tuple[dict[str, int | float | str], list[dict[str, str]]]] = []
    for rule, description, predict in _candidate_rules():
        scored.append(_score_rule(truth, gamma_predictions, student_predictions, noise_predictions, rule, description, predict))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for metrics, _ in scored:
            writer.writerow({field: metrics[field] for field in RESULT_FIELDS})

    best_metrics, best_preview = max(scored, key=lambda item: _best_key(item[0]))
    preview_csv = Path(preview_csv)
    preview_csv.parent.mkdir(parents=True, exist_ok=True)
    with preview_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(best_preview)
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="search safe post-rerank rules for XN Gamma debug outputs")
    parser.add_argument("--gamma_debug", default="output/debug_xn_gamma.csv")
    parser.add_argument("--student_debug", default="output/_xn_gamma_student_debug.csv")
    parser.add_argument("--noise_debug", default="output/_xn_gamma_noise_debug.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_csv", default="output/xn_gamma_safe_rule_results.csv")
    parser.add_argument("--preview_csv", default="output/xn_gamma_safe_rule_best_preview.csv")
    args = parser.parse_args()
    best = tune_xn_gamma_safe_rules(args.gamma_debug, args.student_debug, args.noise_debug, args.gt, args.output_csv, args.preview_csv)
    for field in RESULT_FIELDS:
        print(f"{field}={best[field]}")


if __name__ == "__main__":
    main()
