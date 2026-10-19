import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth


FINAL_WEIGHTS = (0.6, 0.7, 0.8, 0.9)
CONFIDENCE_THRESHOLDS = (0.45, 0.55, 0.65)
CONFIDENCE_MARGINS = (0.05, 0.10, 0.15)
RED_GROUP = {"105", "107", "110", "113", "138"}
PROTECTED_BLUE = {"202", "203", "204", "209", "210", "213", "215"}
RESULT_FIELDS = [
    "rule",
    "rule_description",
    "correct",
    "accuracy",
    "rescued",
    "damaged",
    "net_gain",
    "203_correct_after",
    "wrong_107",
    "wrong_203",
]
PREVIEW_FIELDS = [
    "IMG",
    "true_class",
    "final_pred",
    "wavelet_pred",
    "ensemble_pred",
    "final_correct",
    "ensemble_correct",
    "result",
    "rule_description",
]
XN_BETA_DEBUG_FIELDS = [
    "img",
    "final_class",
    "final_conf",
    "wavelet_class",
    "wavelet_conf",
    "ensemble_class",
    "rule",
    "rule_description",
]


@dataclass(frozen=True)
class DebugPrediction:
    img: str
    pred: str
    confidence: float
    scores: dict[str, float]


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


def read_debug_prediction_scores(path: str | Path) -> dict[str, DebugPrediction]:
    predictions: dict[str, DebugPrediction] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = _field_map(reader.fieldnames)
        for row in reader:
            img = _get(row, mapping, ("img", "IMG", "image"))
            if not img:
                continue
            scores: dict[str, float] = {}
            for index in range(1, 4):
                label = _get(row, mapping, (f"main_top{index}_class", f"top{index}_class", f"top{index}"))
                score = _to_float(_get(row, mapping, (f"main_top{index}_conf", f"top{index}_conf", f"top{index}_confidence")))
                if label:
                    scores[label] = max(scores.get(label, 0.0), score)
            pred = _get(row, mapping, ("final_class", "Class", "class", "top1_class", "main_top1_class", "top1", "pred", "prediction"))
            final_score = _to_float(_get(row, mapping, ("final_conf", "confidence", "conf", "main_top1_conf", "top1_conf", "top1_confidence")))
            if pred:
                scores[pred] = max(scores.get(pred, 0.0), final_score)
            predictions[img] = DebugPrediction(img=img, pred=pred, confidence=final_score, scores=scores)
    return predictions


def _weighted_average_prediction(final_pred: DebugPrediction, wavelet_pred: DebugPrediction, final_weight: float) -> str:
    wavelet_weight = 1.0 - final_weight
    combined: dict[str, float] = {}
    for label, score in final_pred.scores.items():
        combined[label] = combined.get(label, 0.0) + final_weight * score
    for label, score in wavelet_pred.scores.items():
        combined[label] = combined.get(label, 0.0) + wavelet_weight * score
    if not combined:
        return final_pred.pred
    return max(combined.items(), key=lambda item: (item[1], item[0]))[0]


def _class_gated_prediction(final_pred: DebugPrediction, wavelet_pred: DebugPrediction) -> str:
    if wavelet_pred.pred in RED_GROUP:
        return wavelet_pred.pred
    return final_pred.pred


def _protected_blue_prediction(final_pred: DebugPrediction, wavelet_pred: DebugPrediction) -> str:
    if final_pred.pred in PROTECTED_BLUE or wavelet_pred.pred in PROTECTED_BLUE:
        return final_pred.pred
    return wavelet_pred.pred


def _confidence_gated_prediction(final_pred: DebugPrediction, wavelet_pred: DebugPrediction, threshold: float, margin: float) -> str:
    if final_pred.confidence < threshold and wavelet_pred.confidence > final_pred.confidence + margin:
        return wavelet_pred.pred
    return final_pred.pred


def _prediction_for_rule(rule: str) -> tuple[str, str, Callable[[DebugPrediction, DebugPrediction], str]]:
    normalized = rule.strip().lower()
    if normalized in {"protected_blue", "safe", "xn_beta_safe"}:
        return "protected_blue", "protected_blue wavelet_override_except_protected_blue", _protected_blue_prediction
    if normalized in {"red_gated", "class_gated"}:
        return "red_gated", "class_gated red_group_wavelet_override", _class_gated_prediction
    if normalized.startswith("weighted_"):
        weight_text = normalized.split("_", 1)[1]
        final_weight = float(weight_text)
        return (
            f"weighted_{final_weight:.1f}",
            f"weighted_prob_average final_weight={final_weight:.1f} wavelet_weight={1.0 - final_weight:.1f}",
            lambda final_pred, wavelet_pred, weight=final_weight: _weighted_average_prediction(final_pred, wavelet_pred, weight),
        )
    raise ValueError("rule must be protected_blue, red_gated, or weighted_<final_weight>")


def _score_predictions(
    truth: dict[str, str],
    final_predictions: dict[str, DebugPrediction],
    wavelet_predictions: dict[str, DebugPrediction],
    rule: str,
    rule_description: str,
    predict: Callable[[DebugPrediction, DebugPrediction], str],
) -> tuple[dict[str, int | float | str], list[dict[str, str]]]:
    correct = 0
    rescued = 0
    damaged = 0
    correct_203 = 0
    wrong_107 = 0
    wrong_203 = 0
    preview_rows: list[dict[str, str]] = []
    total = 0
    for img, true_label in truth.items():
        if img not in final_predictions or img not in wavelet_predictions:
            continue
        total += 1
        final_pred = final_predictions[img]
        wavelet_pred = wavelet_predictions[img]
        ensemble_pred = predict(final_pred, wavelet_pred)
        final_correct = final_pred.pred == true_label
        ensemble_correct = ensemble_pred == true_label
        correct += int(ensemble_correct)
        if true_label == "203" and ensemble_pred == "203":
            correct_203 += 1
        if ensemble_pred == "107" and true_label != "107":
            wrong_107 += 1
        if ensemble_pred == "203" and true_label != "203":
            wrong_203 += 1
        result = "unchanged"
        if ensemble_pred != final_pred.pred:
            result = "neutral"
            if not final_correct and ensemble_correct:
                rescued += 1
                result = "rescued"
            elif final_correct and not ensemble_correct:
                damaged += 1
                result = "damaged"
            preview_rows.append(
                {
                    "IMG": img,
                    "true_class": true_label,
                    "final_pred": final_pred.pred,
                    "wavelet_pred": wavelet_pred.pred,
                    "ensemble_pred": ensemble_pred,
                    "final_correct": str(final_correct),
                    "ensemble_correct": str(ensemble_correct),
                    "result": result,
                    "rule_description": rule_description,
                }
            )
    metrics: dict[str, int | float | str] = {
        "rule": rule,
        "rule_description": rule_description,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "rescued": rescued,
        "damaged": damaged,
        "net_gain": rescued - damaged,
        "203_correct_after": correct_203,
        "wrong_107": wrong_107,
        "wrong_203": wrong_203,
    }
    return metrics, preview_rows


def _best_key(metrics: dict[str, int | float | str]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics["correct"]),
        float(metrics["net_gain"]),
        -float(metrics["damaged"]),
        -float(metrics["wrong_203"]),
        -float(metrics["wrong_107"]),
    )


def tune_model_ensemble(
    final_debug_csv: str | Path = "output/debug_latest.csv",
    wavelet_debug_csv: str | Path = "output/wavelet_ablation/high/debug_predictions.csv",
    gt_csv: str | Path = "input/ground_truth.csv",
    output_csv: str | Path = "output/ensemble_results.csv",
    preview_csv: str | Path = "output/ensemble_best_preview.csv",
    rule_prefix: str = "",
) -> dict[str, int | float | str]:
    truth = read_ground_truth(gt_csv)
    final_predictions = read_debug_prediction_scores(final_debug_csv)
    wavelet_predictions = read_debug_prediction_scores(wavelet_debug_csv)

    candidates: list[tuple[dict[str, int | float | str], list[dict[str, str]]]] = []
    for final_weight in FINAL_WEIGHTS:
        rule = f"weighted_{final_weight:.1f}"
        if rule_prefix:
            rule = f"{rule_prefix}:{rule}"
        description = f"weighted_prob_average final_weight={final_weight:.1f} wavelet_weight={1.0 - final_weight:.1f}"
        candidates.append(
            _score_predictions(
                truth,
                final_predictions,
                wavelet_predictions,
                rule,
                description,
                lambda final_pred, wavelet_pred, weight=final_weight: _weighted_average_prediction(final_pred, wavelet_pred, weight),
            )
        )
    rule = "red_gated"
    if rule_prefix:
        rule = f"{rule_prefix}:{rule}"
    candidates.append(
        _score_predictions(
            truth,
            final_predictions,
            wavelet_predictions,
            rule,
            "class_gated red_group_wavelet_override",
            _class_gated_prediction,
        )
    )
    rule = "protected_blue"
    if rule_prefix:
        rule = f"{rule_prefix}:{rule}"
    candidates.append(
        _score_predictions(
            truth,
            final_predictions,
            wavelet_predictions,
            rule,
            "protected_blue wavelet_override_except_protected_blue",
            _protected_blue_prediction,
        )
    )
    for threshold in CONFIDENCE_THRESHOLDS:
        for margin in CONFIDENCE_MARGINS:
            rule = f"confidence_gated_t{threshold:.2f}_m{margin:.2f}"
            if rule_prefix:
                rule = f"{rule_prefix}:{rule}"
            description = f"confidence_gated final_conf_lt={threshold:.2f} wavelet_margin={margin:.2f}"
            candidates.append(
                _score_predictions(
                    truth,
                    final_predictions,
                    wavelet_predictions,
                    rule,
                    description,
                    lambda final_pred, wavelet_pred, t=threshold, m=margin: _confidence_gated_prediction(final_pred, wavelet_pred, t, m),
                )
            )

    results = [metrics for metrics, _ in candidates]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for metrics in results:
            writer.writerow({field: metrics[field] for field in RESULT_FIELDS})

    best_metrics, best_preview = max(candidates, key=lambda item: _best_key(item[0]))
    preview_csv = Path(preview_csv)
    preview_csv.parent.mkdir(parents=True, exist_ok=True)
    with preview_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(best_preview)
    return best_metrics


def write_rule_submission(
    final_debug_csv: str | Path,
    wavelet_debug_csv: str | Path,
    output_submission: str | Path,
    output_debug: str | Path,
    rule: str,
) -> None:
    final_predictions = read_debug_prediction_scores(final_debug_csv)
    wavelet_predictions = read_debug_prediction_scores(wavelet_debug_csv)
    rule_name, rule_description, predict = _prediction_for_rule(rule)

    output_submission = Path(output_submission)
    output_debug = Path(output_debug)
    output_submission.parent.mkdir(parents=True, exist_ok=True)
    output_debug.parent.mkdir(parents=True, exist_ok=True)

    with output_submission.open("w", encoding="utf-8-sig", newline="") as sub_handle, output_debug.open(
        "w", encoding="utf-8-sig", newline=""
    ) as debug_handle:
        sub_writer = csv.writer(sub_handle)
        debug_writer = csv.DictWriter(debug_handle, fieldnames=XN_BETA_DEBUG_FIELDS)
        sub_writer.writerow(["img", "class"])
        debug_writer.writeheader()
        for img, final_pred in final_predictions.items():
            if img not in wavelet_predictions:
                raise ValueError(f"missing wavelet prediction for image: {img}")
            wavelet_pred = wavelet_predictions[img]
            ensemble_pred = predict(final_pred, wavelet_pred)
            sub_writer.writerow([img, ensemble_pred])
            debug_writer.writerow(
                {
                    "img": img,
                    "final_class": final_pred.pred,
                    "final_conf": f"{final_pred.confidence:.10g}",
                    "wavelet_class": wavelet_pred.pred,
                    "wavelet_conf": f"{wavelet_pred.confidence:.10g}",
                    "ensemble_class": ensemble_pred,
                    "rule": rule_name,
                    "rule_description": rule_description,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="search lightweight final/wavelet-high ensemble rules")
    parser.add_argument("--final_debug", default="output/debug_latest.csv")
    parser.add_argument("--wavelet_debug", default="output/wavelet_ablation/high/debug_predictions.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_csv", default="output/ensemble_results.csv")
    parser.add_argument("--preview_csv", default="output/ensemble_best_preview.csv")
    parser.add_argument("--rule_prefix", default="")
    parser.add_argument("--write_submission", action="store_true")
    parser.add_argument("--rule", default="protected_blue")
    parser.add_argument("--output_submission", default="output/submission_xn_beta_safe.csv")
    parser.add_argument("--output_debug", default="output/debug_xn_beta_safe.csv")
    args = parser.parse_args()
    if args.write_submission:
        write_rule_submission(args.final_debug, args.wavelet_debug, args.output_submission, args.output_debug, args.rule)
        print(f"submission={Path(args.output_submission).resolve()}")
        print(f"debug={Path(args.output_debug).resolve()}")
        return
    best = tune_model_ensemble(args.final_debug, args.wavelet_debug, args.gt, args.output_csv, args.preview_csv, args.rule_prefix)
    for field in RESULT_FIELDS:
        print(f"{field}={best[field]}")


if __name__ == "__main__":
    main()
