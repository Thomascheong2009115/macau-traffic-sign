import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth
from src.tune_model_ensemble import (
    CONFIDENCE_MARGINS,
    CONFIDENCE_THRESHOLDS,
    FINAL_WEIGHTS,
    PROTECTED_BLUE,
    RED_GROUP,
    DebugPrediction,
    read_debug_prediction_scores,
)


RESULT_FIELDS = [
    "mode",
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


def _weighted_prediction(models: list[tuple[DebugPrediction, float]]) -> str:
    combined: dict[str, float] = {}
    for prediction, weight in models:
        for label, score in prediction.scores.items():
            combined[label] = combined.get(label, 0.0) + weight * score
    if not combined:
        return models[0][0].pred
    return max(combined.items(), key=lambda item: (item[1], item[0]))[0]


def _protected_blue_prediction(final_pred: DebugPrediction, candidates: list[DebugPrediction]) -> str:
    if final_pred.pred in PROTECTED_BLUE or any(candidate.pred in PROTECTED_BLUE for candidate in candidates):
        return final_pred.pred
    return max(candidates, key=lambda item: item.confidence).pred


def _red_gated_prediction(final_pred: DebugPrediction, candidates: list[DebugPrediction]) -> str:
    for candidate in candidates:
        if candidate.pred in RED_GROUP:
            return candidate.pred
    return final_pred.pred


def _confidence_gated_prediction(final_pred: DebugPrediction, candidates: list[DebugPrediction], threshold: float, margin: float) -> str:
    if final_pred.confidence >= threshold:
        return final_pred.pred
    best_candidate = max(candidates, key=lambda item: item.confidence)
    if best_candidate.confidence > final_pred.confidence + margin:
        return best_candidate.pred
    return final_pred.pred


def _score(
    truth: dict[str, str],
    final_predictions: dict[str, DebugPrediction],
    candidate_predictions: list[dict[str, DebugPrediction]],
    mode: str,
    rule: str,
    description: str,
    predict: Callable[[DebugPrediction, list[DebugPrediction]], str],
) -> dict[str, int | float | str]:
    correct = 0
    rescued = 0
    damaged = 0
    correct_203 = 0
    wrong_107 = 0
    wrong_203 = 0
    total = 0
    for img, true_label in truth.items():
        if img not in final_predictions or any(img not in predictions for predictions in candidate_predictions):
            continue
        total += 1
        final_pred = final_predictions[img]
        candidates = [predictions[img] for predictions in candidate_predictions]
        ensemble_pred = predict(final_pred, candidates)
        final_correct = final_pred.pred == true_label
        ensemble_correct = ensemble_pred == true_label
        correct += int(ensemble_correct)
        if true_label == "203" and ensemble_pred == "203":
            correct_203 += 1
        if ensemble_pred == "107" and true_label != "107":
            wrong_107 += 1
        if ensemble_pred == "203" and true_label != "203":
            wrong_203 += 1
        if ensemble_pred != final_pred.pred:
            if not final_correct and ensemble_correct:
                rescued += 1
            elif final_correct and not ensemble_correct:
                damaged += 1
    return {
        "mode": mode,
        "rule": rule,
        "rule_description": description,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "rescued": rescued,
        "damaged": damaged,
        "net_gain": rescued - damaged,
        "203_correct_after": correct_203,
        "wrong_107": wrong_107,
        "wrong_203": wrong_203,
    }


def _best_key(row: dict[str, int | float | str]) -> tuple[float, float, float, float, float]:
    return (
        float(row["correct"]),
        float(row["net_gain"]),
        -float(row["damaged"]),
        -float(row["wrong_203"]),
        -float(row["wrong_107"]),
    )


def _rows_for_pair(
    truth: dict[str, str],
    final_predictions: dict[str, DebugPrediction],
    noise_predictions: dict[str, DebugPrediction],
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for final_weight in FINAL_WEIGHTS:
        noise_weight = 1.0 - final_weight
        rows.append(
            _score(
                truth,
                final_predictions,
                [noise_predictions],
                "final_noise",
                f"weighted_{final_weight:.1f}",
                f"weighted average final={final_weight:.1f} noise={noise_weight:.1f}",
                lambda final_pred, candidates, fw=final_weight, nw=noise_weight: _weighted_prediction(
                    [(final_pred, fw), (candidates[0], nw)]
                ),
            )
        )
    rows.append(_score(truth, final_predictions, [noise_predictions], "final_noise", "protected_blue", "protected-blue final guard", _protected_blue_prediction))
    rows.append(_score(truth, final_predictions, [noise_predictions], "final_noise", "red_gated", "red-gated noise override", _red_gated_prediction))
    for threshold in CONFIDENCE_THRESHOLDS:
        for margin in CONFIDENCE_MARGINS:
            rows.append(
                _score(
                    truth,
                    final_predictions,
                    [noise_predictions],
                    "final_noise",
                    f"confidence_gated_t{threshold:.2f}_m{margin:.2f}",
                    f"confidence-gated final_conf<{threshold:.2f} candidate_margin>{margin:.2f}",
                    lambda final_pred, candidates, t=threshold, m=margin: _confidence_gated_prediction(final_pred, candidates, t, m),
                )
            )
    return rows


def _rows_for_triple(
    truth: dict[str, str],
    final_predictions: dict[str, DebugPrediction],
    wavelet_predictions: dict[str, DebugPrediction],
    noise_predictions: dict[str, DebugPrediction],
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for final_weight, wavelet_weight in ((0.5, 0.2), (0.5, 0.3), (0.6, 0.2), (0.7, 0.1)):
        noise_weight = 1.0 - final_weight - wavelet_weight
        rows.append(
            _score(
                truth,
                final_predictions,
                [wavelet_predictions, noise_predictions],
                "final_wavelet_noise",
                f"weighted_f{final_weight:.1f}_w{wavelet_weight:.1f}_n{noise_weight:.1f}",
                f"weighted average final={final_weight:.1f} wavelet={wavelet_weight:.1f} noise={noise_weight:.1f}",
                lambda final_pred, candidates, fw=final_weight, ww=wavelet_weight, nw=noise_weight: _weighted_prediction(
                    [(final_pred, fw), (candidates[0], ww), (candidates[1], nw)]
                ),
            )
        )
    rows.append(
        _score(
            truth,
            final_predictions,
            [wavelet_predictions, noise_predictions],
            "final_wavelet_noise",
            "protected_blue",
            "protected-blue final guard",
            _protected_blue_prediction,
        )
    )
    rows.append(
        _score(
            truth,
            final_predictions,
            [wavelet_predictions, noise_predictions],
            "final_wavelet_noise",
            "red_gated",
            "red-gated wavelet/noise override",
            _red_gated_prediction,
        )
    )
    for threshold in CONFIDENCE_THRESHOLDS:
        for margin in CONFIDENCE_MARGINS:
            rows.append(
                _score(
                    truth,
                    final_predictions,
                    [wavelet_predictions, noise_predictions],
                    "final_wavelet_noise",
                    f"confidence_gated_t{threshold:.2f}_m{margin:.2f}",
                    f"confidence-gated final_conf<{threshold:.2f} candidate_margin>{margin:.2f}",
                    lambda final_pred, candidates, t=threshold, m=margin: _confidence_gated_prediction(final_pred, candidates, t, m),
                )
            )
    return rows


def tune_noise_ensemble(
    final_debug_csv: str | Path = "output/debug_latest.csv",
    wavelet_debug_csv: str | Path = "output/wavelet_ablation/high/debug_predictions.csv",
    noise_debug_csv: str | Path = "output/debug_noise_cnn_candidate.csv",
    gt_csv: str | Path = "input/ground_truth.csv",
    output_csv: str | Path = "output/ensemble_noise_results.csv",
) -> dict[str, int | float | str]:
    truth = read_ground_truth(gt_csv)
    final_predictions = read_debug_prediction_scores(final_debug_csv)
    wavelet_predictions = read_debug_prediction_scores(wavelet_debug_csv)
    noise_predictions = read_debug_prediction_scores(noise_debug_csv)
    rows = _rows_for_pair(truth, final_predictions, noise_predictions)
    rows.extend(_rows_for_triple(truth, final_predictions, wavelet_predictions, noise_predictions))

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RESULT_FIELDS})
    return max(rows, key=_best_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="search final+noise and final+wavelet+noise ensemble rules")
    parser.add_argument("--final_debug", default="output/debug_latest.csv")
    parser.add_argument("--wavelet_debug", default="output/wavelet_ablation/high/debug_predictions.csv")
    parser.add_argument("--noise_debug", default="output/debug_noise_cnn_candidate.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_csv", default="output/ensemble_noise_results.csv")
    args = parser.parse_args()
    best = tune_noise_ensemble(args.final_debug, args.wavelet_debug, args.noise_debug, args.gt, args.output_csv)
    for field in RESULT_FIELDS:
        print(f"{field}={best[field]}")


if __name__ == "__main__":
    main()
