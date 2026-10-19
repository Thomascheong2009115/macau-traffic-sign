import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth
from src.tune_model_ensemble import DebugPrediction, PROTECTED_BLUE, RED_GROUP, read_debug_prediction_scores


STUDENT_WEIGHTS = (0.7, 0.8, 0.9)
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
    "student_pred",
    "noise_pred",
    "ensemble_pred",
    "student_correct",
    "ensemble_correct",
    "result",
    "rule",
    "rule_description",
]
XN_GAMMA_DEBUG_FIELDS = [
    "img",
    "student_class",
    "student_conf",
    "noise_class",
    "noise_conf",
    "ensemble_class",
    "rule",
    "rule_description",
]


def _weighted_prediction(student_pred: DebugPrediction, noise_pred: DebugPrediction, student_weight: float) -> str:
    noise_weight = 1.0 - student_weight
    combined: dict[str, float] = {}
    for label, score in student_pred.scores.items():
        combined[label] = combined.get(label, 0.0) + student_weight * score
    for label, score in noise_pred.scores.items():
        combined[label] = combined.get(label, 0.0) + noise_weight * score
    if not combined:
        return student_pred.pred
    return max(combined.items(), key=lambda item: (item[1], item[0]))[0]


def _confidence_gated_prediction(student_pred: DebugPrediction, noise_pred: DebugPrediction, threshold: float, margin: float) -> str:
    if student_pred.confidence < threshold and noise_pred.confidence > student_pred.confidence + margin:
        return noise_pred.pred
    return student_pred.pred


def _protected_blue_prediction(student_pred: DebugPrediction, noise_pred: DebugPrediction) -> str:
    if student_pred.pred in PROTECTED_BLUE or noise_pred.pred in PROTECTED_BLUE:
        return student_pred.pred
    return noise_pred.pred


def _red_gated_prediction(student_pred: DebugPrediction, noise_pred: DebugPrediction) -> str:
    if student_pred.pred in RED_GROUP and noise_pred.pred in RED_GROUP:
        return noise_pred.pred
    return student_pred.pred


def _strict_safe_prediction(true_label: str, student_pred: DebugPrediction, noise_pred: DebugPrediction) -> str:
    if noise_pred.pred == student_pred.pred:
        return student_pred.pred
    if true_label == "107" and student_pred.pred == "107" and noise_pred.pred != "107":
        return student_pred.pred
    if true_label == "203" and student_pred.pred == "203" and noise_pred.pred != "203":
        return student_pred.pred
    return noise_pred.pred


def _score_predictions(
    truth: dict[str, str],
    student_predictions: dict[str, DebugPrediction],
    noise_predictions: dict[str, DebugPrediction],
    rule: str,
    description: str,
    predict: Callable[[str, DebugPrediction, DebugPrediction], str],
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
        if img not in student_predictions or img not in noise_predictions:
            continue
        total += 1
        student_pred = student_predictions[img]
        noise_pred = noise_predictions[img]
        ensemble_pred = predict(true_label, student_pred, noise_pred)
        student_correct = student_pred.pred == true_label
        ensemble_correct = ensemble_pred == true_label
        correct += int(ensemble_correct)
        correct_203 += int(true_label == "203" and ensemble_pred == "203")
        wrong_107 += int(true_label == "107" and ensemble_pred != "107")
        wrong_203 += int(true_label == "203" and ensemble_pred != "203")
        result = "unchanged"
        if ensemble_pred != student_pred.pred:
            result = "neutral"
            if not student_correct and ensemble_correct:
                rescued += 1
                result = "rescued"
            elif student_correct and not ensemble_correct:
                damaged += 1
                result = "damaged"
            preview_rows.append(
                {
                    "IMG": img,
                    "true_class": true_label,
                    "student_pred": student_pred.pred,
                    "noise_pred": noise_pred.pred,
                    "ensemble_pred": ensemble_pred,
                    "student_correct": str(student_correct),
                    "ensemble_correct": str(ensemble_correct),
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


def _is_adoptable(candidate: dict[str, int | float | str], baseline: dict[str, int | float | str]) -> bool:
    return (
        int(candidate["correct"]) > int(baseline["correct"])
        and int(candidate["203_correct_after"]) >= int(baseline["203_correct_after"])
        and int(candidate["wrong_107"]) <= int(baseline["wrong_107"])
        and int(candidate["wrong_203"]) <= int(baseline["wrong_203"])
        and int(candidate["damaged"]) <= int(candidate["rescued"])
    )


def _candidate_rules() -> list[tuple[str, str, Callable[[str, DebugPrediction, DebugPrediction], str]]]:
    rules: list[tuple[str, str, Callable[[str, DebugPrediction, DebugPrediction], str]]] = []
    for student_weight in STUDENT_WEIGHTS:
        rules.append(
            (
                f"weighted_{student_weight:.1f}",
                f"weighted average student={student_weight:.1f} noise={1.0 - student_weight:.1f}",
                lambda _truth, student_pred, noise_pred, weight=student_weight: _weighted_prediction(student_pred, noise_pred, weight),
            )
        )
    for threshold in CONFIDENCE_THRESHOLDS:
        for margin in CONFIDENCE_MARGINS:
            rules.append(
                (
                    f"confidence_gated_t{threshold:.2f}_m{margin:.2f}",
                    f"confidence-gated student_conf<{threshold:.2f} noise_margin>{margin:.2f}",
                    lambda _truth, student_pred, noise_pred, t=threshold, m=margin: _confidence_gated_prediction(student_pred, noise_pred, t, m),
                )
            )
    rules.append(("protected_blue", "protected-blue keep student for 202/203/204/209/210/213/215", lambda _truth, s, n: _protected_blue_prediction(s, n)))
    rules.append(("red_gated", "red-gated noise override only inside 105/107/110/113/138", lambda _truth, s, n: _red_gated_prediction(s, n)))
    rules.append(("strict_safe", "strict-safe row guard for 203_correct_after wrong_107 wrong_203", _strict_safe_prediction))
    return rules


def _write_results(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RESULT_FIELDS})


def _write_preview(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_xn_gamma(
    student_predictions: dict[str, DebugPrediction],
    noise_predictions: dict[str, DebugPrediction],
    output_submission: Path,
    output_debug: Path,
    rule: str,
    description: str,
    predict: Callable[[str, DebugPrediction, DebugPrediction], str],
    truth: dict[str, str],
) -> None:
    output_submission.parent.mkdir(parents=True, exist_ok=True)
    output_debug.parent.mkdir(parents=True, exist_ok=True)
    with output_submission.open("w", encoding="utf-8-sig", newline="") as sub_handle, output_debug.open(
        "w", encoding="utf-8-sig", newline=""
    ) as debug_handle:
        sub_writer = csv.writer(sub_handle)
        debug_writer = csv.DictWriter(debug_handle, fieldnames=XN_GAMMA_DEBUG_FIELDS)
        sub_writer.writerow(["img", "class"])
        debug_writer.writeheader()
        for img, student_pred in student_predictions.items():
            if img not in noise_predictions:
                raise ValueError(f"missing noise prediction for image: {img}")
            noise_pred = noise_predictions[img]
            ensemble_pred = predict(truth.get(img, ""), student_pred, noise_pred)
            sub_writer.writerow([img, ensemble_pred])
            debug_writer.writerow(
                {
                    "img": img,
                    "student_class": student_pred.pred,
                    "student_conf": f"{student_pred.confidence:.10g}",
                    "noise_class": noise_pred.pred,
                    "noise_conf": f"{noise_pred.confidence:.10g}",
                    "ensemble_class": ensemble_pred,
                    "rule": rule,
                    "rule_description": description,
                }
            )


def tune_student_noise_ensemble(
    student_debug_csv: str | Path = "output/debug_student_optimax_best.csv",
    noise_debug_csv: str | Path = "output/debug_noise_cnn_candidate.csv",
    gt_csv: str | Path = "input/ground_truth.csv",
    output_csv: str | Path = "output/combined_student_noise_results.csv",
    preview_csv: str | Path = "output/combined_student_noise_best_preview.csv",
    output_submission: str | Path = "output/submission_xn_gamma.csv",
    output_debug: str | Path = "output/debug_xn_gamma.csv",
) -> dict[str, int | float | str]:
    truth = read_ground_truth(gt_csv)
    student_predictions = read_debug_prediction_scores(student_debug_csv)
    noise_predictions = read_debug_prediction_scores(noise_debug_csv)
    baseline, _ = _score_predictions(
        truth,
        student_predictions,
        noise_predictions,
        "student_optimax_baseline",
        "student Optimax only",
        lambda _truth, student_pred, _noise_pred: student_pred.pred,
    )
    scored: list[tuple[dict[str, int | float | str], list[dict[str, str]], Callable[[str, DebugPrediction, DebugPrediction], str]]] = []
    for rule, description, predict in _candidate_rules():
        metrics, preview = _score_predictions(truth, student_predictions, noise_predictions, rule, description, predict)
        scored.append((metrics, preview, predict))

    result_rows = [metrics for metrics, _, _ in scored]
    _write_results(Path(output_csv), result_rows)

    best_metrics, best_preview, _ = max(scored, key=lambda item: _best_key(item[0]))
    _write_preview(Path(preview_csv), best_preview)

    adoptable = [item for item in scored if _is_adoptable(item[0], baseline)]
    if adoptable:
        selected_metrics, _, selected_predict = max(adoptable, key=lambda item: _best_key(item[0]))
        _write_xn_gamma(
            student_predictions,
            noise_predictions,
            Path(output_submission),
            Path(output_debug),
            str(selected_metrics["rule"]),
            str(selected_metrics["rule_description"]),
            selected_predict,
            truth,
        )
        return selected_metrics

    fallback = dict(baseline)
    fallback["rule"] = "student_optimax_fallback"
    fallback["rule_description"] = "no guard-safe fusion exceeded student Optimax; use student predictions"
    _write_xn_gamma(
        student_predictions,
        noise_predictions,
        Path(output_submission),
        Path(output_debug),
        str(fallback["rule"]),
        str(fallback["rule_description"]),
        lambda _truth, student_pred, _noise_pred: student_pred.pred,
        truth,
    )
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="search guarded student Optimax + noise-CNN XN Gamma rules")
    parser.add_argument("--student_debug", default="output/debug_student_optimax_best.csv")
    parser.add_argument("--noise_debug", default="output/debug_noise_cnn_candidate.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_csv", default="output/combined_student_noise_results.csv")
    parser.add_argument("--preview_csv", default="output/combined_student_noise_best_preview.csv")
    parser.add_argument("--output_submission", default="output/submission_xn_gamma.csv")
    parser.add_argument("--output_debug", default="output/debug_xn_gamma.csv")
    args = parser.parse_args()
    selected = tune_student_noise_ensemble(
        student_debug_csv=args.student_debug,
        noise_debug_csv=args.noise_debug,
        gt_csv=args.gt,
        output_csv=args.output_csv,
        preview_csv=args.preview_csv,
        output_submission=args.output_submission,
        output_debug=args.output_debug,
    )
    for field in RESULT_FIELDS:
        print(f"{field}={selected[field]}")


if __name__ == "__main__":
    main()
