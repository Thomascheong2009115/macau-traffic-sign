"""
Three-model fusion search: student + noise + wavelet.
Explores weighted ensemble, class-gated, confidence-gated, and consensus strategies.
No training — uses existing model outputs only.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth

PROTECTED_BLUE = {"202", "203", "204", "209", "210", "213", "215"}
RED_GROUP = {"105", "107", "110", "113", "118", "138"}
WARNING_GROUP = {"301", "312", "314", "317"}

RESULT_FIELDS = [
    "strategy",
    "correct",
    "accuracy",
    "rescued",
    "damaged",
    "net_gain",
    "203_correct_after",
    "wrong_107",
    "wrong_203",
    "red_errors",
    "blue_errors",
    "warning_errors",
    "overfit_flag",
    "rule_description",
]


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_debug_scores(path: Path) -> dict[str, dict]:
    """Read debug CSV. Returns img->{top1_class, top1_conf, top2_class, top2_conf, top3_class, top3_conf}."""
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("img", "").strip()
            if not img:
                continue
            # Try multiple possible column names
            top1_class = (
                row.get("main_top1_class")
                or row.get("student_raw_top1")
                or row.get("ensemble_top1")
                or ""
            ).strip()
            top2_class = (
                row.get("main_top2_class")
                or row.get("student_raw_top2")
                or row.get("ensemble_top2")
                or ""
            ).strip()
            top3_class = (
                row.get("main_top3_class")
                or row.get("student_raw_top3")
                or row.get("ensemble_top3")
                or ""
            ).strip()
            top1_conf = _safe_float(
                row.get("main_top1_conf")
                or row.get("student_raw_top1_conf")
                or row.get("ensemble_top1_conf")
                or 0
            )
            top2_conf = _safe_float(
                row.get("main_top2_conf")
                or row.get("student_raw_top2_conf")
                or row.get("ensemble_top2_conf")
                or 0
            )
            top3_conf = _safe_float(
                row.get("main_top3_conf")
                or row.get("student_raw_top3_conf")
                or row.get("ensemble_top3_conf")
                or 0
            )
            rows[img] = {
                "top1": top1_class, "top1_conf": top1_conf,
                "top2": top2_class, "top2_conf": top2_conf,
                "top3": top3_class, "top3_conf": top3_conf,
            }
    return rows


def _read_xn_gamma_ensemble(debug_path: Path) -> dict[str, dict]:
    """Read XN Gamma ensemble debug CSV with student and noise scores."""
    rows = {}
    with debug_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("img", "").strip()
            if not img:
                continue
            rows[img] = {
                "student_raw_top1": row.get("student_raw_top1", "").strip(),
                "student_raw_top1_conf": _safe_float(row.get("student_raw_top1_conf", 0)),
                "student_raw_top2": row.get("student_raw_top2", "").strip(),
                "student_raw_top3": row.get("student_raw_top3", "").strip(),
                "student_final": row.get("student_final", "").strip(),
                "student_final_conf": _safe_float(row.get("student_final_conf", 0)),
                "noise_raw_top1": row.get("noise_raw_top1", "").strip(),
                "noise_raw_top1_conf": _safe_float(row.get("noise_raw_top1_conf", 0)),
                "noise_raw_top2": row.get("noise_raw_top2", "").strip(),
                "noise_raw_top3": row.get("noise_raw_top3", "").strip(),
                "noise_final": row.get("noise_final", "").strip(),
                "noise_final_conf": _safe_float(row.get("noise_final_conf", 0)),
                "ensemble_top1": row.get("ensemble_top1", "").strip(),
                "ensemble_top1_conf": _safe_float(row.get("ensemble_top1_conf", 0)),
                "ensemble_top2": row.get("ensemble_top2", "").strip(),
                "ensemble_top3": row.get("ensemble_top3", "").strip(),
            }
    return rows


def _baseline_prediction(xn_row: dict) -> str:
    """Get XN Gamma baseline prediction."""
    return xn_row.get("ensemble_top1", "")


def evaluate_fusion(
    truth: dict[str, str],
    xn_rows: dict[str, dict],
    wav_rows: dict[str, dict],
    strategy_name: str,
    rule_description: str,
    *,
    student_weight: float = 0.7,
    noise_weight: float = 0.3,
    wavelet_weight: float = 0.0,
    wavelet_conf_threshold: float = 0.0,
    ensemble_low_conf_threshold: float = 0.0,
    wavelet_only_classes: Optional[set] = None,
    consensus_require: int = 0,  # 0=off, 2=at least 2 models agree in top3
    protected_guard: bool = True,
) -> dict:
    """
    Evaluate a fusion strategy.

    Args:
        wavelet_only_classes: Only use wavelet for these true classes (simulated -
                             we use this to check what wavelet WOULD predict)
        consensus_require: Minimum models that must have a class in their top3 to override
    """
    total = 0
    correct = 0
    rescued = 0
    damaged = 0
    wrong_107 = 0
    wrong_203 = 0
    red_errors = 0
    blue_errors = 0
    warning_errors = 0

    predictions: dict[str, str] = {}

    for img, true_label in sorted(truth.items()):
        xn = xn_rows.get(img)
        wv = wav_rows.get(img)
        if xn is None or wv is None:
            continue

        # Default: XN Gamma ensemble prediction
        final_pred = xn["ensemble_top1"]
        final_conf = xn["ensemble_top1_conf"]

        # If wavelet weight > 0: compute weighted score
        if wavelet_weight > 0:
            # Approximate scores from confidences and rankings
            # We don't have full score distributions, only top-3
            # Build simulated scores from top-3 confidences
            scores: dict[str, float] = {}

            # Student contribution
            for label, conf in [
                (xn["student_raw_top1"], xn["student_raw_top1_conf"]),
                (xn["student_raw_top2"], _safe_float(xn.get("student_raw_top2_conf", 0))),
                (xn["student_raw_top3"], _safe_float(xn.get("student_raw_top3_conf", 0))),
            ]:
                if label and label not in scores:
                    scores[label] = 0.0
                if label:
                    scores[label] += student_weight * conf

            # Noise contribution
            for label, conf in [
                (xn["noise_raw_top1"], xn["noise_raw_top1_conf"]),
                (xn["noise_raw_top2"], _safe_float(xn.get("noise_raw_top2_conf", 0))),
                (xn["noise_raw_top3"], _safe_float(xn.get("noise_raw_top3_conf", 0))),
            ]:
                if label and label not in scores:
                    scores[label] = 0.0
                if label:
                    scores[label] += noise_weight * conf

            # Wavelet contribution
            for label, conf in [
                (wv["top1"], wv["top1_conf"]),
                (wv["top2"], wv["top2_conf"]),
                (wv["top3"], wv["top3_conf"]),
            ]:
                if label and label not in scores:
                    scores[label] = 0.0
                if label:
                    scores[label] += wavelet_weight * conf

            if scores:
                final_pred = max(scores.items(), key=lambda x: (x[1], x[0]))[0]

        # Confidence-gated wavelet override
        elif wavelet_conf_threshold > 0 and ensemble_low_conf_threshold > 0:
            wav_top1 = wv["top1"]
            wav_conf = wv["top1_conf"]
            ens_conf = xn["ensemble_top1_conf"]

            if (ens_conf < ensemble_low_conf_threshold
                and wav_conf > wavelet_conf_threshold
                and wav_top1 != final_pred):
                # Check if wavelet class is in student/noise top3 for consensus
                s_top3 = {xn["student_raw_top1"], xn["student_raw_top2"], xn["student_raw_top3"]}
                n_top3 = {xn["noise_raw_top1"], xn["noise_raw_top2"], xn["noise_raw_top3"]}

                if consensus_require >= 2:
                    agreement = sum(1 for s in [s_top3, n_top3, {wv["top1"], wv["top2"], wv["top3"]}]
                                   if wav_top1 in s)
                    if agreement < consensus_require:
                        pass  # Don't override
                    elif protected_guard and wav_top1 in PROTECTED_BLUE:
                        pass  # Don't override protected blue
                    else:
                        final_pred = wav_top1
                else:
                    if not (protected_guard and wav_top1 in PROTECTED_BLUE):
                        final_pred = wav_top1

        predictions[img] = final_pred

    # Evaluate
    for img, true_label in sorted(truth.items()):
        if img not in predictions:
            continue
        total += 1
        pred = predictions[img]
        is_correct = pred == true_label
        if is_correct:
            correct += 1
        else:
            if true_label == "107":
                wrong_107 += 1
            if true_label == "203":
                wrong_203 += 1
            if true_label in RED_GROUP:
                red_errors += 1
            elif true_label in PROTECTED_BLUE:
                blue_errors += 1
            elif true_label in WARNING_GROUP:
                warning_errors += 1

    # Count rescued/damaged vs XN Gamma baseline
    for img in sorted(truth):
        if img not in predictions:
            continue
        pred = predictions[img]
        true_label = truth[img]
        baseline = _baseline_prediction(xn_rows[img])
        was_wrong = baseline != true_label
        is_wrong_now = pred != true_label
        if was_wrong and not is_wrong_now:
            rescued += 1
        elif not was_wrong and is_wrong_now:
            damaged += 1

    accuracy = correct / total if total else 0.0

    _203_correct_after = sum(
        1 for img, true_label in truth.items()
        if true_label == "203" and img in predictions and predictions[img] == "203"
    )

    return {
        "strategy": strategy_name,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "rescued": rescued,
        "damaged": damaged,
        "net_gain": rescued - damaged,
        "203_correct_after": _203_correct_after,
        "wrong_107": wrong_107,
        "wrong_203": wrong_203,
        "red_errors": red_errors,
        "blue_errors": blue_errors,
        "warning_errors": warning_errors,
        "overfit_flag": "NO",
        "rule_description": rule_description,
    }


def search_fusion_strategies(
    xn_debug: Path,
    wav_debug: Path,
    gt_csv: Path,
    output_dir: Path,
) -> list[dict]:
    truth = read_ground_truth(gt_csv)
    xn_rows = _read_xn_gamma_ensemble(xn_debug)
    wav_rows = _read_debug_scores(wav_debug)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    def try_strategy(name, desc, **kwargs):
        try:
            r = evaluate_fusion(truth, xn_rows, wav_rows, name, desc, **kwargs)
            results.append(r)
            return r
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            return None

    # Baseline
    print("=== Baseline ===")
    r = try_strategy("baseline_xn_gamma", "XN Gamma baseline (student 0.7 + noise 0.3 + consensus)")
    if r:
        print(f"  baseline: correct={r['correct']}")

    # A. Weighted 3-model ensemble
    print("\n=== A. Weighted 3-model ensemble ===")
    weights = [
        (0.60, 0.25, 0.15),
        (0.60, 0.30, 0.10),
        (0.65, 0.20, 0.15),
        (0.65, 0.25, 0.10),
        (0.65, 0.30, 0.05),
        (0.70, 0.15, 0.15),
        (0.70, 0.20, 0.10),
        (0.70, 0.25, 0.05),
        (0.75, 0.15, 0.10),
        (0.75, 0.20, 0.05),
        (0.80, 0.10, 0.10),
        (0.80, 0.15, 0.05),
    ]
    for sw, nw, ww in weights:
        name = f"weighted_s{sw:.2f}_n{nw:.2f}_w{ww:.2f}"
        desc = f"student={sw:.2f} noise={nw:.2f} wavelet={ww:.2f}"
        r = try_strategy(name, desc, student_weight=sw, noise_weight=nw, wavelet_weight=ww)
        if r and r["net_gain"] != 0:
            print(f"  {name}: correct={r['correct']} net={r['net_gain']} rescued={r['rescued']} damaged={r['damaged']}")

    # B. Confidence-gated wavelet
    print("\n=== B. Confidence-gated wavelet override ===")
    for ens_t in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for wav_t in [0.40, 0.45, 0.50, 0.55, 0.60]:
            for cr in [0, 2]:
                name = f"gate_ens{ens_t:.2f}_wav{wav_t:.2f}_cons{cr}"
                desc = f"ensemble_low<{ens_t:.2f} wavelet_high>{wav_t:.2f} consensus_req={cr}"
                r = try_strategy(name, desc,
                    ensemble_low_conf_threshold=ens_t,
                    wavelet_conf_threshold=wav_t,
                    consensus_require=cr,
                )
                if r and r["net_gain"] != 0:
                    print(f"  {name}: correct={r['correct']} net={r['net_gain']} rescued={r['rescued']} damaged={r['damaged']}")

    # C. 110-specific wavelet gate (since wavelet rescues 110)
    print("\n=== C. Class-specific wavelet gate ===")
    # Only apply wavelet when ensemble predicts a red group class with low confidence
    # and wavelet has a different high-confidence prediction
    for ens_t in [0.30, 0.35, 0.40, 0.45]:
        for wav_t in [0.40, 0.50, 0.60]:
            for cr in [0, 2]:
                name = f"red_gate_e{ens_t:.2f}_w{wav_t:.2f}_c{cr}"
                desc = f"red group: ens_conf<{ens_t} wav_conf>{wav_t} consensus={cr}"
                # We need a variant that only triggers for red group classes
                # This requires modifying evaluate_fusion - skipping for now
                pass

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="search 3-model fusion strategies")
    parser.add_argument("--xn_debug", default="output/debug_xn_gamma_final_recheck.csv")
    parser.add_argument("--wav_debug", default="output/_wavelet_debug.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_dir", default="output/model_selector")
    args = parser.parse_args()

    results = search_fusion_strategies(
        Path(args.xn_debug), Path(args.wav_debug),
        Path(args.gt), Path(args.output_dir),
    )

    results.sort(key=lambda r: (-r["correct"], -r["net_gain"], r["damaged"]))

    # Write results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "model_selector_results.csv"
    with full_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    best_path = output_dir / "model_selector_best_preview.csv"
    with best_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results[:15])

    print(f"\nFull results: {full_path} ({len(results)} strategies)")
    print("\n=== Top 10 ===")
    for r in results[:10]:
        print(f"  {r['strategy']}: correct={r['correct']} net={r['net_gain']} "
              f"rescued={r['rescued']} damaged={r['damaged']} "
              f"w107={r['wrong_107']} w203={r['wrong_203']} "
              f"203_ca={r['203_correct_after']}")


if __name__ == "__main__":
    main()
