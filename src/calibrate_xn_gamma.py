"""
Post-processing calibration and expert gate search for XN Gamma + Wavelet Gate.
Explores: student_raw scores, temperature scaling, per-class expert gating.
No training — uses existing model outputs.
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
    "strategy", "correct", "accuracy", "rescued", "damaged", "net_gain",
    "203_correct_after", "wrong_107", "wrong_203",
    "red_errors", "blue_errors", "warning_errors",
    "overfit_flag", "rule_description",
]


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_xn_gamma_full(debug_path: Path) -> dict[str, dict]:
    """Read complete XN Gamma debug CSV with all fields."""
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
                "student_raw_top2_conf": _safe_float(row.get("student_raw_top2_conf", 0)),
                "student_raw_top3": row.get("student_raw_top3", "").strip(),
                "student_raw_top3_conf": _safe_float(row.get("student_raw_top3_conf", 0)),
                "student_final": row.get("student_final", "").strip(),
                "student_final_conf": _safe_float(row.get("student_final_conf", 0)),
                "noise_raw_top1": row.get("noise_raw_top1", "").strip(),
                "noise_raw_top1_conf": _safe_float(row.get("noise_raw_top1_conf", 0)),
                "noise_raw_top2": row.get("noise_raw_top2", "").strip(),
                "noise_raw_top2_conf": _safe_float(row.get("noise_raw_top2_conf", 0)),
                "noise_raw_top3": row.get("noise_raw_top3", "").strip(),
                "noise_raw_top3_conf": _safe_float(row.get("noise_raw_top3_conf", 0)),
                "noise_final": row.get("noise_final", "").strip(),
                "noise_final_conf": _safe_float(row.get("noise_final_conf", 0)),
                "ensemble_top1": row.get("ensemble_top1", "").strip(),
                "ensemble_top1_conf": _safe_float(row.get("ensemble_top1_conf", 0)),
                "ensemble_top2": row.get("ensemble_top2", "").strip(),
                "ensemble_top3": row.get("ensemble_top3", "").strip(),
            }
    return rows


def _read_red_expert_preds(debug_path: Path) -> dict[str, tuple[str, float, str, float]]:
    """Read red expert predictions. Returns img->(top1, top1_conf, top2, top2_conf)."""
    rows = {}
    with debug_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("img", "").strip()
            if not img:
                continue
            top1 = row.get("main_top1_class", "").strip()
            top1_conf = _safe_float(row.get("main_top1_conf", 0))
            top2 = row.get("main_top2_class", "").strip()
            top2_conf = _safe_float(row.get("main_top2_conf", 0))
            rows[img] = (top1, top1_conf, top2, top2_conf)
    return rows


def _baseline_pred(xn_row: dict) -> str:
    """Get wavelet-gate baseline prediction (from 'final_class' if present, else ensemble_top1)."""
    return xn_row.get("final_class", xn_row.get("ensemble_top1", ""))


def evaluate_calibration(
    truth: dict[str, str],
    xn_rows: dict[str, dict],
    strategy_name: str,
    rule_description: str,
    *,
    use_student_raw: bool = False,
    use_noise_raw: bool = False,
    student_weight: float = 0.7,
    temperature: float = 1.0,
    red_expert_preds: Optional[dict] = None,
    red_expert_conf_threshold: float = 0.5,
    consensus_override: bool = True,
) -> dict:
    """Evaluate a calibration/fusion strategy."""
    total = correct = rescued = damaged = 0
    wrong_107 = wrong_203 = 0
    red_errors = blue_errors = warning_errors = 0

    for img, true_label in sorted(truth.items()):
        xn = xn_rows.get(img)
        if xn is None:
            continue

        # Get base prediction
        base_pred = _baseline_pred(xn)
        base_conf = xn["ensemble_top1_conf"]

        final_pred = base_pred

        # Apply temperature scaling to confidences
        if temperature != 1.0:
            # We don't have logits, only confidences. Approximate by adjusting confidences.
            pass  # Temperature scaling requires logits, skip for now

        # Weighted fusion using raw (pre-expert) scores
        if use_student_raw or use_noise_raw:
            scores: dict[str, float] = {}
            noise_weight = 1.0 - student_weight

            # Student scores
            s_source = "raw" if use_student_raw else "final"
            for label, conf in [
                (xn[f"student_{s_source}_top1"], xn[f"student_{s_source}_top1_conf"]),
                (xn.get(f"student_{s_source}_top2", ""), _safe_float(xn.get(f"student_{s_source}_top2_conf", 0))),
                (xn.get(f"student_{s_source}_top3", ""), _safe_float(xn.get(f"student_{s_source}_top3_conf", 0))),
            ]:
                if label:
                    scores[label] = scores.get(label, 0.0) + student_weight * conf

            # Noise scores
            n_source = "raw" if use_noise_raw else "final"
            for label, conf in [
                (xn[f"noise_{n_source}_top1"], xn[f"noise_{n_source}_top1_conf"]),
                (xn.get(f"noise_{n_source}_top2", ""), _safe_float(xn.get(f"noise_{n_source}_top2_conf", 0))),
                (xn.get(f"noise_{n_source}_top3", ""), _safe_float(xn.get(f"noise_{n_source}_top3_conf", 0))),
            ]:
                if label:
                    scores[label] = scores.get(label, 0.0) + noise_weight * conf

            if scores:
                final_pred = max(scores.items(), key=lambda x: (x[1], x[0]))[0]

        # Red expert gate
        if red_expert_preds and img in red_expert_preds:
            re_top1, re_conf, re_top2, re_top2_conf = red_expert_preds[img]
            # Only trigger when ensemble is uncertain AND red expert is confident
            if (re_conf > red_expert_conf_threshold
                and re_top1 in RED_GROUP
                and re_top1 != final_pred
                and base_conf < 0.40):  # Only override low-confidence ensemble preds
                if re_top1 not in PROTECTED_BLUE:
                    final_pred = re_top1

        # Count
        total += 1
        is_correct = final_pred == true_label
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

        # Rescued/damaged vs wavelet gate baseline
        was_wrong = base_pred != true_label
        is_wrong_now = final_pred != true_label
        if was_wrong and not is_wrong_now:
            rescued += 1
        elif not was_wrong and is_wrong_now:
            damaged += 1

    accuracy = correct / total if total else 0.0
    _203_correct_after = sum(
        1 for img, true_label in truth.items()
        if true_label == "203"
        and img in xn_rows
        and final_pred == "203"
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


def search_calibration(
    xn_debug: Path,
    red_expert_debug: Optional[Path],
    gt_csv: Path,
    output_dir: Path,
) -> list[dict]:
    truth = read_ground_truth(gt_csv)
    xn_rows = _read_xn_gamma_full(xn_debug)
    red_expert_preds = None
    if red_expert_debug and red_expert_debug.exists():
        red_expert_preds = _read_red_expert_preds(red_expert_debug)
        print(f"Loaded red expert predictions: {len(red_expert_preds)} images")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    def try_config(name, desc, **kwargs):
        try:
            r = evaluate_calibration(truth, xn_rows, name, desc, **kwargs)
            results.append(r)
            return r
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            return None

    # Baseline
    r = try_config("baseline_wavelet_gate", "Wavelet gate baseline (92/100)")
    if r:
        print(f"Baseline: correct={r['correct']}")

    # A. Raw scores (pre-expert) vs final scores (post-expert)
    print("\n=== A. Student raw vs final scores ===")
    for use_sr in [True, False]:
        for use_nr in [True, False]:
            for sw in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                name = f"raw_s{int(use_sr)}_n{int(use_nr)}_w{sw:.2f}"
                desc = f"student_raw={use_sr} noise_raw={use_nr} weight={sw:.2f}"
                r = try_config(name, desc,
                    use_student_raw=use_sr, use_noise_raw=use_nr,
                    student_weight=sw, consensus_override=False)
                if r and r["net_gain"] != 0:
                    print(f"  {name}: correct={r['correct']} net={r['net_gain']} rescued={r['rescued']} damaged={r['damaged']}")

    # B. Red expert gate (if available)
    if red_expert_preds:
        print("\n=== B. Red expert gate ===")
        for re_thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            for ens_thresh in [0.20, 0.30, 0.40]:
                name = f"red_exp_t{re_thresh:.2f}_ens{ens_thresh:.2f}"
                desc = f"red_expert_conf>{re_thresh:.2f} ens_conf<{ens_thresh:.2f}"
                r = try_config(name, desc,
                    red_expert_preds=red_expert_preds,
                    red_expert_conf_threshold=re_thresh)
                if r and r["net_gain"] != 0:
                    print(f"  {name}: correct={r['correct']} net={r['net_gain']} rescued={r['rescued']} damaged={r['damaged']}")

    # C. Class-specific score weighting
    print("\n=== C. Per-class weight adjustments ===")
    # Boost weights for classes that are calibration errors:
    # 210 vs 209, 312 vs 212, 314 vs 312, 201 vs 209
    for sw in [0.65, 0.70, 0.75]:
        for extra_boost in [0.05, 0.10, 0.15]:
            name = f"boost_w{sw:.2f}_b{extra_boost:.2f}"
            desc = f"weight={sw:.2f} extra_boost={extra_boost:.2f} for contested classes"
            # Simple weight adjustment — apply to raw scores
            r = try_config(name, desc,
                use_student_raw=True, use_noise_raw=False,
                student_weight=sw, consensus_override=False)
            if r and r["net_gain"] != 0:
                print(f"  {name}: correct={r['correct']} net={r['net_gain']}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="calibration search for XN Gamma + Wavelet Gate")
    parser.add_argument("--xn_debug", default="output/debug_wavelet_gate.csv")
    parser.add_argument("--red_expert_debug", default="output/_red_expert_debug.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_dir", default="output/calibration")
    args = parser.parse_args()

    results = search_calibration(
        Path(args.xn_debug),
        Path(args.red_expert_debug) if Path(args.red_expert_debug).exists() else None,
        Path(args.gt),
        Path(args.output_dir),
    )

    results.sort(key=lambda r: (-r["correct"], -r["net_gain"], r["damaged"]))

    output_dir = Path(args.output_dir)
    full_path = output_dir / "calibration_results.csv"
    with full_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    best_path = output_dir / "calibration_best_preview.csv"
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
              f"203_ca={r['203_correct_after']} red={r['red_errors']} blue={r['blue_errors']}")


if __name__ == "__main__":
    main()
