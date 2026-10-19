"""
XN Gamma error audit script.
Reads student/noise debug CSVs, ground truth, and ensemble debug CSV.
Outputs error decomposition, rescue candidates, and remaining failure summary.

No ground-truth-based rules — GT is used only for evaluation and candidate selection.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate_test_predictions import read_ground_truth
from src.tune_model_ensemble import DebugPrediction, read_debug_prediction_scores

PROTECTED_BLUE = {"202", "203", "204", "209", "210", "213", "215"}
RED_GROUP = {"105", "107", "110", "113", "138"}

ERROR_DECOMPOSITION_FIELDS = [
    "img",
    "true_class",
    "ensemble_pred",
    "student_raw_top1",
    "student_raw_top2",
    "student_raw_top3",
    "student_final",
    "noise_raw_top1",
    "noise_raw_top2",
    "noise_raw_top3",
    "noise_final",
    "true_in_student_raw_top3",
    "true_in_noise_raw_top3",
    "true_in_union_top3",
    "true_in_student_final_top3",
    "true_in_noise_final_top3",
    "ensemble_top2",
    "ensemble_top3",
    "student_top1_conf",
    "noise_top1_conf",
    "ensemble_top1_conf",
    "error_group",
    "recommended_fix_type",
    "student_conf_margin_top1_vs_true",
    "noise_conf_margin_top1_vs_true",
    "difficult_flag",
    "red_group_flag",
    "blue_group_flag",
]

RESCUE_FIELDS = [
    "img",
    "true_class",
    "ensemble_pred",
    "true_in_student_raw_top3",
    "true_in_noise_raw_top3",
    "true_in_union_top3",
    "error_group",
    "recommended_fix_type",
    "rescue_notes",
]

FAILURE_SUMMARY_FIELDS = [
    "category",
    "count",
    "classes_affected",
    "example_images",
    "recommended_action",
    "estimated_gain",
]


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_raw_top3(debug_path: Path) -> dict[str, list[tuple[str, float]]]:
    """Read raw main_top{1,2,3}_class and _conf from infer.py debug CSV (pre-expert)."""
    raw: dict[str, list[tuple[str, float]]] = {}
    with debug_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img = row.get("img", "").strip()
            if not img:
                continue
            top3: list[tuple[str, float]] = []
            for rank in (1, 2, 3):
                label = row.get(f"main_top{rank}_class", "").strip()
                conf = _safe_float(row.get(f"main_top{rank}_conf", "0"))
                if label:
                    top3.append((label, conf))
            raw[img] = top3
    return raw


def _read_ensemble_debug(debug_path: Path) -> dict[str, dict[str, str]]:
    """Read ensemble debug CSV. Returns img->row dict."""
    rows: dict[str, dict[str, str]] = {}
    with debug_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img = row.get("img", "").strip()
            if img:
                rows[img] = row
    return rows


def _classify_error_group(img: str,
                          true_class: str,
                          ensemble_pred: str,
                          student_raw_top3: list[tuple[str, float]],
                          noise_raw_top3: list[tuple[str, float]],
                          student_final: str,
                          noise_final: str) -> str:
    """Classify error into one of: calibration, model_blind, expert_fired, near_miss, model_division."""
    s_labels = {label for label, _ in student_raw_top3}
    n_labels = {label for label, _ in noise_raw_top3}
    union = s_labels | n_labels

    if true_class in union:
        # At least one model saw it
        if true_class in s_labels and true_class in n_labels:
            return "calibration_both_see"
        elif true_class in s_labels:
            return "calibration_student_only"
        else:
            return "calibration_noise_only"
    else:
        # Neither model top3 has it
        # Check if expert fired
        if ensemble_pred in PROTECTED_BLUE and true_class not in PROTECTED_BLUE:
            return "expert_fired"
        # Check near miss: is the predicted class visually/semantically very close?
        return "model_blind"


def _recommend_fix_type(error_group: str,
                        true_class: str,
                        ensemble_pred: str) -> str:
    """Recommend fix type based on error group and class relationships."""
    if error_group.startswith("calibration"):
        return "rule_calibration"
    if error_group == "expert_fired":
        return "expert_guard_review"
    # model_blind — check if classes are in red group or blue group
    if true_class in RED_GROUP:
        return "specialist_candidate"
    if true_class in PROTECTED_BLUE:
        return "crop_review"
    return "data_needed"


def _calculate_margin(top3: list[tuple[str, float]], true_label: str) -> float:
    """How far is true_label confidence from top1 confidence. Positive means top1 > true."""
    if not top3:
        return 999.0
    top1_conf = top3[0][1]
    for label, conf in top3:
        if label == true_label:
            return top1_conf - conf
    return 999.0


def decompose_errors(
    student_debug: Path,
    noise_debug: Path,
    ensemble_debug: Path,
    gt_csv: Path,
    output_dir: Path,
) -> None:
    truth = read_ground_truth(gt_csv)
    student_predictions = read_debug_prediction_scores(student_debug)
    noise_predictions = read_debug_prediction_scores(noise_debug)
    student_raw = _read_raw_top3(student_debug)
    noise_raw = _read_raw_top3(noise_debug)
    ensemble_rows = _read_ensemble_debug(ensemble_debug)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build error decomposition ---
    error_rows: list[dict[str, str]] = []
    rescue_rows: list[dict[str, str]] = []

    for img, true_label in sorted(truth.items()):
        ensemble_row = ensemble_rows.get(img, {})
        ensemble_pred = ensemble_row.get("ensemble_top1", ensemble_row.get("ensemble_class", "")).strip()
        if not ensemble_pred:
            ensemble_pred = student_predictions.get(img, DebugPrediction(img, "", 0.0, {})).pred
        if ensemble_pred == true_label:
            continue  # correct prediction

        sp = student_predictions.get(img, DebugPrediction(img, "", 0.0, {}))
        np = noise_predictions.get(img, DebugPrediction(img, "", 0.0, {}))
        s_raw = student_raw.get(img, [])
        n_raw = noise_raw.get(img, [])
        s_labels = {label for label, _ in s_raw}
        n_labels = {label for label, _ in n_raw}
        union = s_labels | n_labels

        # Also check student_final / noise_final top3
        s_final_scores = sorted(sp.scores.items(), key=lambda x: (-x[1], x[0]))
        n_final_scores = sorted(np.scores.items(), key=lambda x: (-x[1], x[0]))
        s_final_labels = {label for label, _ in s_final_scores[:3]}
        n_final_labels = {label for label, _ in n_final_scores[:3]}

        error_group = _classify_error_group(
            img, true_label, ensemble_pred, s_raw, n_raw, sp.pred, np.pred
        )
        fix_type = _recommend_fix_type(error_group, true_label, ensemble_pred)

        # Ensemble top2/top3 from debug CSV
        ensemble_top2 = ensemble_row.get("ensemble_top2", ensemble_row.get("main_top2_class", "")).strip()
        ensemble_top3 = ensemble_row.get("ensemble_top3", ensemble_row.get("main_top3_class", "")).strip()

        student_margin = _calculate_margin(s_raw, true_label)
        noise_margin = _calculate_margin(n_raw, true_label)

        difficult_flag = "YES" if (student_margin > 0.3 and noise_margin > 0.3) else "NO"

        error_rows.append({
            "img": img,
            "true_class": true_label,
            "ensemble_pred": ensemble_pred,
            "student_raw_top1": s_raw[0][0] if len(s_raw) > 0 else "",
            "student_raw_top2": s_raw[1][0] if len(s_raw) > 1 else "",
            "student_raw_top3": s_raw[2][0] if len(s_raw) > 2 else "",
            "student_final": sp.pred,
            "noise_raw_top1": n_raw[0][0] if len(n_raw) > 0 else "",
            "noise_raw_top2": n_raw[1][0] if len(n_raw) > 1 else "",
            "noise_raw_top3": n_raw[2][0] if len(n_raw) > 2 else "",
            "noise_final": np.pred,
            "true_in_student_raw_top3": str(true_label in s_labels),
            "true_in_noise_raw_top3": str(true_label in n_labels),
            "true_in_union_top3": str(true_label in union),
            "true_in_student_final_top3": str(true_label in s_final_labels),
            "true_in_noise_final_top3": str(true_label in n_final_labels),
            "ensemble_top2": ensemble_top2,
            "ensemble_top3": ensemble_top3,
            "student_top1_conf": f"{s_raw[0][1]:.10g}" if len(s_raw) > 0 else "",
            "noise_top1_conf": f"{n_raw[0][1]:.10g}" if len(n_raw) > 0 else "",
            "ensemble_top1_conf": ensemble_row.get("ensemble_top1_conf", ensemble_row.get("student_conf", "")).strip(),
            "error_group": error_group,
            "recommended_fix_type": fix_type,
            "student_conf_margin_top1_vs_true": f"{student_margin:.10g}",
            "noise_conf_margin_top1_vs_true": f"{noise_margin:.10g}",
            "difficult_flag": difficult_flag,
            "red_group_flag": str(true_label in RED_GROUP),
            "blue_group_flag": str(true_label in PROTECTED_BLUE),
        })

        # Build rescue candidate row
        rescue_notes = ""
        if true_label in union:
            if true_label in s_labels and true_label in n_labels:
                rescue_notes = "both models see true in top3; calibration can lift"
            elif true_label in s_labels:
                rescue_notes = "only student sees true; student_weight boost or student-primary gating may help"
            else:
                rescue_notes = "only noise sees true; noise-primary gating or student_conf_threshold may help"
        elif error_group == "expert_fired":
            rescue_notes = "expert triggered incorrectly; review expert_trigger_rank or expert_mode"
        else:
            s_top1 = s_raw[0] if s_raw else ("", 0.0)
            n_top1 = n_raw[0] if n_raw else ("", 0.0)
            rescue_notes = f"model blind; student sees {s_top1[0]}, noise sees {n_top1[0]}; crops or specialist needed"

        rescue_rows.append({
            "img": img,
            "true_class": true_label,
            "ensemble_pred": ensemble_pred,
            "true_in_student_raw_top3": str(true_label in s_labels),
            "true_in_noise_raw_top3": str(true_label in n_labels),
            "true_in_union_top3": str(true_label in union),
            "error_group": error_group,
            "recommended_fix_type": fix_type,
            "rescue_notes": rescue_notes,
        })

    # --- Write error decomposition ---
    decomp_path = output_dir / "xn_gamma_error_decomposition.csv"
    with decomp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_DECOMPOSITION_FIELDS)
        writer.writeheader()
        writer.writerows(error_rows)
    print(f"error_decomposition: {decomp_path.resolve()} ({len(error_rows)} errors)")

    # --- Write rescue candidates ---
    rescue_path = output_dir / "xn_gamma_rescue_candidates.csv"
    with rescue_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESCUE_FIELDS)
        writer.writeheader()
        writer.writerows(rescue_rows)
    print(f"rescue_candidates: {rescue_path.resolve()}")

    # --- Build failure summary ---
    # Aggregate by error_group and true_class
    group_counter: Counter[str] = Counter()
    classes_by_group: dict[str, set[str]] = defaultdict(set)
    examples_by_group: dict[str, list[str]] = defaultdict(list)
    fix_by_group: dict[str, str] = {}

    for row in error_rows:
        grp = row["error_group"]
        group_counter[grp] += 1
        classes_by_group[grp].add(row["true_class"])
        examples_by_group[grp].append(row["img"])
        fix_by_group[grp] = row["recommended_fix_type"]

    # Also aggregate by fix_type
    fix_counter: Counter[str] = Counter()
    classes_by_fix: dict[str, set[str]] = defaultdict(set)
    for row in error_rows:
        fix_counter[row["recommended_fix_type"]] += 1
        classes_by_fix[row["recommended_fix_type"]].add(row["true_class"])

    summary_rows: list[dict[str, str]] = []

    # By error group
    for grp in sorted(group_counter):
        est_gain = sum(1 for r in error_rows if r["error_group"] == grp and r["true_in_union_top3"] == "True")
        summary_rows.append({
            "category": f"error_group:{grp}",
            "count": str(group_counter[grp]),
            "classes_affected": ", ".join(sorted(classes_by_group[grp])),
            "example_images": ", ".join(examples_by_group[grp][:3]),
            "recommended_action": fix_by_group.get(grp, "investigate"),
            "estimated_gain": f"up to {est_gain} (true in union top3)",
        })

    # By fix type
    for fix in sorted(fix_counter):
        est_gain = sum(1 for r in error_rows if r["recommended_fix_type"] == fix and r["true_in_union_top3"] == "True")
        summary_rows.append({
            "category": f"fix_type:{fix}",
            "count": str(fix_counter[fix]),
            "classes_affected": ", ".join(sorted(classes_by_fix[fix])),
            "example_images": "",
            "recommended_action": "see error_group breakdown",
            "estimated_gain": f"up to {est_gain}",
        })

    # Topline stats
    total_errors = len(error_rows)
    calibration_errors = sum(1 for r in error_rows if r["error_group"].startswith("calibration"))
    union_errors = sum(1 for r in error_rows if r["true_in_union_top3"] == "True")
    blind_errors = sum(1 for r in error_rows if r["error_group"] == "model_blind")
    expert_errors = sum(1 for r in error_rows if r["error_group"] == "expert_fired")

    summary_rows.append({
        "category": "TOPLINE",
        "count": str(total_errors),
        "classes_affected": "",
        "example_images": "",
        "recommended_action": (
            f"calibration_candidates={calibration_errors}, "
            f"union_top3_rescuable={union_errors}, "
            f"model_blind={blind_errors}, "
            f"expert_fired={expert_errors}"
        ),
        "estimated_gain": f"max calibration gain={union_errors}, min new data required={blind_errors}",
    })

    summary_path = output_dir / "xn_gamma_remaining_failure_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"failure_summary: {summary_path.resolve()}")

    # --- Print topline to stdout ---
    print()
    print(f"=== XN Gamma Error Audit ===")
    print(f"Total errors: {total_errors}")
    print(f"  calibration (true in union top3): {calibration_errors}")
    print(f"    calibration_both_see: {group_counter.get('calibration_both_see', 0)}")
    print(f"    calibration_student_only: {group_counter.get('calibration_student_only', 0)}")
    print(f"    calibration_noise_only: {group_counter.get('calibration_noise_only', 0)}")
    print(f"  model_blind (true not in either top3): {blind_errors}")
    print(f"  expert_fired: {expert_errors}")
    print(f"Max calibration gain: {union_errors}")
    print(f"Remaining after calibration: >= {total_errors - union_errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="audit XN Gamma errors with full per-model top-3 decomposition")
    parser.add_argument("--student_debug", default="output/_xn_gamma_student_debug.csv")
    parser.add_argument("--noise_debug", default="output/_xn_gamma_noise_debug.csv")
    parser.add_argument("--ensemble_debug", default="output/debug_xn_gamma.csv")
    parser.add_argument("--gt", default="input/ground_truth.csv")
    parser.add_argument("--output_dir", default="output/audit_xn_gamma_deep")
    args = parser.parse_args()

    decompose_errors(
        student_debug=Path(args.student_debug),
        noise_debug=Path(args.noise_debug),
        ensemble_debug=Path(args.ensemble_debug),
        gt_csv=Path(args.gt),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
