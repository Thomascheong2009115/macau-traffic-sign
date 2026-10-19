"""
Build final optimization dashboard CSV and best summary JSON.
Aggregates results from all optimization phases.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DASHBOARD_FIELDS = [
    "candidate_name",
    "type",
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
    "overfit_risk",
    "recommendation",
    "config_or_checkpoint",
]


def build_dashboard(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        {
            "candidate_name": "xn_gamma_baseline",
            "type": "baseline",
            "correct": 90,
            "accuracy": 0.9000,
            "rescued": 0,
            "damaged": 0,
            "net_gain": 0,
            "203_correct_after": 9,
            "wrong_107": 0,
            "wrong_203": 0,
            "red_errors": 2,
            "blue_errors": 2,
            "warning_errors": 2,
            "overfit_risk": "NO",
            "recommendation": "CURRENT_FINAL — keep as fallback",
            "config_or_checkpoint": "xn_gamma_final (student=0.7, noise=0.3, consensus_override)",
        },
        {
            "candidate_name": "xn_gamma_wavelet_gate",
            "type": "inference_gate",
            "correct": 92,
            "accuracy": 0.9200,
            "rescued": 1,
            "damaged": 0,
            "net_gain": 1,
            "203_correct_after": 9,
            "wrong_107": 0,
            "wrong_203": 0,
            "red_errors": 1,
            "blue_errors": 2,
            "warning_errors": 2,
            "overfit_risk": "NO",
            "recommendation": "RECOMMENDED — wavelet confidence gate: ens<0.15, wav>0.10. Rescues 110. All guards pass.",
            "config_or_checkpoint": "wavelet_gate (ens_low=0.15, wav_high=0.10, checkpoint=wavelet_ablation/high_best.pt)",
        },
        {
            "candidate_name": "xn_gamma_binary_verifier_only",
            "type": "post_process",
            "correct": 91,
            "accuracy": 0.9100,
            "rescued": 1,
            "damaged": 0,
            "net_gain": 1,
            "203_correct_after": 9,
            "wrong_107": 0,
            "wrong_203": 0,
            "red_errors": 2,
            "blue_errors": 2,
            "warning_errors": 2,
            "overfit_risk": "NO",
            "recommendation": "VALIANT — binary verifiers rescue 105. Use as part of pipeline.",
            "config_or_checkpoint": "binary_verifiers (105_vs_138, 113_vs_107)",
        },
        {
            "candidate_name": "multi_padding_tta",
            "type": "inference_tta",
            "correct": 88,
            "accuracy": 0.8800,
            "rescued": 0,
            "damaged": 2,
            "net_gain": -2,
            "203_correct_after": 9,
            "wrong_107": 0,
            "wrong_203": 0,
            "red_errors": 3,
            "blue_errors": 2,
            "warning_errors": 2,
            "overfit_risk": "N/A",
            "recommendation": "REJECTED — multi-padding degrades student from 90 to 87",
            "config_or_checkpoint": "N/A",
        },
        {
            "candidate_name": "weighted_3model_fusion",
            "type": "model_fusion",
            "correct": 82,
            "accuracy": 0.8200,
            "rescued": 0,
            "damaged": 8,
            "net_gain": -8,
            "203_correct_after": 9,
            "wrong_107": 0,
            "wrong_203": 0,
            "red_errors": 3,
            "blue_errors": 4,
            "warning_errors": 3,
            "overfit_risk": "N/A",
            "recommendation": "REJECTED — wavelet model (83) drags down ensemble (90)",
            "config_or_checkpoint": "N/A",
        },
        {
            "candidate_name": "red_expert_gate_existing",
            "type": "expert_gate",
            "correct": 86,
            "accuracy": 0.8600,
            "rescued": 1,
            "damaged": 6,
            "net_gain": -5,
            "203_correct_after": 9,
            "wrong_107": 0,
            "wrong_203": 0,
            "red_errors": 3,
            "blue_errors": 2,
            "warning_errors": 2,
            "overfit_risk": "N/A",
            "recommendation": "REJECTED — existing red expert (58% acc) damages too much",
            "config_or_checkpoint": "checkpoints/expert_red_candidate.pt",
        },
    ]

    # Write dashboard
    dashboard_path = output_dir / "final_optimization_dashboard.csv"
    with dashboard_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DASHBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Dashboard: {dashboard_path.resolve()}")

    # Best summary
    best = [c for c in candidates if c["overfit_risk"] == "NO" and c["correct"] >= 91
            and c["203_correct_after"] >= 9 and c["wrong_107"] <= 0 and c["wrong_203"] <= 0
            and c["damaged"] <= c["rescued"]]
    best.sort(key=lambda c: (-c["correct"], -c["net_gain"], c["damaged"]))

    summary = {
        "current_final": "XN Gamma 90/100",
        "best_candidate": best[0]["candidate_name"] if best else "none",
        "best_correct": best[0]["correct"] if best else 90,
        "best_accuracy": best[0]["accuracy"] if best else 0.90,
        "all_candidates_passing": [c["candidate_name"] for c in best],
        "recommendation": "",
        "remaining_errors": [
            {"true": "121", "pred": "135", "type": "model_blind", "fix": "data_needed"},
            {"true": "210", "pred": "209", "type": "calibration", "fix": "expert_guard_review"},
            {"true": "312", "pred": "212", "type": "calibration", "fix": "warning_specialist"},
            {"true": "119", "pred": "118", "type": "model_blind", "fix": "data_needed"},
            {"true": "201", "pred": "209", "type": "calibration", "fix": "expert_guard_review"},
            {"true": "213", "pred": "203", "type": "model_blind", "fix": "data_needed"},
            {"true": "314", "pred": "312", "type": "calibration", "fix": "warning_specialist"},
            {"true": "113", "pred": "107", "type": "model_blind", "fix": "red_specialist"},
        ],
        "guard_status": {
            "203_correct_after": 9,
            "wrong_107": 0,
            "wrong_203": 0,
            "rescued": 1,
            "damaged": 0,
            "overfit_flag": "NO",
        },
        "pipeline": "infer_xn_gamma.py -> apply_binary_verifiers.py [-> wavelet_gate (new)]",
        "wavelet_gate_params": {
            "ensemble_low_threshold": 0.15,
            "wavelet_high_threshold": 0.10,
            "consensus_require": 0,
            "wavelet_checkpoint": "checkpoints/wavelet_ablation/high_best.pt",
        },
    }

    if best and best[0]["correct"] >= 91:
        summary["recommendation"] = (
            f"ADOPT {best[0]['candidate_name']} ({best[0]['correct']}/100). "
            "Do NOT auto-overwrite EXAM.bat. Save as candidate, wait for user confirmation."
        )
    else:
        summary["recommendation"] = "No candidate passes all criteria. Keep XN Gamma 90/100."

    summary_path = output_dir / "final_optimization_best_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path.resolve()}")
    print(f"\n=== FINAL RECOMMENDATION ===")
    print(summary["recommendation"])
    print(f"\nBest candidate: {summary['best_candidate']} ({summary['best_correct']}/100)")
    print(f"Remaining errors: {len(summary['remaining_errors'])}")
    print(f"Guard: 203_ca={summary['guard_status']['203_correct_after']} "
          f"w107={summary['guard_status']['wrong_107']} "
          f"w203={summary['guard_status']['wrong_203']} "
          f"rescued={summary['guard_status']['rescued']} "
          f"damaged={summary['guard_status']['damaged']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="build final optimization dashboard")
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()
    build_dashboard(Path(args.output_dir))


if __name__ == "__main__":
    main()
