"""
XN Gamma + Wavelet Gate inference pipeline.
Extends the existing XN Gamma fusion with an optional wavelet confidence gate
for low-confidence ensemble predictions. No training required.
"""
import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infer_xn_gamma import (
    _run_single_model_inference,
    write_weighted_gamma_outputs,
    write_submission_from_debug,
)
from src.tune_model_ensemble import PROTECTED_BLUE


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_wavelet_top(wav_debug: Path) -> dict[str, tuple[str, float, str, str]]:
    """Read wavelet debug CSV. Returns img->(top1, conf, top2, top3)."""
    rows = {}
    with wav_debug.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("img", "").strip()
            if not img:
                continue
            top1 = row.get("main_top1_class", "").strip()
            conf = _safe_float(row.get("main_top1_conf", 0))
            top2 = row.get("main_top2_class", "").strip()
            top3 = row.get("main_top3_class", "").strip()
            rows[img] = (top1, conf, top2, top3)
    return rows


def apply_wavelet_gate(
    xn_gamma_debug: Path,
    wav_debug: Path,
    output_debug: Path,
    output_submission: Path,
    ensemble_low_threshold: float = 0.15,
    wavelet_high_threshold: float = 0.10,
    consensus_require: int = 0,
) -> dict:
    """
    Apply wavelet confidence gate to XN Gamma output.

    When ensemble confidence is very low (< ensemble_low_threshold) and
    wavelet has a different prediction with confidence > wavelet_high_threshold,
    override with wavelet prediction.

    Protected blue classes are never overridden.
    """
    wav_data = _read_wavelet_top(wav_debug)

    output_debug.parent.mkdir(parents=True, exist_ok=True)
    triggered = 0
    overridden = 0

    with xn_gamma_debug.open("r", encoding="utf-8-sig", newline="") as in_f, \
         output_debug.open("w", encoding="utf-8-sig", newline="") as out_f:
        reader = csv.DictReader(in_f)
        in_fields = reader.fieldnames or []
        out_fields = list(in_fields) + [
            "wavelet_top1", "wavelet_top1_conf",
            "wavelet_gate_triggered", "wavelet_gate_override",
            "final_class",
        ]
        writer = csv.DictWriter(out_f, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            img = row.get("img", "").strip()
            ensemble_top1 = row.get("ensemble_top1", "").strip()
            ensemble_conf = _safe_float(row.get("ensemble_top1_conf", 0))
            student_raw_top1 = row.get("student_raw_top1", "").strip()
            student_raw_top2 = row.get("student_raw_top2", "").strip()
            student_raw_top3 = row.get("student_raw_top3", "").strip()
            noise_raw_top1 = row.get("noise_raw_top1", "").strip()
            noise_raw_top2 = row.get("noise_raw_top2", "").strip()
            noise_raw_top3 = row.get("noise_raw_top3", "").strip()

            wav_info = wav_data.get(img, ("", 0.0, "", ""))
            wav_top1, wav_conf, wav_top2, wav_top3 = wav_info

            new_row = dict(row)
            new_row["wavelet_top1"] = wav_top1
            new_row["wavelet_top1_conf"] = f"{wav_conf:.10g}" if wav_top1 else ""
            new_row["wavelet_gate_triggered"] = "0"
            new_row["wavelet_gate_override"] = "0"
            new_row["final_class"] = ensemble_top1

            # Wavelet gate logic
            if (ensemble_conf < ensemble_low_threshold
                and wav_top1
                and wav_top1 != ensemble_top1
                and wav_conf > wavelet_high_threshold
                and wav_top1 not in PROTECTED_BLUE):
                triggered += 1
                new_row["wavelet_gate_triggered"] = "1"

                # Consensus check (if required)
                s_top3 = {student_raw_top1, student_raw_top2, student_raw_top3}
                n_top3 = {noise_raw_top1, noise_raw_top2, noise_raw_top3}
                w_top3 = {wav_top1, wav_top2, wav_top3}

                agree_count = sum(1 for ts in [s_top3, n_top3, w_top3] if wav_top1 in ts)

                if consensus_require > 0 and agree_count < consensus_require:
                    pass  # Don't override without consensus
                else:
                    overridden += 1
                    new_row["wavelet_gate_override"] = "1"
                    new_row["final_class"] = wav_top1
                    # Update ensemble_top1 so downstream verifiers see corrected prediction
                    new_row["ensemble_top1"] = wav_top1
                    new_row["ensemble_top1_conf"] = f"{wav_conf:.10g}"

            writer.writerow(new_row)

    # Write submission
    with output_debug.open("r", encoding="utf-8-sig", newline="") as debug_f, \
         output_submission.open("w", encoding="utf-8-sig", newline="") as sub_f:
        reader = csv.DictReader(debug_f)
        writer = csv.writer(sub_f)
        writer.writerow(["img", "class"])
        for row in reader:
            writer.writerow([row["img"], row.get("final_class", row.get("ensemble_top1", ""))])

    return {"triggered": triggered, "overridden": overridden}


def run_full_pipeline(args: argparse.Namespace) -> None:
    output_submission = Path(args.output)
    output_debug = Path(args.debug_csv)

    temp_student_submission = Path(args.temp_student_submission)
    temp_student_debug = Path(args.temp_student_debug)
    temp_noise_submission = Path(args.temp_noise_submission)
    temp_noise_debug = Path(args.temp_noise_debug)
    temp_wavelet_submission = Path(args.temp_wavelet_submission)
    temp_wavelet_debug = Path(args.temp_wavelet_debug)

    # Step 1: Run student model
    print("=== Step 1: Student model inference ===")
    _run_single_model_inference(
        Path(args.infer_config),
        Path(args.input_csv),
        Path(args.image_dir),
        Path(args.student_checkpoint),
        temp_student_submission,
        temp_student_debug,
        args.image_size,
        args.pad_to_square,
        Path(args.expert_checkpoint),
        args.expert_mode,
        tta=args.tta,
        use_red_expert=bool(args.use_red_expert),
        red_expert_checkpoint=Path(args.red_expert_checkpoint) if args.use_red_expert else None,
    )

    # Step 2: Run noise model
    print("=== Step 2: Noise model inference ===")
    _run_single_model_inference(
        Path(args.infer_config),
        Path(args.input_csv),
        Path(args.image_dir),
        Path(args.noise_checkpoint),
        temp_noise_submission,
        temp_noise_debug,
        args.image_size,
        args.pad_to_square,
        Path(args.expert_checkpoint),
        args.expert_mode,
        tta=args.tta,
        use_red_expert=bool(args.use_red_expert),
        red_expert_checkpoint=Path(args.red_expert_checkpoint) if args.use_red_expert else None,
    )

    # Step 3: Weighted fusion (student + noise)
    print("=== Step 3: Student + Noise weighted fusion ===")
    temp_xn_gamma_debug = Path(args.temp_xn_gamma_debug)
    temp_xn_gamma_submission = Path(args.temp_xn_gamma_submission)
    write_weighted_gamma_outputs(
        temp_student_debug,
        temp_noise_debug,
        temp_xn_gamma_submission,
        temp_xn_gamma_debug,
        args.student_weight,
        consensus_override=bool(args.consensus_override),
    )

    # Step 4: Run wavelet model (if wavelet checkpoint provided)
    if args.wavelet_checkpoint:
        print("=== Step 4: Wavelet model inference ===")
        from src.infer_wavelet_model import run_wavelet_inference
        run_wavelet_inference(
            checkpoint_path=Path(args.wavelet_checkpoint),
            input_csv=Path(args.input_csv),
            image_dir=Path(args.image_dir),
            output_submission=temp_wavelet_submission,
            output_debug=temp_wavelet_debug,
        )

        # Step 5: Apply wavelet gate
        print("=== Step 5: Wavelet confidence gate ===")
        stats = apply_wavelet_gate(
            temp_xn_gamma_debug,
            temp_wavelet_debug,
            output_debug,
            output_submission,
            ensemble_low_threshold=args.wavelet_ens_low_threshold,
            wavelet_high_threshold=args.wavelet_high_threshold,
            consensus_require=args.wavelet_consensus_require,
        )
        print(f"wavelet_gate: triggered={stats['triggered']} overridden={stats['overridden']}")
    else:
        # Just copy XN Gamma output
        print("=== Step 4: No wavelet gate (copying XN Gamma output) ===")
        write_submission_from_debug(temp_xn_gamma_debug, output_submission)
        import shutil
        shutil.copy(temp_xn_gamma_debug, output_debug)

    print(f"\nsubmission: {output_submission.resolve()}")
    print(f"debug: {output_debug.resolve()}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XN Gamma + Wavelet Gate inference")
    parser.add_argument("--infer_config", default="config.yaml")
    parser.add_argument("--input_csv", default="input/input.csv")
    parser.add_argument("--image_dir", default="input/images")
    parser.add_argument("--student_checkpoint", default="checkpoints/optimax_student_batch/trial_004_best.pt")
    parser.add_argument("--noise_checkpoint", default="checkpoints/noise_cnn_candidate.pt")
    parser.add_argument("--expert_checkpoint", default="checkpoints/expert_202_203_204.pt")
    parser.add_argument("--expert_mode", choices=("conservative", "broad"), default="conservative")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pad_to_square", type=int, choices=(0, 1), default=1)
    parser.add_argument("--tta", type=int, choices=(0, 1), default=0)
    parser.add_argument("--student_weight", type=float, default=0.7)
    parser.add_argument("--consensus_override", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output", default="output/submission_wavelet_gate.csv")
    parser.add_argument("--debug_csv", default="output/debug_wavelet_gate.csv")
    parser.add_argument("--temp_student_submission", default="output/_wg_student_submission.csv")
    parser.add_argument("--temp_student_debug", default="output/_wg_student_debug.csv")
    parser.add_argument("--temp_noise_submission", default="output/_wg_noise_submission.csv")
    parser.add_argument("--temp_noise_debug", default="output/_wg_noise_debug.csv")
    parser.add_argument("--temp_xn_gamma_submission", default="output/_wg_xn_gamma_submission.csv")
    parser.add_argument("--temp_xn_gamma_debug", default="output/_wg_xn_gamma_debug.csv")
    parser.add_argument("--temp_wavelet_submission", default="output/_wg_wavelet_submission.csv")
    parser.add_argument("--temp_wavelet_debug", default="output/_wg_wavelet_debug.csv")
    parser.add_argument("--use_red_expert", type=int, choices=(0, 1), default=0)
    parser.add_argument("--red_expert_checkpoint", default="checkpoints/expert_red_candidate.pt")
    # Wavelet gate parameters
    parser.add_argument("--wavelet_checkpoint", default="checkpoints/wavelet_ablation/high_best.pt")
    parser.add_argument("--wavelet_ens_low_threshold", type=float, default=0.15)
    parser.add_argument("--wavelet_high_threshold", type=float, default=0.10)
    parser.add_argument("--wavelet_consensus_require", type=int, default=0)
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    run_full_pipeline(args)


if __name__ == "__main__":
    main()
