import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tune_model_ensemble import PROTECTED_BLUE, DebugPrediction, read_debug_prediction_scores


CONSENSUS_OVERRIDE_T_STUDENT = 0.40
CONSENSUS_OVERRIDE_T_NOISE = 0.35

DEBUG_FIELDS = [
    "img",
    "student_raw_top1",
    "student_raw_top1_conf",
    "student_raw_top2",
    "student_raw_top2_conf",
    "student_raw_top3",
    "student_raw_top3_conf",
    "student_final",
    "student_final_conf",
    "noise_raw_top1",
    "noise_raw_top1_conf",
    "noise_raw_top2",
    "noise_raw_top2_conf",
    "noise_raw_top3",
    "noise_raw_top3_conf",
    "noise_final",
    "noise_final_conf",
    "ensemble_top1",
    "ensemble_top1_conf",
    "ensemble_top2",
    "ensemble_top2_conf",
    "ensemble_top3",
    "ensemble_top3_conf",
    "rule",
    "rule_description",
]
RULE = "weighted_0.7"
RULE_DESCRIPTION = "weighted average student=0.7 noise=0.3"
CONSENSUS_RULE = "consensus_override"
CONSENSUS_DESCRIPTION = (
    f"consensus_override T_s={CONSENSUS_OVERRIDE_T_STUDENT} T_n={CONSENSUS_OVERRIDE_T_NOISE} "
    "protected_blue_guard"
)


def _read_raw_top3_from_debug_csv(debug_path: Path) -> dict[str, dict[str, object]]:
    """Read raw main_top{1,2,3}_class and _conf from infer.py debug CSV (pre-expert)."""
    raw: dict[str, dict[str, object]] = {}
    with debug_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img = row.get("img", "").strip()
            if not img:
                continue
            raw[img] = {
                "raw_top1": row.get("main_top1_class", "").strip(),
                "raw_top1_conf": row.get("main_top1_conf", "").strip(),
                "raw_top2": row.get("main_top2_class", "").strip(),
                "raw_top2_conf": row.get("main_top2_conf", "").strip(),
                "raw_top3": row.get("main_top3_class", "").strip(),
                "raw_top3_conf": row.get("main_top3_conf", "").strip(),
            }
    return raw


def _run_single_model_inference(
    infer_config: Path,
    input_csv: Path,
    image_dir: Path,
    checkpoint: Path,
    submission: Path,
    debug_csv: Path,
    image_size: int,
    pad_to_square: int,
    expert_checkpoint: Path,
    expert_mode: str,
    tta: int = 0,
    use_red_expert: bool = False,
    red_expert_checkpoint: Path | None = None,
) -> None:
    infer_py = Path(__file__).resolve().parent / "infer.py"
    command = [
        sys.executable,
        str(infer_py),
        "--config",
        str(infer_config),
        "--input_csv",
        str(input_csv),
        "--image_dir",
        str(image_dir),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(submission),
        "--debug_csv",
        str(debug_csv),
        "--image_size",
        str(image_size),
        "--pad_to_square",
        str(pad_to_square),
        "--tta",
        str(tta),
        "--coarse_filter",
        "0",
        "--use_expert_202_203_204",
        "1",
        "--expert_202_203_204_checkpoint",
        str(expert_checkpoint),
        "--expert_trigger_rank",
        "3",
        "--expert_mode",
        expert_mode,
    ]
    if use_red_expert and red_expert_checkpoint is not None:
        command.extend([
            "--use_generic_expert",
            "1",
            "--generic_expert_checkpoint",
            str(red_expert_checkpoint),
            "--generic_expert_classes",
            "105,110,113,118,138",
            "--generic_expert_trigger_rank",
            "3",
            "--generic_expert_mode",
            "broad",
        ])
    subprocess.run(command, check=True)


def _weighted_prediction(
    student_pred: DebugPrediction, noise_pred: DebugPrediction, student_weight: float
) -> str:
    noise_weight = 1.0 - student_weight
    combined: dict[str, float] = {}
    for label, score in student_pred.scores.items():
        combined[label] = combined.get(label, 0.0) + student_weight * score
    for label, score in noise_pred.scores.items():
        combined[label] = combined.get(label, 0.0) + noise_weight * score
    if not combined:
        return student_pred.pred
    return max(combined.items(), key=lambda item: (item[1], item[0]))[0]


def _weighted_top3(
    student_pred: DebugPrediction,
    noise_pred: DebugPrediction,
    student_weight: float,
) -> list[tuple[str, float]]:
    noise_weight = 1.0 - student_weight
    combined: dict[str, float] = {}
    for label, score in student_pred.scores.items():
        combined[label] = combined.get(label, 0.0) + student_weight * score
    for label, score in noise_pred.scores.items():
        combined[label] = combined.get(label, 0.0) + noise_weight * score
    sorted_items = sorted(combined.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return [(label, score) for label, score in sorted_items[:3]]


def _apply_consensus_override(
    weighted_class: str,
    student_raw: dict[str, object],
    noise_raw: dict[str, object],
    student_pred: DebugPrediction,
    noise_pred: DebugPrediction,
    student_weight: float,
) -> str:
    """Non-leaking calibration rule.

    When both models agree on their raw top-1 class, both are uncertain
    (below confidence thresholds), the agreed class is not a protected
    blue class, and the intersection of raw top-3 labels contains an
    alternative — pick the alternative with the highest weighted score.

    This rule uses only model outputs; it never references ground truth
    or image filenames.
    """
    s_top1 = str(student_raw.get("raw_top1", "")).strip()
    n_top1 = str(noise_raw.get("raw_top1", "")).strip()
    if not s_top1 or s_top1 != n_top1:
        return weighted_class
    if s_top1 in PROTECTED_BLUE:
        return weighted_class

    s_conf = float(str(student_raw.get("raw_top1_conf", "0") or "0"))
    n_conf = float(str(noise_raw.get("raw_top1_conf", "0") or "0"))
    if s_conf >= CONSENSUS_OVERRIDE_T_STUDENT or n_conf >= CONSENSUS_OVERRIDE_T_NOISE:
        return weighted_class

    # Build raw top-3 label sets
    s_labels: set[str] = set()
    n_labels: set[str] = set()
    for rank in (1, 2, 3):
        label = str(student_raw.get(f"raw_top{rank}", "")).strip()
        if label:
            s_labels.add(label)
        label = str(noise_raw.get(f"raw_top{rank}", "")).strip()
        if label:
            n_labels.add(label)

    intersection = s_labels & n_labels
    candidates = intersection - {s_top1}
    if not candidates:
        return weighted_class

    # Compute weighted score for each candidate
    noise_weight = 1.0 - student_weight
    best_score = -1.0
    best_label = weighted_class
    for candidate in candidates:
        s_score = student_pred.scores.get(candidate, 0.0)
        n_score = noise_pred.scores.get(candidate, 0.0)
        weighted = student_weight * s_score + noise_weight * n_score
        if weighted > best_score:
            best_score = weighted
            best_label = candidate

    return best_label


def _safe_conf(value: object, default: str = "") -> str:
    if value is None:
        return default
    try:
        return f"{float(str(value)):.10g}"
    except (TypeError, ValueError):
        return str(value)


def write_weighted_gamma_outputs(
    student_debug: Path,
    noise_debug: Path,
    output_submission: Path,
    output_debug: Path,
    student_weight: float,
    consensus_override: bool = True,
) -> None:
    student_predictions = read_debug_prediction_scores(student_debug)
    noise_predictions = read_debug_prediction_scores(noise_debug)
    student_raw = _read_raw_top3_from_debug_csv(student_debug)
    noise_raw = _read_raw_top3_from_debug_csv(noise_debug)
    output_debug.parent.mkdir(parents=True, exist_ok=True)

    effective_rule = RULE
    effective_description = RULE_DESCRIPTION
    if consensus_override:
        effective_rule = f"{RULE}+{CONSENSUS_RULE}"
        effective_description = f"{RULE_DESCRIPTION} | {CONSENSUS_DESCRIPTION}"

    with output_debug.open("w", encoding="utf-8-sig", newline="") as debug_handle:
        debug_writer = csv.DictWriter(debug_handle, fieldnames=DEBUG_FIELDS)
        debug_writer.writeheader()
        for img, student_pred in student_predictions.items():
            if img not in noise_predictions:
                raise ValueError(f"missing noise prediction for image: {img}")
            noise_pred = noise_predictions[img]
            ensemble_top3 = _weighted_top3(student_pred, noise_pred, student_weight)
            ensemble_class = ensemble_top3[0][0] if ensemble_top3 else student_pred.pred
            s_raw = student_raw.get(img, {})
            n_raw = noise_raw.get(img, {})

            if consensus_override:
                ensemble_class = _apply_consensus_override(
                    ensemble_class,
                    s_raw,
                    n_raw,
                    student_pred,
                    noise_pred,
                    student_weight,
                )

            # Build final top3: if consensus_override changed the top1,
            # recompute the conf from the weighted combined scores.
            if ensemble_class != ensemble_top3[0][0]:
                # Find the confidence of the overridden class in weighted scores
                override_conf = 0.0
                noise_weight = 1.0 - student_weight
                for label, score in student_pred.scores.items():
                    if label == ensemble_class:
                        override_conf += student_weight * score
                for label, score in noise_pred.scores.items():
                    if label == ensemble_class:
                        override_conf += noise_weight * score
                final_top1 = ensemble_class
                final_top1_conf = override_conf
            else:
                final_top1 = ensemble_top3[0][0] if len(ensemble_top3) > 0 else ""
                final_top1_conf = ensemble_top3[0][1] if len(ensemble_top3) > 0 else 0.0

            debug_writer.writerow(
                {
                    "img": img,
                    "student_raw_top1": s_raw.get("raw_top1", ""),
                    "student_raw_top1_conf": _safe_conf(s_raw.get("raw_top1_conf")),
                    "student_raw_top2": s_raw.get("raw_top2", ""),
                    "student_raw_top2_conf": _safe_conf(s_raw.get("raw_top2_conf")),
                    "student_raw_top3": s_raw.get("raw_top3", ""),
                    "student_raw_top3_conf": _safe_conf(s_raw.get("raw_top3_conf")),
                    "student_final": student_pred.pred,
                    "student_final_conf": f"{student_pred.confidence:.10g}",
                    "noise_raw_top1": n_raw.get("raw_top1", ""),
                    "noise_raw_top1_conf": _safe_conf(n_raw.get("raw_top1_conf")),
                    "noise_raw_top2": n_raw.get("raw_top2", ""),
                    "noise_raw_top2_conf": _safe_conf(n_raw.get("raw_top2_conf")),
                    "noise_raw_top3": n_raw.get("raw_top3", ""),
                    "noise_raw_top3_conf": _safe_conf(n_raw.get("raw_top3_conf")),
                    "noise_final": noise_pred.pred,
                    "noise_final_conf": f"{noise_pred.confidence:.10g}",
                    "ensemble_top1": final_top1,
                    "ensemble_top1_conf": _safe_conf(final_top1_conf),
                    "ensemble_top2": ensemble_top3[1][0] if len(ensemble_top3) > 1 else "",
                    "ensemble_top2_conf": _safe_conf(ensemble_top3[1][1] if len(ensemble_top3) > 1 else 0),
                    "ensemble_top3": ensemble_top3[2][0] if len(ensemble_top3) > 2 else "",
                    "ensemble_top3_conf": _safe_conf(ensemble_top3[2][1] if len(ensemble_top3) > 2 else 0),
                    "rule": effective_rule,
                    "rule_description": effective_description,
                }
            )
    write_submission_from_debug(output_debug, output_submission)


def write_submission_from_debug(debug_csv: Path, output_submission: Path) -> None:
    output_submission.parent.mkdir(parents=True, exist_ok=True)
    with debug_csv.open("r", encoding="utf-8-sig", newline="") as debug_handle, output_submission.open(
        "w", encoding="utf-8-sig", newline=""
    ) as sub_handle:
        reader = csv.DictReader(debug_handle)
        if "img" not in (reader.fieldnames or []) or "ensemble_top1" not in (reader.fieldnames or []):
            raise ValueError("debug_xn_gamma.csv must contain img and ensemble_top1")
        writer = csv.writer(sub_handle)
        writer.writerow(["img", "class"])
        for row in reader:
            writer.writerow([row["img"], row["ensemble_top1"]])


def run_xn_gamma(args: argparse.Namespace) -> None:
    output_submission = Path(args.output)
    output_debug = Path(args.debug_csv)
    temp_student_submission = Path(args.temp_student_submission)
    temp_student_debug = Path(args.temp_student_debug)
    temp_noise_submission = Path(args.temp_noise_submission)
    temp_noise_debug = Path(args.temp_noise_debug)

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
    write_weighted_gamma_outputs(
        temp_student_debug,
        temp_noise_debug,
        output_submission,
        output_debug,
        args.student_weight,
        consensus_override=bool(args.consensus_override),
    )
    print(f"submission.csv: {output_submission.resolve()}")
    print(f"debug_xn_gamma.csv: {output_debug.resolve()}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run XN Gamma student Optimax + noise-CNN weighted fusion inference")
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
    parser.add_argument("--output", default="output/submission.csv")
    parser.add_argument("--debug_csv", default="output/debug_xn_gamma.csv")
    parser.add_argument("--temp_student_submission", default="output/_xn_gamma_student_submission.csv")
    parser.add_argument("--temp_student_debug", default="output/_xn_gamma_student_debug.csv")
    parser.add_argument("--temp_noise_submission", default="output/_xn_gamma_noise_submission.csv")
    parser.add_argument("--temp_noise_debug", default="output/_xn_gamma_noise_debug.csv")
    parser.add_argument("--consensus_override", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use_red_expert", type=int, choices=(0, 1), default=0)
    parser.add_argument("--red_expert_checkpoint", default="checkpoints/expert_red_candidate.pt")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    run_xn_gamma(args)


if __name__ == "__main__":
    main()
