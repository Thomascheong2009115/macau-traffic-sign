import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.class_filter import is_valid_class, load_classes
from src.generate_augmented_dataset import generate_student_batch_augmented_dataset
from src.import_student_zip import StudentImportSummary, import_student_zip


BASE_FULL_MANIFEST = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv"
RAW_MERGED_MANIFEST = "data/manifests/train_full_plus_student_batch_raw.csv"
AUG_MERGED_MANIFEST = "data/manifests/train_full_plus_student_batch_aug.csv"
LOCKED_FINAL_CHECKPOINT = "checkpoints/full_reinforced_student_aug_best.pt"
EXPERT_CHECKPOINT = "checkpoints/expert_202_203_204.pt"
FINAL_COMPARE_METRICS = ("output/audit_latest/test_metrics.txt", "output/audit_expert_202_203_204/test_metrics.txt", "output/test_metrics.txt")


@dataclass(frozen=True)
class PythonCudaInfo:
    command: list[str]
    executable: str
    torch_version: str
    torchvision_version: str
    gpu_name: str


@dataclass(frozen=True)
class MergeSummary:
    base_rows: int
    student_rows_seen: int
    student_rows_appended: int
    duplicate_paths_skipped: int
    output_csv: Path


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _resolve(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def _relative_posix(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_manifest_row(row: dict[str, str], classes: list[str], project_root: Path, source: str) -> None:
    label = row.get("label", "").strip()
    if not is_valid_class(label, classes):
        raise ValueError(f"invalid label: {label}")
    if row.get("source", "").strip() != source:
        raise ValueError(f"expected source={source}, got {row.get('source', '')}")
    path = _resolve(project_root, row.get("path", "").strip())
    if not path.exists():
        raise FileNotFoundError(f"manifest image path not found: {path}")


def merge_base_with_student_batch(
    base_csv: str | Path = BASE_FULL_MANIFEST,
    student_manifest_csv: str | Path = "",
    output_csv: str | Path = RAW_MERGED_MANIFEST,
    classes_path: str | Path = "classes.txt",
    project_root: str | Path | None = None,
) -> MergeSummary:
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    base_csv = Path(base_csv)
    student_manifest_csv = Path(student_manifest_csv)
    output_csv = Path(output_csv)
    classes = load_classes(classes_path)

    base_fields, base_rows = _read_csv(base_csv)
    if base_fields != ["path", "label", "source"]:
        raise ValueError(f"{base_csv} columns must be path,label,source")
    student_fields, student_rows = _read_csv(student_manifest_csv)
    if student_fields not in (
        ["path", "label", "source"],
        ["path", "label", "source", "batch_id", "original_name", "sha256"],
        ["path", "label", "source", "parent_path", "aug_id"],
    ):
        raise ValueError(f"{student_manifest_csv} has unsupported columns: {student_fields}")

    output_rows: list[list[str]] = []
    seen_paths: set[str] = set()
    for row in base_rows:
        source = row.get("source", "").strip()
        if source not in {"real", "augmented_real", "real_student_images", "augmented_student", "augmented_hardcase"}:
            raise ValueError(f"unsupported base source={source}")
        _validate_manifest_row(row, classes, project_root, source)
        path = row["path"].replace("\\", "/")
        seen_paths.add(path.lower())
        output_rows.append([path, row["label"].strip(), source])

    appended = 0
    duplicates = 0
    for row in student_rows:
        source = row.get("source", "").strip()
        if source not in {"real_student_images", "augmented_student"}:
            raise ValueError(f"student batch source must be real_student_images or augmented_student, got {source}")
        _validate_manifest_row(row, classes, project_root, source)
        path = row["path"].replace("\\", "/")
        if path.lower() in seen_paths:
            duplicates += 1
            continue
        seen_paths.add(path.lower())
        output_rows.append([path, row["label"].strip(), source])
        appended += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "source"])
        writer.writerows(output_rows)
    return MergeSummary(
        base_rows=len(base_rows),
        student_rows_seen=len(student_rows),
        student_rows_appended=appended,
        duplicate_paths_skipped=duplicates,
        output_csv=output_csv,
    )


def build_stage_train_command(
    stage: str,
    train_csv: str,
    checkpoint: str,
    log_path: str,
    metrics_path: str,
    epochs: int,
    lr: float,
    head_lr: float,
    optimizer: str,
    label_smoothing: float,
    image_size: int,
    pad_to_square: int,
    init_checkpoint: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "src/train.py",
        "--train_csv",
        train_csv,
        "--val_csv",
        "splits/val_real.csv",
        "--model",
        "resnet50",
        "--epochs",
        str(epochs),
        "--batch_size",
        "64",
        "--lr",
        f"{lr:.10g}",
        "--head_lr",
        f"{head_lr:.10g}",
        "--optimizer",
        optimizer,
        "--weight_decay",
        "0.0001",
        "--label_smoothing",
        f"{label_smoothing:.10g}",
        "--use_amp",
        "1",
        "--balanced_sampler",
        "1",
        "--class_weight_loss",
        "0",
        "--early_stop_patience",
        "4" if stage != "finetune" else "3",
        "--aug_strength",
        "light",
        "--scheduler",
        "cosine",
        "--image_size",
        str(image_size),
        "--pad_to_square",
        str(pad_to_square),
        "--checkpoint_path",
        checkpoint,
        "--log_path",
        log_path,
        "--metrics_path",
        metrics_path,
        "--output_dir",
        f"output/student_batch_{stage}_train",
    ]
    if init_checkpoint:
        command.extend(["--init_checkpoint", init_checkpoint])
    return command


def build_stage_eval_command(
    stage: str,
    checkpoint: str,
    image_size: int,
    pad_to_square: int,
    expert_checkpoint: str = EXPERT_CHECKPOINT,
    baseline_debug: str = "output/debug_expert_202_203_204.csv",
) -> tuple[list[str], list[str], list[str]]:
    submission_csv = f"output/submission_student_batch_{stage}.csv"
    debug_csv = f"output/debug_student_batch_{stage}.csv"
    audit_dir = f"output/audit_student_batch_{stage}"
    infer_cmd = [
        sys.executable,
        "src/infer.py",
        "--input_csv",
        "input/input.csv",
        "--image_dir",
        "input/images",
        "--checkpoint",
        checkpoint,
        "--output",
        submission_csv,
        "--debug_csv",
        debug_csv,
        "--image_size",
        str(image_size),
        "--pad_to_square",
        str(pad_to_square),
        "--tta",
        "0",
        "--coarse_filter",
        "0",
        "--use_expert_202_203_204",
        "1",
        "--expert_202_203_204_checkpoint",
        expert_checkpoint,
        "--expert_trigger_rank",
        "3",
        "--expert_mode",
        "conservative",
    ]
    eval_cmd = [
        sys.executable,
        "src/evaluate_test_predictions.py",
        "--pred",
        debug_csv,
        "--gt",
        "input/ground_truth.csv",
        "--output_dir",
        audit_dir,
    ]
    impact_cmd = [
        sys.executable,
        "src/evaluate_expert_impact.py",
        "--baseline_debug",
        baseline_debug,
        "--expert_debug",
        debug_csv,
        "--gt",
        "input/ground_truth.csv",
        "--output_csv",
        f"output/student_batch_{stage}_impact.csv",
    ]
    return infer_cmd, eval_cmd, impact_cmd


def _candidate_python_commands() -> list[list[str]]:
    commands: list[list[str]] = [[sys.executable]]
    for name in ("python", "py", "python3"):
        exe = shutil.which(name)
        if not exe:
            continue
        command = [exe, "-3"] if Path(exe).name.lower() == "py.exe" else [exe]
        if command not in commands:
            commands.append(command)
    return commands


def find_cuda_python() -> PythonCudaInfo:
    code = (
        "import json, sys, torch, torchvision; "
        "ok=torch.cuda.is_available(); "
        "print(json.dumps({'executable':sys.executable,'torch':torch.__version__,"
        "'torchvision':torchvision.__version__,'cuda':ok,"
        "'gpu':torch.cuda.get_device_name(0) if ok else ''}))"
    )
    for command in _candidate_python_commands():
        try:
            result = subprocess.run([*command, "-c", code], check=False, capture_output=True, text=True)
        except OSError:
            continue
        if result.returncode != 0:
            continue
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            continue
        if payload.get("cuda"):
            return PythonCudaInfo(
                command=command,
                executable=str(payload["executable"]),
                torch_version=str(payload["torch"]),
                torchvision_version=str(payload["torchvision"]),
                gpu_name=str(payload["gpu"]),
            )
    raise RuntimeError("no Python with import torch, torchvision and torch.cuda.is_available()=True was found")


def _with_python(command: list[str], python_command: list[str]) -> list[str]:
    return [*python_command, *command[1:]]


def run_checked(command: list[str], python_command: list[str] | None = None, dry_run: bool = False) -> None:
    command = _with_python(command, python_command) if python_command is not None else command
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _best_log_row(path: Path, best_epoch: int) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if int(row.get("epoch", "0") or 0) == best_epoch:
            return row
    return rows[-1] if rows else {}


def _wrong_counts(audit_dir: Path) -> tuple[int, int]:
    wrong_csv = audit_dir / "wrong_cases.csv"
    if not wrong_csv.exists():
        return 0, 0
    wrong_107 = 0
    wrong_203 = 0
    with wrong_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            wrong_107 += int(row.get("true_class", "") == "107")
            wrong_203 += int(row.get("true_class", "") == "203")
    return wrong_107, wrong_203


def _correct_for_class(audit_dir: Path, label: str) -> int:
    per_class_csv = audit_dir / "per_class_accuracy.csv"
    if not per_class_csv.exists():
        return 0
    with per_class_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Class") == label:
                return int(float(row.get("correct", 0) or 0))
    return 0


def current_locked_final_correct() -> int:
    for path_text in FINAL_COMPARE_METRICS:
        values = _read_key_values(Path(path_text))
        if "correct" in values:
            return int(float(values["correct"]))
    return 0


def import_summary_from_manifest(path: str | Path) -> StudentImportSummary:
    manifest_csv = Path(path)
    rows: list[dict[str, str]] = []
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    batch_id = rows[0].get("batch_id", manifest_csv.stem) if rows else manifest_csv.stem
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get("label", "").strip()
        if label:
            counts[label] = counts.get(label, 0) + 1
    timestamp = batch_id.replace("student_batch_", "", 1)
    return StudentImportSummary(
        batch_id=batch_id,
        extract_dir=Path("data/imports") / batch_id,
        manifest_csv=manifest_csv,
        report_csv=Path("output") / f"student_import_report_{timestamp}.csv",
        total_images_found=len(rows),
        imported_images=len(rows),
        duplicate_skipped=0,
        invalid_class_skipped=0,
        csv_rows=0,
        images_missing_csv=0,
        classes_with_new_images=counts,
    )


def recommendation_for(external_correct: int, locked_final_correct: int, overfit_flag: str, wrong_107: int, wrong_203: int) -> str:
    if overfit_flag == "YES":
        return "do_not_adopt_overfit"
    if external_correct >= locked_final_correct + 1 and external_correct >= 87 and wrong_107 == 0 and wrong_203 == 0:
        return "strong_candidate_adopt"
    if external_correct >= locked_final_correct + 1:
        return "candidate_review_required"
    if 83 <= external_correct <= 86:
        return "keep_candidate"
    return "do_not_adopt"


def collect_stage_summary(stage: str, checkpoint: str, metrics_path: str, log_path: str, locked_final_correct: int) -> dict[str, object]:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8")) if Path(metrics_path).exists() else {}
    best_epoch = int(metrics.get("best_epoch", 0) or 0)
    log_row = _best_log_row(Path(log_path), best_epoch)
    audit_dir = Path(f"output/audit_student_batch_{stage}")
    external = _read_key_values(audit_dir / "test_metrics.txt")
    external_correct = int(float(external.get("correct", 0) or 0))
    total = int(float(external.get("total", 0) or 0))
    wrong_107, wrong_203 = _wrong_counts(audit_dir)
    overfit_flag = log_row.get("overfit_flag", "")
    train_acc = float(log_row.get("train_top1_acc", 0.0) or 0.0)
    val_acc = float(metrics.get("best_top1_acc", 0.0) or 0.0)
    return {
        "stage": stage,
        "checkpoint": checkpoint,
        "val_acc": val_acc,
        "val_top3": float(metrics.get("best_top3_acc", 0.0) or 0.0),
        "macro_f1_present": float(metrics.get("best_macro_f1_present_classes", 0.0) or 0.0),
        "external_correct": external_correct,
        "external_accuracy": external_correct / total if total else 0.0,
        "203_correct_after": _correct_for_class(audit_dir, "203"),
        "wrong_107": wrong_107,
        "wrong_203": wrong_203,
        "train_val_gap": train_acc - val_acc,
        "overfit_flag": overfit_flag,
        "recommendation": recommendation_for(external_correct, locked_final_correct, overfit_flag, wrong_107, wrong_203),
    }


def write_pipeline_summary(rows: Iterable[dict[str, object]], locked_final_correct: int) -> dict[str, object]:
    rows = list(rows)
    fields = [
        "stage",
        "checkpoint",
        "val_acc",
        "val_top3",
        "macro_f1_present",
        "external_correct",
        "external_accuracy",
        "203_correct_after",
        "wrong_107",
        "wrong_203",
        "train_val_gap",
        "overfit_flag",
        "recommendation",
    ]
    summary_csv = Path("output/student_batch_pipeline_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    best = max(
        rows,
        key=lambda row: (
            int(row.get("external_correct", 0)),
            str(row.get("overfit_flag", "")) != "YES",
            float(row.get("macro_f1_present", 0.0)),
        ),
        default={},
    )
    summary = {
        "locked_final_correct": locked_final_correct,
        "best": best,
        "adoption_rule": "recommend only when external_correct >= locked_final + 1, strong if >= 87, with no overfit regression",
        "summary_csv": summary_csv.as_posix(),
    }
    Path("output/student_batch_best_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _stage_specs() -> list[dict[str, object]]:
    return [
        {
            "stage": "raw",
            "train_csv": RAW_MERGED_MANIFEST,
            "checkpoint": "checkpoints/student_batch_raw_candidate.pt",
            "log_path": "output/student_batch_raw_train_log.csv",
            "metrics_path": "output/student_batch_raw_metrics.json",
            "epochs": 10,
            "lr": 8e-5,
            "head_lr": 2e-4,
            "optimizer": "adam",
            "label_smoothing": 0.05,
            "image_size": 256,
            "pad_to_square": 0,
        },
        {
            "stage": "aug",
            "train_csv": AUG_MERGED_MANIFEST,
            "checkpoint": "checkpoints/student_batch_aug_candidate.pt",
            "log_path": "output/student_batch_aug_train_log.csv",
            "metrics_path": "output/student_batch_aug_metrics.json",
            "epochs": 12,
            "lr": 8e-5,
            "head_lr": 2e-4,
            "optimizer": "adam",
            "label_smoothing": 0.05,
            "image_size": 256,
            "pad_to_square": 0,
        },
        {
            "stage": "finetune",
            "train_csv": AUG_MERGED_MANIFEST,
            "checkpoint": "checkpoints/student_batch_finetune_candidate.pt",
            "log_path": "output/student_batch_finetune_train_log.csv",
            "metrics_path": "output/student_batch_finetune_metrics.json",
            "epochs": 8,
            "lr": 3e-5,
            "head_lr": 1e-4,
            "optimizer": "adamw",
            "label_smoothing": 0.03,
            "image_size": 256,
            "pad_to_square": 0,
            "init_checkpoint": "checkpoints/student_batch_aug_candidate.pt",
        },
    ]


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    cuda = find_cuda_python()
    print(f"python={cuda.executable}")
    print(f"torch={cuda.torch_version}")
    print(f"torchvision={cuda.torchvision_version}")
    print(f"gpu={cuda.gpu_name}")
    if args.student_manifest:
        imported = import_summary_from_manifest(args.student_manifest)
        print(f"student_manifest={imported.manifest_csv.as_posix()} rows={imported.imported_images}")
    else:
        imported = import_student_zip(args.zip_path, timestamp=args.timestamp)
    raw_merge = merge_base_with_student_batch(BASE_FULL_MANIFEST, imported.manifest_csv, RAW_MERGED_MANIFEST)
    print(f"raw_manifest={raw_merge.output_csv.as_posix()} appended={raw_merge.student_rows_appended}")
    locked_final_correct = current_locked_final_correct()
    rows: list[dict[str, object]] = []
    raw_spec, aug_spec, finetune_spec = _stage_specs()

    for spec in (raw_spec,):
        train_cmd = build_stage_train_command(**spec)
        run_checked(train_cmd, cuda.command, dry_run=args.dry_run)
        for command in build_stage_eval_command(spec["stage"], spec["checkpoint"], spec["image_size"], spec["pad_to_square"]):
            run_checked(command, cuda.command, dry_run=args.dry_run)
        if not args.dry_run:
            rows.append(collect_stage_summary(spec["stage"], spec["checkpoint"], spec["metrics_path"], spec["log_path"], locked_final_correct))

    aug_timestamp = imported.batch_id.replace("student_batch_", "")
    augmented_manifest = Path(f"data/manifests/augmented_student_batch_{aug_timestamp}.csv")
    generate_student_batch_augmented_dataset(
        input_csv=imported.manifest_csv,
        base_manifest_csv=BASE_FULL_MANIFEST,
        output_dir=f"data/augmented_student_batch_{aug_timestamp}",
        output_csv=augmented_manifest,
        default_num_aug=8,
        image_size=256,
        aug_strength="light",
        overwrite=True,
    )
    merge_base_with_student_batch(BASE_FULL_MANIFEST, imported.manifest_csv, AUG_MERGED_MANIFEST)
    merge_base_with_student_batch(AUG_MERGED_MANIFEST, augmented_manifest, AUG_MERGED_MANIFEST)

    for spec in (aug_spec, finetune_spec):
        train_cmd = build_stage_train_command(**spec)
        run_checked(train_cmd, cuda.command, dry_run=args.dry_run)
        for command in build_stage_eval_command(spec["stage"], spec["checkpoint"], spec["image_size"], spec["pad_to_square"]):
            run_checked(command, cuda.command, dry_run=args.dry_run)
        if not args.dry_run:
            rows.append(collect_stage_summary(spec["stage"], spec["checkpoint"], spec["metrics_path"], spec["log_path"], locked_final_correct))

    optimax_cmd = [
        sys.executable,
        "src/optimize_training.py",
        "--train_csv",
        AUG_MERGED_MANIFEST,
        "--val_csv",
        "splits/val_real.csv",
        "--model",
        "resnet50",
        "--trials",
        "6",
        "--batch_size",
        "64",
        "--expert_checkpoint",
        EXPERT_CHECKPOINT,
        "--baseline_debug",
        "output/debug_expert_202_203_204.csv",
        "--results",
        "output/optimax_student_batch/results.csv",
        "--summary",
        "output/optimax_student_batch/best_summary.json",
        "--checkpoint_dir",
        "checkpoints/optimax_student_batch",
        "--output_dir",
        "output/optimax_student_batch",
        "--max_epochs",
        "10",
        "--early_stop_patience",
        "3",
    ]
    run_checked(optimax_cmd, cuda.command, dry_run=args.dry_run)
    if not args.dry_run:
        optimax_rows = []
        results_path = Path("output/optimax_student_batch/results.csv")
        if results_path.exists():
            with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
                optimax_rows = list(csv.DictReader(handle))
        if optimax_rows:
            best = max(
                optimax_rows,
                key=lambda row: (
                    int(float(row.get("external_correct", 0) or 0)),
                    float(row.get("val_acc", 0.0) or 0.0),
                    float(row.get("macro_f1_present", 0.0) or 0.0),
                ),
            )
            rows.append(
                {
                    "stage": "optimax",
                    "checkpoint": best.get("checkpoint", ""),
                    "val_acc": best.get("val_acc", 0),
                    "val_top3": best.get("val_top3", 0),
                    "macro_f1_present": best.get("macro_f1_present", 0),
                    "external_correct": best.get("external_correct", 0),
                    "external_accuracy": float(best.get("external_correct", 0) or 0) / 100.0,
                    "203_correct_after": "",
                    "wrong_107": "",
                    "wrong_203": "",
                    "train_val_gap": best.get("train_val_gap", 0),
                    "overfit_flag": best.get("overfit_flag", ""),
                    "recommendation": best.get("adoption_decision", ""),
                }
            )
        return write_pipeline_summary(rows, locked_final_correct)
    return {}


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="import a student ZIP and run the raw/aug/finetune/Optimax candidate pipeline")
    parser.add_argument("--zip", dest="zip_path", default="students.zip")
    parser.add_argument("--timestamp")
    parser.add_argument("--student_manifest")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    run_pipeline(create_arg_parser().parse_args())


if __name__ == "__main__":
    main()
