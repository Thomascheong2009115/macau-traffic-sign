import csv
import json
import os
import re
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = "data/manifests/train_real_plus_aug_plus_student_plus_student_aug.csv"
VAL_CSV = "splits/val_real.csv"
TEST_CSV = "input/input.csv"
TEST_IMAGE_DIR = "input/images"
GT_CSV = "input/ground_truth.csv"
EXPERT_CKPT = "checkpoints/expert_202_203_204.pt"
BASELINE_MAIN = "checkpoints/full_reinforced_student_aug_best.pt"
BASELINE_203_CORRECT = 9
BASELINE_WRONG_107 = 3
BASELINE_WRONG_203 = 1

RESULT_FIELDS = [
    "experiment_id",
    "val_acc",
    "val_top3",
    "macro_f1_present",
    "external_correct",
    "external_accuracy",
    "203_correct_after",
    "wrong_107_count",
    "wrong_203_count",
    "train_val_gap",
    "overfit_flag",
    "checkpoint",
]

EXPERIMENTS = [
    {
        "id": "E1",
        "name": "baseline_recipe_reproduce",
        "image_size": 224,
        "pad_to_square": 0,
        "optimizer": "adamw",
        "lr": "1e-4",
        "head_lr": "3e-4",
        "weight_decay": "1e-4",
        "label_smoothing": "0.05",
        "balanced_sampler": 1,
        "class_weight_loss": 0,
        "aug_strength": "light",
        "freeze_backbone_epochs": 0,
        "unfreeze_layer4_only": 0,
    },
    {
        "id": "E2",
        "name": "pad_to_square_256",
        "image_size": 256,
        "pad_to_square": 1,
        "optimizer": "adamw",
        "lr": "8e-5",
        "head_lr": "3e-4",
        "weight_decay": "1e-4",
        "label_smoothing": "0.05",
        "balanced_sampler": 1,
        "class_weight_loss": 0,
        "aug_strength": "light",
        "freeze_backbone_epochs": 0,
        "unfreeze_layer4_only": 0,
    },
    {
        "id": "E3",
        "name": "pad_to_square_288",
        "image_size": 288,
        "pad_to_square": 1,
        "optimizer": "adamw",
        "lr": "8e-5",
        "head_lr": "3e-4",
        "weight_decay": "1e-4",
        "label_smoothing": "0.05",
        "balanced_sampler": 1,
        "class_weight_loss": 0,
        "aug_strength": "light",
        "freeze_backbone_epochs": 0,
        "unfreeze_layer4_only": 0,
    },
    {
        "id": "E4",
        "name": "weighted_ce",
        "image_size": 224,
        "pad_to_square": 0,
        "optimizer": "adamw",
        "lr": "8e-5",
        "head_lr": "3e-4",
        "weight_decay": "1e-4",
        "label_smoothing": "0.03",
        "balanced_sampler": 0,
        "class_weight_loss": 1,
        "aug_strength": "light",
        "freeze_backbone_epochs": 0,
        "unfreeze_layer4_only": 0,
    },
    {
        "id": "E5",
        "name": "freeze_backbone_2",
        "image_size": 224,
        "pad_to_square": 0,
        "optimizer": "adamw",
        "lr": "8e-5",
        "head_lr": "3e-4",
        "weight_decay": "1e-4",
        "label_smoothing": "0.05",
        "balanced_sampler": 1,
        "class_weight_loss": 0,
        "aug_strength": "light",
        "freeze_backbone_epochs": 2,
        "unfreeze_layer4_only": 0,
    },
    {
        "id": "E6",
        "name": "safe_medium_aug",
        "image_size": 224,
        "pad_to_square": 0,
        "optimizer": "adamw",
        "lr": "8e-5",
        "head_lr": "3e-4",
        "weight_decay": "1e-4",
        "label_smoothing": "0.03",
        "balanced_sampler": 1,
        "class_weight_loss": 0,
        "aug_strength": "medium",
        "freeze_backbone_epochs": 0,
        "unfreeze_layer4_only": 0,
    },
]


def candidate_python_paths() -> list[str]:
    paths: list[str] = []

    def add(path: str | None) -> None:
        if path and path not in paths and Path(path).exists():
            paths.append(path)

    add(sys.executable)
    add(str(PROJECT_ROOT / ".venv3060" / "Scripts" / "python.exe"))
    add(str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"))
    found = shutil.which("python")
    add(found)
    return paths


def probe_python(path: str) -> dict | None:
    code = (
        "import json, sys, torch, torchvision; "
        "print(json.dumps({"
        "'executable': sys.executable, "
        "'torch': torch.__version__, "
        "'torchvision': torchvision.__version__, "
        "'cuda': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''"
        "}))"
    )
    try:
        proc = subprocess.run([path, "-c", code], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=30)
    except Exception as exc:
        print(f"[PYTHON PROBE FAILED] {path}: {exc}", flush=True)
        return None
    if proc.returncode != 0:
        print(f"[PYTHON REJECTED] {path}: {proc.stderr.strip()}", flush=True)
        return None
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        print(f"[PYTHON BAD PROBE OUTPUT] {path}: {proc.stdout.strip()}", flush=True)
        return None
    print(f"[PYTHON PROBE] {data}", flush=True)
    return data if data.get("cuda") is True else None


def choose_python() -> tuple[str, dict]:
    for path in candidate_python_paths():
        data = probe_python(path)
        if data is not None:
            return path, data
    raise SystemExit("No usable Python with torch, torchvision, and CUDA found.")


def check_required_files() -> None:
    required = [
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
        TEST_IMAGE_DIR,
        GT_CSV,
        BASELINE_MAIN,
        EXPERT_CKPT,
        "src/train.py",
        "src/infer.py",
        "src/evaluate_test_predictions.py",
    ]
    missing = [path for path in required if not (PROJECT_ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))


def stream_command(cmd: list[str], log_path: Path, monitor_training: bool = False) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_parts: list[str] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    stopped_by_monitor = False
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with log_path.open("w", encoding="utf-8") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            output_parts.append(line)
            if monitor_training:
                match = re.search(r"val_loss=([0-9.]+)\s+\|\s+val_acc=([0-9.]+)", line)
                if match:
                    val_losses.append(float(match.group(1)))
                    val_accs.append(float(match.group(2)))
                    if len(val_losses) >= 5:
                        recent_losses = val_losses[-5:]
                        loss_rising_4 = all(recent_losses[i] < recent_losses[i + 1] for i in range(4))
                        best_before = max(val_accs[:-4]) if len(val_accs) > 4 else 0.0
                        no_val_acc_improve = max(val_accs[-4:]) <= best_before
                        if loss_rising_4 and no_val_acc_improve:
                            print("[MONITOR] val_loss rose for 4 epochs and val_acc did not improve; stopping current experiment.", flush=True)
                            log.write("[MONITOR] val_loss rose for 4 epochs and val_acc did not improve; stopping current experiment.\n")
                            stopped_by_monitor = True
                            proc.terminate()
                            break
        if stopped_by_monitor:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return proc.wait(), "".join(output_parts)


def run_simple(cmd: list[str], log_path: Path) -> None:
    code, output = stream_command(cmd, log_path, monitor_training=False)
    if code != 0:
        raise RuntimeError(f"command failed ({code}): {' '.join(cmd)}\n{output[-2000:]}")


def train_experiment(py: str, exp: dict, batch_size: int) -> tuple[int, str]:
    exp_dir = PROJECT_ROOT / "output" / "lambda_matrix" / exp["id"]
    checkpoint = PROJECT_ROOT / "checkpoints" / "lambda_matrix" / f"{exp['id']}_best.pt"
    cmd = [
        py,
        "src/train.py",
        "--train_csv",
        TRAIN_CSV,
        "--val_csv",
        VAL_CSV,
        "--model",
        "resnet50",
        "--epochs",
        "15",
        "--batch_size",
        str(batch_size),
        "--lr",
        exp["lr"],
        "--head_lr",
        exp["head_lr"],
        "--optimizer",
        exp["optimizer"],
        "--weight_decay",
        exp["weight_decay"],
        "--label_smoothing",
        exp["label_smoothing"],
        "--use_amp",
        "1",
        "--balanced_sampler",
        str(exp["balanced_sampler"]),
        "--class_weight_loss",
        str(exp["class_weight_loss"]),
        "--early_stop_patience",
        "4",
        "--aug_strength",
        exp["aug_strength"],
        "--scheduler",
        "cosine",
        "--freeze_backbone_epochs",
        str(exp["freeze_backbone_epochs"]),
        "--unfreeze_layer4_only",
        str(exp["unfreeze_layer4_only"]),
        "--image_size",
        str(exp["image_size"]),
        "--pad_to_square",
        str(exp["pad_to_square"]),
        "--checkpoint_path",
        str(checkpoint.relative_to(PROJECT_ROOT)),
        "--log_path",
        str((exp_dir / "train_log.csv").relative_to(PROJECT_ROOT)),
        "--metrics_path",
        str((exp_dir / "metrics.json").relative_to(PROJECT_ROOT)),
    ]
    return stream_command(cmd, exp_dir / "train_stdout.log", monitor_training=True)


def is_oom(output: str) -> bool:
    lowered = output.lower()
    return "out of memory" in lowered or "cuda oom" in lowered


def parse_key_value_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def read_train_row(exp_dir: Path, best_epoch: int | None) -> dict[str, str]:
    log_path = exp_dir / "train_log.csv"
    if not log_path.exists():
        return {}
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    if best_epoch is not None:
        for row in rows:
            if int(row["epoch"]) == best_epoch:
                return row
    return max(rows, key=lambda row: float(row.get("score", "0") or 0))


def final_prediction(row: dict[str, str]) -> str:
    return row.get("final_class") or row.get("top1_class") or row.get("main_top1_class") or row.get("class") or ""


def compute_guard_metrics(debug_csv: Path, gt_csv: Path) -> dict[str, int]:
    with gt_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        truth_reader = csv.DictReader(handle)
        truth = {}
        for row in truth_reader:
            img = row.get("img") or row.get("IMG") or row.get("image")
            label = row.get("class") or row.get("Class") or row.get("CLASS") or row.get("label") or row.get("true_class")
            if img:
                truth[img] = label
    metrics = {"203_correct_after": 0, "wrong_107_count": 0, "wrong_203_count": 0}
    with debug_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img = row.get("img") or row.get("IMG") or row.get("image")
            true = truth.get(img or "", "")
            pred = final_prediction(row)
            if true == "203" and pred == "203":
                metrics["203_correct_after"] += 1
            if pred == "107" and true != "107":
                metrics["wrong_107_count"] += 1
            if pred == "203" and true != "203":
                metrics["wrong_203_count"] += 1
    return metrics


def evaluate_experiment(py: str, exp: dict) -> dict[str, str]:
    exp_dir = PROJECT_ROOT / "output" / "lambda_matrix" / exp["id"]
    checkpoint = PROJECT_ROOT / "checkpoints" / "lambda_matrix" / f"{exp['id']}_best.pt"
    submission = exp_dir / "submission.csv"
    debug = exp_dir / "debug.csv"
    audit_dir = exp_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    run_simple(
        [
            py,
            "src/infer.py",
            "--input_csv",
            TEST_CSV,
            "--image_dir",
            TEST_IMAGE_DIR,
            "--checkpoint",
            str(checkpoint.relative_to(PROJECT_ROOT)),
            "--output",
            str(submission.relative_to(PROJECT_ROOT)),
            "--debug_csv",
            str(debug.relative_to(PROJECT_ROOT)),
            "--image_size",
            str(exp["image_size"]),
            "--pad_to_square",
            str(exp["pad_to_square"]),
            "--tta",
            "0",
            "--coarse_filter",
            "0",
            "--use_expert_202_203_204",
            "1",
            "--expert_202_203_204_checkpoint",
            EXPERT_CKPT,
            "--expert_trigger_rank",
            "3",
            "--expert_mode",
            "conservative",
        ],
        exp_dir / "infer_stdout.log",
    )
    run_simple(
        [
            py,
            "src/evaluate_test_predictions.py",
            "--pred",
            str(debug.relative_to(PROJECT_ROOT)),
            "--gt",
            GT_CSV,
            "--output_dir",
            str(audit_dir.relative_to(PROJECT_ROOT)),
        ],
        exp_dir / "evaluate_stdout.log",
    )
    run_simple(
        [
            py,
            "src/audit_validation_split.py",
            "--train_csv",
            TRAIN_CSV,
            "--val_csv",
            VAL_CSV,
            "--checkpoint",
            str(checkpoint.relative_to(PROJECT_ROOT)),
            "--image_size",
            str(exp["image_size"]),
            "--pad_to_square",
            str(exp["pad_to_square"]),
            "--output_dir",
            str((exp_dir / "validation_audit").relative_to(PROJECT_ROOT)),
        ],
        exp_dir / "validation_audit_stdout.log",
    )
    metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    best_epoch = int(metrics.get("best_epoch", 0) or 0)
    train_row = read_train_row(exp_dir, best_epoch)
    test_metrics = parse_key_value_file(audit_dir / "test_metrics.txt")
    guards = compute_guard_metrics(debug, PROJECT_ROOT / GT_CSV)
    return {
        "experiment_id": exp["id"],
        "val_acc": str(metrics.get("best_top1_acc", "")),
        "val_top3": str(metrics.get("best_top3_acc", "")),
        "macro_f1_present": str(metrics.get("best_macro_f1_present_classes", "")),
        "external_correct": test_metrics.get("correct", ""),
        "external_accuracy": test_metrics.get("test_accuracy", ""),
        "203_correct_after": str(guards["203_correct_after"]),
        "wrong_107_count": str(guards["wrong_107_count"]),
        "wrong_203_count": str(guards["wrong_203_count"]),
        "train_val_gap": train_row.get("gap_acc", ""),
        "overfit_flag": train_row.get("overfit_flag", ""),
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
    }


def write_results(rows: list[dict[str, str]]) -> None:
    output_path = PROJECT_ROOT / "output" / "lambda_matrix" / "results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_results() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "output" / "lambda_matrix" / "results.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate-only Lambda/CUDA training matrix.")
    parser.add_argument("--only", default="", help="Comma-separated experiment ids to run, for example E1,E2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    check_required_files()
    py, py_info = choose_python()
    print(f"[USING PYTHON] {py}", flush=True)
    print(f"[CUDA] {py_info}", flush=True)
    (PROJECT_ROOT / "checkpoints" / "lambda_matrix").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "lambda_matrix").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "lambda_matrix" / "python_info.json").write_text(
        json.dumps({"python": py, **py_info}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    selected_ids = {value.strip().upper() for value in args.only.split(",") if value.strip()}
    selected = [exp for exp in EXPERIMENTS if not selected_ids or exp["id"] in selected_ids]
    if selected_ids and len(selected) != len(selected_ids):
        known = ", ".join(exp["id"] for exp in EXPERIMENTS)
        raise SystemExit(f"unknown experiment in --only={args.only}; known experiments: {known}")

    rows: list[dict[str, str]] = [row for row in read_existing_results() if row.get("experiment_id") not in {exp["id"] for exp in selected}]
    for exp in selected:
        print(f"\n========== {exp['id']} {exp['name']} ==========", flush=True)
        code, output = train_experiment(py, exp, batch_size=64)
        if code != 0 and is_oom(output):
            print(f"[OOM] {exp['id']} failed with batch_size=64; retrying batch_size=32.", flush=True)
            code, output = train_experiment(py, exp, batch_size=32)
        if code != 0:
            raise RuntimeError(f"{exp['id']} training failed with exit code {code}\n{output[-4000:]}")
        checkpoint = PROJECT_ROOT / "checkpoints" / "lambda_matrix" / f"{exp['id']}_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found after training: {checkpoint}")
        row = evaluate_experiment(py, exp)
        rows.append(row)
        write_results(rows)
        print(f"[RESULT] {row}", flush=True)

    print("[DONE] output/lambda_matrix/results.csv", flush=True)


if __name__ == "__main__":
    main()
