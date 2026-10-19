from pathlib import Path
import subprocess
import sys
import os
import pandas as pd

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Find project directory: must contain src/ensemble_infer.py and checkpoints
project_candidates = []
for path in INPUT_ROOT.rglob("ensemble_infer.py"):
    if path.parent.name == "src":
        root = path.parent.parent
        if (root / "classes.txt").exists() and (root / "checkpoints").exists():
            project_candidates.append(root)

assert project_candidates, "Cannot find deploy package. Make sure Dataset contains traffic_sign_deploy_cpu/src/ensemble_infer.py."
project_dir = project_candidates[0]
print("PROJECT_DIR =", project_dir)

# 2. Find competition test.csv
csv_candidates = [path for path in INPUT_ROOT.rglob("test.csv") if "traffic_sign_deploy_cpu" not in str(path)]
print("test.csv candidates:", csv_candidates)
assert csv_candidates, "Cannot find competition test.csv. Add detect-traffic-sign competition input."
input_csv = csv_candidates[0]

# 3. Find image directory
image_dir = input_csv.parent / "test"
if not image_dir.exists():
    image_dirs = []
    for directory in input_csv.parent.rglob("*"):
        if directory.is_dir():
            images = list(directory.glob("*.jpg")) + list(directory.glob("*.png")) + list(directory.glob("*.jpeg"))
            if images:
                image_dirs.append(directory)
    assert image_dirs, "Cannot find test image directory."
    image_dir = image_dirs[0]

print("INPUT_CSV =", input_csv)
print("IMAGE_DIR =", image_dir)

# 4. Check model files
required = [
    project_dir / "classes.txt",
    project_dir / "checkpoints" / "final_single_resnet50.pt",
    project_dir / "src" / "ensemble_infer.py",
]

missing = [str(path) for path in required if not path.exists()]
assert not missing, "Missing files:\n" + "\n".join(missing)

# 5. Run single-model compatibility ensemble_infer
command = [
    sys.executable, str(project_dir / "src" / "ensemble_infer.py"),
    "--checkpoint", str(project_dir / "checkpoints" / "final_single_resnet50.pt"),
    "--input_csv", str(input_csv),
    "--image_dir", str(image_dir),
    "--output", str(OUTPUT_DIR / "submission.csv"),
    "--debug_csv", str(OUTPUT_DIR / "debug_predictions.csv"),
    "--missing_csv", str(OUTPUT_DIR / "missing_images.csv"),
    "--classes", str(project_dir / "classes.txt"),
    "--device", "auto",
]

env = os.environ.copy()
env["PYTHONPATH"] = str(project_dir) + os.pathsep + env.get("PYTHONPATH", "")
env["PYTHONDONTWRITEBYTECODE"] = "1"

print("\nRunning command:")
print(" ".join(command))

subprocess.run(command, check=True, cwd=str(project_dir), env=env)

# 6. Check output
submission = OUTPUT_DIR / "submission.csv"
predictions = pd.read_csv(submission)

print("\nDONE")
print("submission:", submission)
print("shape:", predictions.shape)
print(predictions.head())
print("columns:", predictions.columns.tolist())

assert submission.exists(), "submission.csv was not generated"
