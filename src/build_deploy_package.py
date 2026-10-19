import argparse
import shutil
from pathlib import Path


REQUIREMENTS_CPU = """torch
torchvision
pillow
numpy
pandas
opencv-python
scikit-learn
pyyaml
tqdm
"""

RUNTIME_SRC_FILES = (
    "__init__.py",
    "augment.py",
    "check_deploy_env.py",
    "check_submission.py",
    "class_filter.py",
    "crop.py",
    "data_io.py",
    "infer.py",
    "model.py",
)

CONFIG_CPU = """device: cpu
classes_path: classes.txt
checkpoint_path: checkpoints/best_resnet50.pt
input_csv: input/input.csv
image_dir: input/images
output_csv: output/submission.csv
debug_csv: output/debug_predictions.csv
crop_output_dir: output/crops
debug_crop_dir: output/debug_crops
bad_crop_csv: output/bad_crop.csv
image_size: 224
model_name: resnet50
tta: 0
coarse_filter: 0
batch_size: 1
num_workers: 0
"""

SETUP_CPU_BAT = """@echo off
cd /d %~dp0
python -m venv .venv_cpu
call .venv_cpu\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements_cpu.txt
echo.
echo CPU environment setup completed.
echo Please run check_env.bat to verify.
pause
"""

CHECK_ENV_BAT = """@echo off
cd /d %~dp0
if not exist .venv_cpu\\Scripts\\activate.bat (
  echo .venv_cpu was not found. Run setup_cpu.bat first.
  pause
  exit /b 1
)
call .venv_cpu\\Scripts\\activate
python src\\check_deploy_env.py --config config_cpu.yaml
if errorlevel 1 (
  echo.
  echo ENV CHECK FAILED
  pause
  exit /b 1
)
echo.
echo ENV CHECK PASSED
pause
"""

RUN_EXAM_CPU_BAT = """@echo off
cd /d %~dp0
if not exist .venv_cpu\\Scripts\\activate.bat (
  echo .venv_cpu was not found. Run setup_cpu.bat first.
  pause
  exit /b 1
)
call .venv_cpu\\Scripts\\activate
if exist output\\submission.csv del output\\submission.csv
if exist output\\debug_predictions.csv del output\\debug_predictions.csv
python src\\infer.py --config config_cpu.yaml
if errorlevel 1 (
  pause
  exit /b 1
)
python src\\check_submission.py --submission output\\submission.csv --input_csv input\\input.csv --classes classes.txt
if errorlevel 1 (
  pause
  exit /b 1
)
echo.
echo Final submission:
echo output\\submission.csv
pause
"""

CLEAN_OUTPUT_BAT = """@echo off
cd /d %~dp0
if exist output\\submission.csv del output\\submission.csv
if exist output\\debug_predictions.csv del output\\debug_predictions.csv
if exist output\\bad_crop.csv del output\\bad_crop.csv
if exist output\\missing_images.csv del output\\missing_images.csv
echo output cleaned.
pause
"""

README_DEPLOY = """# CPU Deploy Package

This package runs traffic-sign inference on a CPU-only Windows environment.

## Files

- `input/images/`: put exam images here.
- `input/input.csv`: exam CSV. Required header: `img,x1,y1,x2,y2`.
- `output/submission.csv`: final submission. Header is exactly `img,class`.
- `output/debug_predictions.csv`: top-3 debug output.
- `checkpoints/best_resnet50.pt`: packaged ResNet-50 checkpoint.

## First Run

1. Run `setup_cpu.bat`.
2. Run `check_env.bat`.
3. Put images in `input/images/`.
4. Replace `input/input.csv` with the exam CSV.
5. Run `run_exam_cpu.bat`.

The final answer file is `output/submission.csv`.

## CSV Contract

Input CSV:

```csv
img,x1,y1,x2,y2
sample.jpg,0,0,100,100
```

Submission CSV:

```csv
img,class
sample.jpg,101
```
"""


def _copy_src(project_root: Path, deploy_dir: Path) -> None:
    src_dir = project_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"missing source directory: {src_dir}")
    target_dir = deploy_dir / "src"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in RUNTIME_SRC_FILES:
        source = src_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def build_deploy_package(
    project_root: str | Path = ".",
    deploy_dir: str | Path = "deploy_cpu",
    make_zip: bool = True,
) -> Path:
    project_root = Path(project_root).resolve()
    deploy_dir = Path(deploy_dir)
    if not deploy_dir.is_absolute():
        deploy_dir = project_root / deploy_dir
    deploy_dir = deploy_dir.resolve()
    if deploy_dir == project_root:
        raise ValueError("deploy_dir must not be the project root")

    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)

    _copy_src(project_root, deploy_dir)

    checkpoint_src = project_root / "checkpoints" / "best_resnet50.pt"
    if not checkpoint_src.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_src}")
    checkpoint_dst = deploy_dir / "checkpoints" / "best_resnet50.pt"
    checkpoint_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_src, checkpoint_dst)

    classes_src = project_root / "classes.txt"
    if not classes_src.exists():
        raise FileNotFoundError(f"missing classes.txt: {classes_src}")
    shutil.copy2(classes_src, deploy_dir / "classes.txt")

    (deploy_dir / "input" / "images").mkdir(parents=True, exist_ok=True)
    (deploy_dir / "output").mkdir(parents=True, exist_ok=True)
    _write_text(deploy_dir / "input" / "input.csv", "img,x1,y1,x2,y2\n")
    _write_text(deploy_dir / "requirements_cpu.txt", REQUIREMENTS_CPU)
    _write_text(deploy_dir / "setup_cpu.bat", SETUP_CPU_BAT)
    _write_text(deploy_dir / "check_env.bat", CHECK_ENV_BAT)
    _write_text(deploy_dir / "run_exam_cpu.bat", RUN_EXAM_CPU_BAT)
    _write_text(deploy_dir / "clean_output.bat", CLEAN_OUTPUT_BAT)
    _write_text(deploy_dir / "README_DEPLOY.md", README_DEPLOY)
    _write_text(deploy_dir / "config_cpu.yaml", CONFIG_CPU)

    if make_zip:
        zip_base = project_root / "deploy_cpu"
        archive_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=deploy_dir))
        return archive_path
    return deploy_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="build the CPU deploy package")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--deploy_dir", default="deploy_cpu")
    parser.add_argument("--no_zip", action="store_true")
    args = parser.parse_args()
    result = build_deploy_package(args.project_root, args.deploy_dir, make_zip=not args.no_zip)
    print(result)


if __name__ == "__main__":
    main()
