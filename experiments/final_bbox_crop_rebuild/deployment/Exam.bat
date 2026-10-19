@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0\..\..\.."

echo ============================================================
echo   Traffic Sign Recognition - Final BBox Crop Rebuild
echo   Model: ResNet50 crop-first single checkpoint
echo   Val: 0.9696969697
echo ============================================================
echo.

set "PYTHON_EXE=%CD%\.venv3060\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  set "PYTHON_EXE=python"
)

set "INPUT_CSV=input\input.csv"
set "IMAGE_DIR=input\images"
set "OUTPUT_CSV=output\submission.csv"
set "DEBUG_CSV=output\debug_predictions.csv"
set "CONFIG=experiments\final_bbox_crop_rebuild\deployment\exam_best_config.json"
set "CLASSES=classes.txt"

if not exist "%INPUT_CSV%" (
  echo [ERROR] Missing input CSV: %INPUT_CSV%
  exit /b 1
)
if not exist "%IMAGE_DIR%" (
  echo [ERROR] Missing image directory: %IMAGE_DIR%
  exit /b 1
)
if not exist "%CONFIG%" (
  echo [ERROR] Missing exam config: %CONFIG%
  exit /b 1
)
if not exist "%CLASSES%" (
  echo [ERROR] Missing classes file: %CLASSES%
  exit /b 1
)
if not exist output mkdir output

echo [INFO] Python:      %PYTHON_EXE%
echo [INFO] Input CSV:   %INPUT_CSV%
echo [INFO] Image dir:   %IMAGE_DIR%
echo [INFO] Output CSV:  %OUTPUT_CSV%
echo [INFO] Config:      %CONFIG%
echo.

"%PYTHON_EXE%" experiments\final_bbox_crop_rebuild\deployment\infer_exam_final.py ^
  --input_csv "%INPUT_CSV%" ^
  --image_dir "%IMAGE_DIR%" ^
  --output_csv "%OUTPUT_CSV%" ^
  --debug_csv "%DEBUG_CSV%" ^
  --config "%CONFIG%" ^
  --classes "%CLASSES%" ^
  --device cuda

if %ERRORLEVEL% neq 0 (
  echo [ERROR] Inference failed with exit code %ERRORLEVEL%.
  exit /b %ERRORLEVEL%
)

"%PYTHON_EXE%" experiments\cnn_student_diffusion_mixup_robust\verify_submission_robust.py ^
  --submission "%OUTPUT_CSV%" ^
  --input_csv "%INPUT_CSV%" ^
  --classes "%CLASSES%"

if %ERRORLEVEL% neq 0 (
  echo [ERROR] Submission verification failed.
  exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   SUCCESS - Submission ready: %OUTPUT_CSV%
echo ============================================================
endlocal
