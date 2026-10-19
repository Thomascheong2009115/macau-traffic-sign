@echo off
setlocal
cd /d "%~dp0\..\..\..\.."

if exist output\submission.csv del output\submission.csv
if exist output\debug_predictions.csv del output\debug_predictions.csv

python experiments\final_bbox_crop_rebuild\long_horizon_97plus\deployment\infer_exam_long_horizon.py --input_csv input\input.csv --image_dir input\images --output_csv output\submission.csv --debug_csv output\debug_predictions.csv --device cuda
if errorlevel 1 exit /b 1

if exist src\inference\check_submission.py (
  python src\inference\check_submission.py --input input\input.csv --submission output\submission.csv --classes classes.txt
) else if exist src\check_submission.py (
  python src\check_submission.py --input input\input.csv --submission output\submission.csv --classes classes.txt
) else if exist verify_submission_robust.py (
  python verify_submission_robust.py --submission output\submission.csv --input_csv input\input.csv --classes classes.txt
) else if exist experiments\cnn_student_diffusion_mixup_robust\verify_submission_robust.py (
  python experiments\cnn_student_diffusion_mixup_robust\verify_submission_robust.py --submission output\submission.csv --input_csv input\input.csv --classes classes.txt
)

endlocal
