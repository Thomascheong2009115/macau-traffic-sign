# Single Model Kaggle Deploy Package

This package intentionally follows the old working `traffic_sign_deploy_cpu` shape:

- `traffic_sign_deploy_cpu/src/ensemble_infer.py`
- `traffic_sign_deploy_cpu/checkpoints/`
- `traffic_sign_deploy_cpu/classes.txt`
- `traffic_sign_deploy_cpu/run_kaggle.py`

The implementation is now a single-model wrapper around the final bbox-crop release checkpoint:

- `checkpoints/final_single_resnet50.pt`

Kaggle output:

```text
/kaggle/working/submission.csv
```

The submission header is:

```csv
img,class
```

Use `run_kaggle.py` as the notebook cell code, or copy the separate `run_kaggle_single_final.py` generated next to the zip.
