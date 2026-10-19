from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
FINAL_INFER = PROJECT_ROOT / "experiments" / "final_bbox_crop_rebuild" / "deployment" / "infer_exam_final.py"


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(
            [
                "--config",
                str(
                    PROJECT_ROOT
                    / "experiments"
                    / "final_bbox_crop_rebuild"
                    / "long_horizon_97plus"
                    / "deployment"
                    / "exam_best_config_long_horizon.json"
                ),
            ]
        )
    runpy.run_path(str(FINAL_INFER), run_name="__main__")

