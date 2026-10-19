import argparse
import importlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infer import _load_config, _normalize_config


REQUIRED_IMPORTS = ["torch", "torchvision", "PIL", "numpy", "pandas", "cv2", "sklearn", "yaml", "tqdm"]


def _check_imports() -> None:
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)


def check_deploy_env(config_path: str | Path = "config_cpu.yaml") -> None:
    config_path = Path(config_path)
    config = _normalize_config(_load_config(config_path))
    if str(config.get("device", "")).lower() != "cpu":
        raise ValueError("config_cpu.yaml must set device: cpu")
    if int(config.get("tta", 0)) != 0:
        raise ValueError("config_cpu.yaml must set tta: 0")
    if int(config.get("coarse_filter", 0)) != 0:
        raise ValueError("config_cpu.yaml must set coarse_filter: 0")

    _check_imports()

    required_paths = [
        Path(config["checkpoint_path"]),
        Path(config["classes_path"]),
        Path(config["input_csv"]),
        Path(config["input_image_dir"]),
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"required deploy path was not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="check the CPU deploy environment")
    parser.add_argument("--config", default="config_cpu.yaml")
    args = parser.parse_args()
    check_deploy_env(args.config)
    print(f"Python: {sys.version.split()[0]}")
    print("ENV CHECK PASSED")


if __name__ == "__main__":
    main()
