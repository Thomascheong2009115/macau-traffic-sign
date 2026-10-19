"""TrafficSignDataset and loaders for the clean training pipeline.

Reads manifest CSV with columns: path, label, source, parent_path.
"""
import csv
import sys
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from src.class_filter import load_classes


def resolve_image_path(path_value: str, project_root: Path, manifest_dir: Path) -> Path:
    """Resolve image path from manifest entry.

    Tries: absolute → project_root/path → manifest_dir/path → project_root/path
    """
    normalized = str(path_value).strip().replace("\\", "/")
    raw = Path(normalized)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [
        project_root / normalized,
        manifest_dir / normalized,
    ]
    # Also try without 'data/' prefix if path starts with data/
    if normalized.lower().startswith("data/"):
        candidates.append(project_root / normalized[5:])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(f"Image not found: {path_value}\nTried:\n{tried}")


class TrafficSignDataset(Dataset):
    """Clean traffic sign dataset using new manifest format.

    Manifest columns: path, label, source, parent_path
    """

    def __init__(
        self,
        manifest_path: str | Path,
        classes_path: str | Path = "classes.txt",
        transform: Optional[Callable] = None,
        project_root: Optional[str | Path] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.classes = load_classes(classes_path)
        self.class_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.transform = transform
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.manifest_dir = self.manifest_path.parent
        self.samples: list[tuple[Path, str]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["path"].strip()
                label = row["label"].strip()
                if label not in self.class_to_idx:
                    raise ValueError(f"Label {label} not in classes.txt")
                image_path = resolve_image_path(path, self.project_root, self.manifest_dir)
                self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[label]
