import csv
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from src.class_filter import is_valid_class, load_classes


def _append_candidate(candidates: list[Path], candidate: Path) -> None:
    if candidate not in candidates:
        candidates.append(candidate)


def resolve_image_path(path_value: str | Path, project_root: str | Path, manifest_dir: str | Path) -> Path:
    """中文说明：按稳定顺序解析 manifest 图片路径，避免 data/data 这类重复拼接。

    解析顺序：
    1. 绝对路径存在则直接返回。
    2. project_root / path_value。
    3. manifest_dir / path_value。
    4. project_root / "data" / path_value，但 path_value 已经以 data/ 开头时跳过。
    """
    project_root = Path(project_root)
    manifest_dir = Path(manifest_dir)
    normalized = str(path_value).strip().replace("\\", "/")
    raw_path = Path(normalized)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        _append_candidate(candidates, raw_path)
        if raw_path.exists():
            return raw_path
    else:
        _append_candidate(candidates, project_root / raw_path)
        _append_candidate(candidates, manifest_dir / raw_path)
        starts_with_data = normalized.lower() == "data" or normalized.lower().startswith("data/")
        if not starts_with_data:
            _append_candidate(candidates, project_root / "data" / raw_path)
        for candidate in candidates:
            if candidate.exists():
                return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"image not found for manifest path: {path_value}\ntried candidate paths:\n{tried}")


class TrafficSignDataset(Dataset):
    """中文说明：读取训练/验证 manifest，支持 path,label,source 或旧 crop_path,label。"""

    def __init__(
        self,
        manifest_path: str | Path,
        classes_path: str | Path = "classes.txt",
        transform: Callable | None = None,
        project_root: str | Path | None = None,
        allowed_sources: set[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.classes = load_classes(classes_path)
        self.class_to_idx = {label: index for index, label in enumerate(self.classes)}
        self.transform = transform
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.manifest_dir = self.manifest_path.parent
        self.allowed_sources = allowed_sources
        self.samples: List[Tuple[Path, str]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == ["path", "label", "source"]:
                path_column = "path"
            elif reader.fieldnames == ["crop_path", "label"]:
                path_column = "crop_path"
            else:
                raise ValueError("manifest columns must be path,label,source or crop_path,label")
            for row_number, row in enumerate(reader, start=2):
                label = str(row["label"]).strip()
                if not is_valid_class(label, self.classes):
                    raise ValueError(f"manifest row {row_number}: label is not in classes.txt: {label}")
                source = row.get("source", "real").strip()
                if self.allowed_sources is not None and source not in self.allowed_sources:
                    raise ValueError(f"manifest row {row_number}: source is not allowed: {source}")
                try:
                    image_path = resolve_image_path(row[path_column], self.project_root, self.manifest_dir)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(f"manifest row {row_number}: {exc}") from exc
                self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[label]
