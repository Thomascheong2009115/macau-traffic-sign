import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_COLUMNS = ("IMG", "X1", "Y1", "X2", "Y2")


@dataclass(frozen=True)
class InputRow:
    """考试输入 CSV 的一行，IMG 保持原始文件名不改写。"""
    row_number: int
    img: str
    x1: float
    y1: float
    x2: float
    y2: float
    image_path: Path

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def _normalize_header(header: str) -> str:
    return header.strip().upper()


def _build_column_map(fieldnames: List[str] | None) -> Dict[str, str]:
    if not fieldnames:
        raise ValueError("input.csv 缺少表头")
    mapping = {_normalize_header(name): name for name in fieldnames}
    missing = [name for name in REQUIRED_COLUMNS if name not in mapping]
    if missing:
        raise ValueError(f"input.csv 缺少必要列: {', '.join(missing)}")
    return mapping


def read_input_csv(csv_path: str | Path, image_dir: str | Path | None = None) -> Tuple[List[InputRow], List[str]]:
    """读取考试 CSV，兼容大小写和表头空格，并检查图片是否存在。"""
    csv_path = Path(csv_path)
    if image_dir is None:
        image_dir = csv_path.parent / "images"
    image_dir = Path(image_dir)

    rows: List[InputRow] = []
    missing_images: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        column_map = _build_column_map(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            img = row[column_map["IMG"]].strip()
            try:
                x1 = float(row[column_map["X1"]])
                y1 = float(row[column_map["Y1"]])
                x2 = float(row[column_map["X2"]])
                y2 = float(row[column_map["Y2"]])
            except ValueError as exc:
                raise ValueError(f"第 {row_number} 行 bbox 坐标不是数字") from exc

            image_path = Path(img)
            if not image_path.is_absolute():
                image_path = image_dir / img
            if not image_path.exists():
                missing_images.append(str(image_path))
            rows.append(InputRow(row_number, img, x1, y1, x2, y2, image_path))
    return rows, missing_images


def write_missing_report(missing_images: List[str], output_path: str | Path) -> None:
    """把缺失图片写成 CSV，方便考试前快速定位文件问题。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["missing_image"])
        for item in missing_images:
            writer.writerow([item])
