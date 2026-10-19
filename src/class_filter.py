import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence


# 中文说明：这里是当前 ID 对应表的显式白名单，不能用连续数字区间自动生成。
# 217 是有效类别；218 是缺号，必须排除；217X 等带字母类别也必须排除。
EXPECTED_CLASSES = (
    "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "120",
    "121", "122", "123", "124", "125", "126", "127", "128", "129", "130",
    "131", "132", "133", "134", "135", "136", "137", "138", "201", "202",
    "203", "204", "205", "206", "207", "208", "209", "210", "211", "212",
    "213", "214", "215", "216", "217", "219", "301", "302", "303", "304",
    "305", "306", "307", "308", "309", "310", "311", "312", "313", "314",
    "315", "316", "317", "318", "319", "320", "321", "322", "323", "324",
    "325", "326", "327", "328", "329", "330", "331", "332", "333",
)
EXPECTED_CLASS_COUNT = len(EXPECTED_CLASSES)
EXCLUDED_LABELS = {"135X", "136X", "137X", "138X", "217X", "331F", "332F", "333F", "218"}
THREE_DIGIT_RE = re.compile(r"^\d{3}$")


def _format_is_three_digit(label: object) -> bool:
    value = str(label).strip()
    return value.isdigit() and len(value) == 3 and THREE_DIGIT_RE.fullmatch(value) is not None


def is_valid_class(label: object, classes: Sequence[str] | None = None) -> bool:
    """中文说明：类别必须是纯三位数字符串，并且必须在显式白名单中。"""
    value = str(label).strip()
    whitelist = set(EXPECTED_CLASSES if classes is None else classes)
    return _format_is_three_digit(value) and value in whitelist and value not in EXCLUDED_LABELS


def normalize_classes(labels: Iterable[object], expected_count: int | None = EXPECTED_CLASS_COUNT) -> List[str]:
    """中文说明：从外部 ID 表过滤合法类别；不会自动补入缺号 218。"""
    classes = sorted({str(label).strip() for label in labels if is_valid_class(label)})
    if expected_count is not None and len(classes) != expected_count:
        raise ValueError(f"classes.txt must contain {expected_count} whitelist classes, got {len(classes)}")
    return classes


def load_classes(classes_path: str | Path = "classes.txt", expected_count: int | None = EXPECTED_CLASS_COUNT) -> List[str]:
    """中文说明：读取 classes.txt，并默认强制匹配当前 89 类显式白名单。"""
    path = Path(classes_path)
    if not path.exists():
        raise FileNotFoundError(f"classes file not found: {path}")
    raw_labels = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    classes = normalize_classes(raw_labels, expected_count=expected_count)
    if raw_labels != classes:
        raise ValueError("classes.txt must be sorted, unique, and contain only explicit whitelist labels")
    if expected_count == EXPECTED_CLASS_COUNT and tuple(classes) != EXPECTED_CLASSES:
        raise ValueError("classes.txt must exactly match the current explicit 89-class whitelist")
    return classes


def write_classes(labels: Iterable[object], output_path: str | Path, expected_count: int | None = EXPECTED_CLASS_COUNT) -> List[str]:
    """中文说明：根据外部 ID 表写出 classes.txt；只保留显式白名单中的类别。"""
    classes = normalize_classes(labels, expected_count=expected_count)
    Path(output_path).write_text("\n".join(classes) + "\n", encoding="utf-8")
    return classes


def main() -> None:
    parser = argparse.ArgumentParser(description="check or write explicit traffic sign classes.txt")
    parser.add_argument("--source", help="optional source ID file, one label per line")
    parser.add_argument("--output", default="classes.txt")
    args = parser.parse_args()

    if args.source:
        labels = Path(args.source).read_text(encoding="utf-8-sig").splitlines()
        classes = write_classes(labels, args.output)
        print(f"wrote {args.output}: {len(classes)} classes")
    else:
        classes = load_classes(args.output)
        print(f"class check passed: {len(classes)} classes")


if __name__ == "__main__":
    main()
