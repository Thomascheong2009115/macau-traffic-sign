import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.class_filter import is_valid_class, load_classes
from src.data_io import read_input_csv


SUBMISSION_HEADER = ["img", "class"]


def check_submission(
    submission_csv: str | Path = "output/submission.csv",
    input_csv: str | Path = "input/input.csv",
    classes_path: str | Path = "classes.txt",
) -> None:
    """严格检查 submission.csv，任何不合格都直接报错并指出行号。"""
    submission_csv = Path(submission_csv)
    if not submission_csv.exists():
        raise FileNotFoundError(f"submission.csv 不存在: {submission_csv}")

    classes = load_classes(classes_path, expected_count=None)
    input_rows, _ = read_input_csv(input_csv, image_dir=Path(input_csv).parent / "images")
    expected_imgs = [row.img for row in input_rows]

    with submission_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SUBMISSION_HEADER:
            raise ValueError("submission.csv columns must exactly be img,class")
        rows = list(reader)

    actual_imgs = [row.get("img", "") for row in rows]
    if actual_imgs != expected_imgs:
        raise ValueError("submission.csv img values must exactly match input.csv")

    for row_number, row in enumerate(rows, start=2):
        label = row.get("class", "").strip()
        if not label:
            raise ValueError(f"row {row_number}: class is empty")
        if not is_valid_class(label, classes):
            raise ValueError(f"row {row_number}: class is invalid or not in classes.txt: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 submission.csv 格式和类别合同")
    parser.add_argument("--submission", default="output/submission.csv")
    parser.add_argument("--input", "--input_csv", dest="input_csv", default="input/input.csv")
    parser.add_argument("--classes", default="classes.txt")
    args = parser.parse_args()
    check_submission(args.submission, args.input_csv, args.classes)
    print("submission.csv check passed")


if __name__ == "__main__":
    main()
