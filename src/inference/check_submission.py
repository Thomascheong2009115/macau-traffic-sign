"""Check submission.csv format and content validity."""
import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.class_filter import is_valid_class, load_classes
from src.data_io import read_input_csv


SUBMISSION_HEADER = ["img", "class"]


def check_submission(
    submission_csv: str | Path = "output/submission.csv",
    input_csv: str | Path = "input/input.csv",
    classes_path: str | Path = "classes.txt",
) -> None:
    """Validate submission.csv against input.csv and classes.txt."""
    submission_csv = Path(submission_csv)
    if not submission_csv.exists():
        raise FileNotFoundError(f"submission.csv not found: {submission_csv}")

    classes = load_classes(classes_path, expected_count=None)
    input_rows, _ = read_input_csv(input_csv, image_dir=Path(input_csv).parent / "images")
    expected_imgs = [row.img for row in input_rows]

    with submission_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SUBMISSION_HEADER:
            raise ValueError(f"submission.csv columns must be exactly: {SUBMISSION_HEADER}")
        rows = list(reader)

    actual_imgs = [row.get("img", "").strip() for row in rows]
    if actual_imgs != expected_imgs:
        raise ValueError("submission.csv img values must exactly match input.csv order and content")

    for i, row in enumerate(rows, start=2):
        label = row.get("class", "").strip()
        if not label:
            raise ValueError(f"row {i}: class is empty")
        if not is_valid_class(label, classes):
            raise ValueError(f"row {i}: class '{label}' not in classes.txt")


def main():
    parser = argparse.ArgumentParser(description="Check submission format")
    parser.add_argument("--submission", default="output/submission.csv")
    parser.add_argument("--input", "--input_csv", dest="input_csv", default="input/input.csv")
    parser.add_argument("--classes", default="classes.txt")
    args = parser.parse_args()
    check_submission(args.submission, args.input_csv, args.classes)
    print("submission.csv check PASSED")


if __name__ == "__main__":
    main()
