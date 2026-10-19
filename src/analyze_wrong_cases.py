import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ERROR_TYPES = (
    "crop_possible",
    "coarse_group_error",
    "fine_grained_confusion",
    "low_confidence",
    "high_confidence_wrong",
)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("._") or "unknown"


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def classify_error_type(row: dict[str, str]) -> str:
    true_label = row.get("true_class", row.get("true", "")).strip()
    pred_label = row.get("pred_class", row.get("top1", "")).strip()
    confidence = _float_or_zero(row.get("confidence", row.get("top1_confidence", "")))
    top_labels = {row.get("top1", "").strip(), row.get("top2", "").strip(), row.get("top3", "").strip()}

    if confidence >= 0.85:
        return "high_confidence_wrong"
    if confidence < 0.55:
        return "low_confidence"
    if true_label[:1] and pred_label[:1] and true_label[:1] != pred_label[:1]:
        return "coarse_group_error"
    if true_label in top_labels or true_label[:1] == pred_label[:1]:
        return "fine_grained_confusion"
    return "crop_possible"


def analyze_wrong_cases(
    wrong_csv: str | Path,
    image_dir: str | Path = "input/images",
    output_dir: str | Path = "output/wrong_cases_images",
    analysis_csv: str | Path = "output/wrong_case_analysis.csv",
) -> dict[str, int]:
    wrong_csv = Path(wrong_csv)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    analysis_csv = Path(analysis_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_csv.parent.mkdir(parents=True, exist_ok=True)

    counts = {error_type: 0 for error_type in ERROR_TYPES}
    analysis_rows: list[list[str]] = []
    with wrong_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img = row.get("IMG", row.get("image", row.get("img", ""))).strip()
            true_label = row.get("true_class", row.get("true", "")).strip()
            pred_label = row.get("pred_class", row.get("top1", "")).strip()
            confidence = _float_or_zero(row.get("confidence", row.get("top1_confidence", "")))
            error_type = classify_error_type(row)
            counts[error_type] += 1

            source = Path(img)
            if not source.is_absolute():
                source = image_dir / img
            suffix = source.suffix if source.suffix else ".jpg"
            dest_name = (
                f"{_safe_token(Path(img).stem)}_true_{_safe_token(true_label)}_"
                f"pred_{_safe_token(pred_label)}_conf_{confidence:.3f}{suffix}"
            )
            dest = output_dir / dest_name
            if source.exists():
                shutil.copy2(source, dest)
            analysis_rows.append([img, true_label, pred_label, f"{confidence:.6f}", error_type, str(dest)])

    with analysis_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["IMG", "true_class", "pred_class", "confidence", "error_type", "copied_path"])
        writer.writerows(analysis_rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="copy and classify wrong prediction images")
    parser.add_argument("--wrong_csv", default="output/wrong_cases.csv")
    parser.add_argument("--image_dir", default="input/images")
    parser.add_argument("--output_dir", default="output/wrong_cases_images")
    parser.add_argument("--analysis_csv", default="output/wrong_case_analysis.csv")
    args = parser.parse_args()

    counts = analyze_wrong_cases(args.wrong_csv, args.image_dir, args.output_dir, args.analysis_csv)
    for error_type in ERROR_TYPES:
        print(f"{error_type}={counts[error_type]}")


if __name__ == "__main__":
    main()
