import argparse
import csv
import shutil
from pathlib import Path


def select_best_checkpoint(results: str | Path, output: str | Path) -> dict[str, str]:
    """中文说明：从随机搜索结果中选 best_score 最高的 checkpoint 作为最终模型。"""
    results = Path(results)
    output = Path(output)
    if not results.exists():
        raise FileNotFoundError(f"search results not found: {results}")
    with results.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("search_results.csv is empty")
    best = max(rows, key=lambda row: float(row["best_score"]))
    checkpoint = Path(best["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"best checkpoint not found: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, output)
    return {
        "trial": best["trial"],
        "best_score": best["best_score"],
        "best_top1_acc": best["best_top1_acc"],
        "best_macro_f1": best.get("best_macro_f1_present_classes", best.get("best_macro_f1", "")),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="copy best search checkpoint to final checkpoint path")
    parser.add_argument("--results", default="output/search/search_results.csv")
    parser.add_argument("--output", default="checkpoints/best_resnet50.pt")
    args = parser.parse_args()
    result = select_best_checkpoint(args.results, args.output)
    print(f"best_trial={result['trial']}")
    print(f"best_score={result['best_score']}")
    print(f"best_top1_acc={result['best_top1_acc']}")
    print(f"best_macro_f1={result['best_macro_f1']}")
    print(f"copied_checkpoint={result['output']}")


if __name__ == "__main__":
    main()
