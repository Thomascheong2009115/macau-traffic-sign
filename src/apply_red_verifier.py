"""
后融合红组验证器：在不修改 infer_xn_gamma.py 主流程的前提下，
对 ensemble_top1 属于红组的高风险预测进行专家复核。
仅当专家高置信度地给出不同预测时才覆盖。
"""
import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src.augment import build_val_transform
from src.crop import crop_sign_image
from src.data_io import InputRow, read_input_csv
from src.model import build_model, load_checkpoint

RED_GROUP = {"105", "110", "113", "118", "138"}
RED_EXPERT_CLASSES = ["105", "110", "113", "118", "138"]
RED_CLASS_TO_IDX = {c: i for i, c in enumerate(RED_EXPERT_CLASSES)}

# 验证器参数
CONFIDENCE_THRESHOLD = 0.55  # 专家必须达到此置信度才覆盖
ENSEMBLE_CONF_MAX = 0.45     # ensemble 置信度低于此值才触发验证

DEBUG_FIELDS = [
    "img",
    "ensemble_top1",
    "ensemble_top1_conf",
    "verifier_triggered",
    "verifier_pred",
    "verifier_conf",
    "verifier_override",
    "final_class",
    "final_conf",
    "rule",
]


def _safe_conf(value) -> str:
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError):
        return str(value)


def load_red_expert(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint.get("classes", RED_EXPERT_CLASSES)
    model_name = checkpoint.get("model_name", "resnet50")
    model = build_model(model_name, len(classes), pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()
    return model, list(classes)


def predict_expert(model, transform, image: Image.Image, device: torch.device, classes: list[str]) -> tuple[str, float]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0]
    probs = torch.softmax(logits, dim=0)
    top_idx = int(probs.argmax().item())
    return classes[top_idx], float(probs[top_idx].item())


def apply_red_verifier(
    debug_csv: Path,
    input_csv: Path,
    image_dir: Path,
    expert_checkpoint: Path,
    output_debug: Path,
    output_submission: Path,
    image_size: int = 256,
    pad_to_square: bool = True,
) -> dict[str, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_model, expert_classes = load_red_expert(expert_checkpoint, device)
    transform = build_val_transform(image_size, pad_to_square=pad_to_square)
    rows, _ = read_input_csv(input_csv, image_dir)
    row_map: dict[str, InputRow] = {r.img: r for r in rows}

    triggered = 0
    overridden = 0
    output_debug.parent.mkdir(parents=True, exist_ok=True)

    with open(debug_csv, "r", encoding="utf-8-sig", newline="") as in_handle, \
         output_debug.open("w", encoding="utf-8-sig", newline="") as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(out_handle, fieldnames=DEBUG_FIELDS)
        writer.writeheader()

        for row in reader:
            img = row["img"].strip()
            ensemble_top1 = row.get("ensemble_top1", "").strip()
            ensemble_top1_conf = float(row.get("ensemble_top1_conf", "0") or "0")
            rule = row.get("rule", "")

            verifier_triggered = 0
            verifier_pred = ""
            verifier_conf = 0.0
            verifier_override = 0
            final_class = ensemble_top1
            final_conf = ensemble_top1_conf

            # 触发条件：ensemble_top1 属于红组 且 置信度低
            if ensemble_top1 in RED_GROUP and ensemble_top1_conf < ENSEMBLE_CONF_MAX:
                input_row = row_map.get(img)
                if input_row is not None:
                    triggered += 1
                    verifier_triggered = 1
                    result, crop = crop_sign_image(
                        input_row.image_path, input_row.bbox, padding_ratio=0.10, clamp_padding=False
                    )
                    if result.ok and crop is not None:
                        expert_pred, expert_conf = predict_expert(
                            expert_model, transform, crop, device, expert_classes
                        )
                        verifier_pred = expert_pred
                        verifier_conf = expert_conf

                        # 仅当专家高置信度且预测不同时才覆盖
                        if expert_pred != ensemble_top1 and expert_conf > CONFIDENCE_THRESHOLD:
                            overridden += 1
                            verifier_override = 1
                            final_class = expert_pred
                            final_conf = expert_conf

            writer.writerow({
                "img": img,
                "ensemble_top1": ensemble_top1,
                "ensemble_top1_conf": _safe_conf(ensemble_top1_conf),
                "verifier_triggered": str(verifier_triggered),
                "verifier_pred": verifier_pred,
                "verifier_conf": _safe_conf(verifier_conf) if verifier_pred else "",
                "verifier_override": str(verifier_override),
                "final_class": final_class,
                "final_conf": _safe_conf(final_conf),
                "rule": f"{rule}+red_verifier" if verifier_override else rule,
            })

    # Write submission
    with output_debug.open("r", encoding="utf-8-sig", newline="") as debug_handle, \
         output_submission.open("w", encoding="utf-8-sig", newline="") as sub_handle:
        reader = csv.DictReader(debug_handle)
        writer = csv.writer(sub_handle)
        writer.writerow(["img", "class"])
        for row in reader:
            writer.writerow([row["img"], row["final_class"]])

    return {"triggered": triggered, "overridden": overridden}


def main() -> None:
    parser = argparse.ArgumentParser(description="apply red group verifier post-fusion")
    parser.add_argument("--debug_csv", default="output/debug_xn_gamma.csv")
    parser.add_argument("--input_csv", default="input/input.csv")
    parser.add_argument("--image_dir", default="input/images")
    parser.add_argument("--expert_checkpoint", default="checkpoints/expert_red_candidate.pt")
    parser.add_argument("--output_debug", default="output/debug_red_verifier.csv")
    parser.add_argument("--output_submission", default="output/submission_red_verifier.csv")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pad_to_square", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    stats = apply_red_verifier(
        Path(args.debug_csv),
        Path(args.input_csv),
        Path(args.image_dir),
        Path(args.expert_checkpoint),
        Path(args.output_debug),
        Path(args.output_submission),
        args.image_size,
        bool(args.pad_to_square),
    )
    print(f"triggered={stats['triggered']} overridden={stats['overridden']}")


if __name__ == "__main__":
    main()
