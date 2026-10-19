"""
后融合二分类专家验证器。
加载多个二分类专家，针对特定混淆对进行后融合复核。
无 GT 泄漏 — 仅使用模型输出和专家置信度。
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

# 每个验证器定义: (触发类集合, checkpoint, 阈值, 名称)
# 触发条件: ensemble_top1 在 trigger_set 中 或 ensemble_top3 与 trigger_set 有交集
VERIFIERS = [
    {
        "name": "105_vs_138",
        "trigger_set": {"105", "138"},
        "checkpoint": "checkpoints/expert_105_vs_138.pt",
        "classes": ["105", "138"],
        "threshold": 0.55,
        "trigger_confidence_max": 0.45,
    },
    {
        "name": "113_vs_107",
        "trigger_set": {"107", "113"},
        "checkpoint": "checkpoints/expert_113_vs_107.pt",
        "classes": ["107", "113"],
        "threshold": 0.55,
        "trigger_confidence_max": 0.50,
    },
]


def _safe_conf(value) -> str:
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError):
        return str(value)


def load_expert(checkpoint_path: Path, classes: list[str], device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_classes = checkpoint.get("classes", classes)
    model_name = checkpoint.get("model_name", "resnet50")
    model = build_model(model_name, len(ckpt_classes), pretrained=False).to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()
    return model, list(ckpt_classes)


def predict_expert(model, transform, image: Image.Image, device: torch.device, classes: list[str]) -> tuple[str, float]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0]
    probs = torch.softmax(logits, dim=0)
    top_idx = int(probs.argmax().item())
    return classes[top_idx], float(probs[top_idx].item())


def apply_verifiers(
    debug_csv: Path,
    input_csv: Path,
    image_dir: Path,
    output_debug: Path,
    output_submission: Path,
    image_size: int = 256,
    pad_to_square: bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_val_transform(image_size, pad_to_square=pad_to_square)

    # 预加载所有专家
    experts = []
    for vdef in VERIFIERS:
        model, classes = load_expert(Path(vdef["checkpoint"]), vdef["classes"], device)
        experts.append({**vdef, "model": model, "classes": classes})

    rows_data, _ = read_input_csv(input_csv, image_dir)
    row_map: dict[str, InputRow] = {r.img: r for r in rows_data}

    triggered = 0
    overridden = 0
    output_debug.parent.mkdir(parents=True, exist_ok=True)

    with open(debug_csv, "r", encoding="utf-8-sig", newline="") as in_handle, \
         output_debug.open("w", encoding="utf-8-sig", newline="") as out_handle:
        reader = csv.DictReader(in_handle)
        out_fields = list(reader.fieldnames or []) + ["verifier_name", "verifier_pred", "verifier_conf", "verifier_override", "final_class"]
        writer = csv.DictWriter(out_handle, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            img = row["img"].strip()
            ensemble_top1 = row.get("ensemble_top1", "").strip()
            ensemble_top1_conf = float(row.get("ensemble_top1_conf", "0") or "0")
            ensemble_top2 = row.get("ensemble_top2", "").strip()
            ensemble_top3 = row.get("ensemble_top3", "").strip()
            rule = row.get("rule", "")

            verifier_name = ""
            verifier_pred = ""
            verifier_conf = 0.0
            verifier_override = 0
            final_class = ensemble_top1

            ensemble_top_set = {ensemble_top1, ensemble_top2, ensemble_top3}

            for expert_def in experts:
                trigger_set = expert_def["trigger_set"]
                # 触发：ensemble_top1 在触发集中 且 置信度低
                if ensemble_top1 in trigger_set and ensemble_top1_conf < expert_def["trigger_confidence_max"]:
                    input_row = row_map.get(img)
                    if input_row is not None:
                        triggered += 1
                        verifier_name = expert_def["name"]
                        result, crop = crop_sign_image(
                            input_row.image_path, input_row.bbox, padding_ratio=0.10, clamp_padding=False
                        )
                        if result.ok and crop is not None:
                            expert_pred, expert_conf = predict_expert(
                                expert_def["model"], transform, crop, device, expert_def["classes"]
                            )
                            verifier_pred = expert_pred
                            verifier_conf = expert_conf

                            if expert_pred != ensemble_top1 and expert_conf > expert_def["threshold"]:
                                overridden += 1
                                verifier_override = 1
                                final_class = expert_pred
                        break  # 只触发第一个匹配的验证器

            new_row = dict(row)
            new_row["verifier_name"] = verifier_name
            new_row["verifier_pred"] = verifier_pred
            new_row["verifier_conf"] = _safe_conf(verifier_conf) if verifier_pred else ""
            new_row["verifier_override"] = str(verifier_override)
            new_row["final_class"] = final_class
            writer.writerow(new_row)

    # 写 submission
    with output_debug.open("r", encoding="utf-8-sig", newline="") as debug_handle, \
         output_submission.open("w", encoding="utf-8-sig", newline="") as sub_handle:
        reader = csv.DictReader(debug_handle)
        writer = csv.writer(sub_handle)
        writer.writerow(["img", "class"])
        for row in reader:
            writer.writerow([row["img"], row.get("final_class", row.get("ensemble_top1", ""))])

    return {"triggered": triggered, "overridden": overridden}


def main() -> None:
    parser = argparse.ArgumentParser(description="apply binary expert verifiers post-fusion")
    parser.add_argument("--debug_csv", default="output/debug_xn_gamma.csv")
    parser.add_argument("--input_csv", default="input/input.csv")
    parser.add_argument("--image_dir", default="input/images")
    parser.add_argument("--output_debug", default="output/debug_binary_verifiers.csv")
    parser.add_argument("--output_submission", default="output/submission_binary_verifiers.csv")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pad_to_square", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    stats = apply_verifiers(
        Path(args.debug_csv), Path(args.input_csv), Path(args.image_dir),
        Path(args.output_debug), Path(args.output_submission),
        args.image_size, bool(args.pad_to_square),
    )
    print(f"triggered={stats['triggered']} overridden={stats['overridden']}")


if __name__ == "__main__":
    main()
