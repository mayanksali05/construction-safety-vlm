"""Score every image and rule with P(yes), for the base or fine-tuned model.

    # fine-tuned model on the validation hold-out, then on the test split
    python scripts/05_predict.py --adapter outputs/ft_qlora_r16/final --split val
    python scripts/05_predict.py --adapter outputs/ft_qlora_r16/final --split test

    # the same for the base model (no adapter), for a like-for-like baseline
    python scripts/05_predict.py --adapter none --split val
    python scripts/05_predict.py --adapter none --split test

Instead of generating text, the model reads the yes/no prompt once and we
take the probability that its first word is "Yes" (vs "No"). This is fast
and gives a score per image, so a decision threshold can be tuned on the
validation set (06_thresholds.py).

--split val means the 10% of the train split that 04_finetune.py held out.

Output: JSON lines in the same format as the baseline (gt, pred), plus
p_yes. "pred" here uses a 0.5 threshold. 03_evaluate.py reads it directly.
The run can be interrupted and restarted.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data import ground_truth  # noqa: E402
from src.ft_data import is_val  # noqa: E402
from src.prompts import build_yesno_prompt  # noqa: E402
from src.rules import DATASET_ID, RULE_IDS  # noqa: E402
from src.vlm import load_model, prompt_text, shrink, yes_no_ids  # noqa: E402


@torch.inference_mode()
def p_yes(model, processor, images, prompts, yes_id, no_id):
    texts = [prompt_text(processor, p) for p in prompts]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    try:
        logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]
    except TypeError:  # older transformers
        logits = model(**inputs).logits[:, -1, :]
    two = logits[:, [yes_id, no_id]].float()
    return torch.softmax(two, dim=-1)[:, 0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--adapter", required=True, help="LoRA folder, or 'none' for the base model")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--val-frac", type=float, default=0.1, help="must match 04_finetune.py")
    ap.add_argument("--batch-size", type=int, default=2, help="images per batch (x4 prompts)")
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument(
        "--load-4bit", action="store_true",
        help="4-bit base weights, exactly as in training. Default: 16-bit base "
        "weights with the adapter on top, which is faster on a T4 and is the "
        "same setting as the baseline runs.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = "base" if args.adapter == "none" else Path(args.adapter).resolve().parent.name
    out_path = Path(args.out or ROOT / "outputs" / f"pyes_{name}_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["image_id"] for line in f if line.strip()}
        print(f"Resuming: {len(done)} images already done.")

    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train" if args.split == "val" else "test")
    ids = [str(x) for x in ds["image_id"]]
    keep = [i for i, x in enumerate(ids) if args.split == "test" or is_val(x, args.val_frac)]
    if args.limit:
        keep = keep[: args.limit]
    todo = [i for i in keep if ids[i] not in done]
    print(f"{args.split}: {len(keep)} images, {len(todo)} to do")

    model, processor = load_model(args.model, args.dtype, load_4bit=args.load_4bit)
    if args.adapter != "none":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor.tokenizer.padding_side = "left"  # last position = where the answer starts
    yes_id, no_id = yes_no_ids(processor.tokenizer)

    with open(out_path, "a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(todo), args.batch_size)):
            rows = [ds[i] for i in todo[start : start + args.batch_size]]
            images = [shrink(r["image"].convert("RGB"), args.max_pixels) for r in rows]
            probs = p_yes(
                model, processor,
                [img for img in images for _ in RULE_IDS],
                [build_yesno_prompt(k) for _ in images for k in RULE_IDS],
                yes_id, no_id,
            )
            n = len(RULE_IDS)
            for j, row in enumerate(rows):
                gt = ground_truth(row)
                p = dict(zip(RULE_IDS, probs[j * n : (j + 1) * n]))
                f.write(json.dumps({
                    "image_id": str(row["image_id"]),
                    "model": args.model,
                    "adapter": args.adapter,
                    "mode": "yes_no_prob",
                    "gt": {str(k): gt[k] for k in RULE_IDS},
                    "p_yes": {str(k): round(p[k], 6) for k in RULE_IDS},
                    "pred": {str(k): p[k] >= 0.5 for k in RULE_IDS},
                    "pred_reason": {str(k): "" for k in RULE_IDS},
                    "raw": "",
                }) + "\n")
            f.flush()
    print(f"Done. Scores in {out_path}")


if __name__ == "__main__":
    main()
