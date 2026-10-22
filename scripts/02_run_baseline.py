"""Plan step 1: zero-shot baseline of a base VLM on the test split.

    python scripts/02_run_baseline.py --limit 20      # quick smoke test
    python scripts/02_run_baseline.py                 # full test split

Writes one JSON line per image to outputs/baseline_<model>.jsonl with the
ground truth, the parsed prediction and the raw model text. The run can be
interrupted and restarted; images already in the file are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: E402

from src.data import ground_truth  # noqa: E402
from src.prompts import (  # noqa: E402
    build_prompt,
    build_rule_prompt,
    build_yesno_prompt,
    parse_answer,
    parse_rule_answer,
    parse_yesno_answer,
)  # noqa: E402
from src.rules import DATASET_ID, RULE_IDS  # noqa: E402


DTYPES = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def load_model(model_id, max_pixels, load_4bit=False, dtype="auto"):
    kwargs = {"torch_dtype": DTYPES[dtype], "device_map": "auto"}
    if load_4bit:
        # For small GPUs (e.g. 4 GB). Needs `pip install bitsandbytes`.
        from transformers import BitsAndBytesConfig

        kwargs["torch_dtype"] = torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    print(f"Model dtype: {model.dtype}")
    device_map = getattr(model, "hf_device_map", None)
    if device_map and any(str(d) in ("cpu", "disk") for d in device_map.values()):
        print("WARNING: part of the model was placed on CPU/disk. This will be very slow.")
    # Image size is limited by shrink() below, not by processor arguments,
    # because those arguments changed between transformers versions.
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"  # needed for batched generation
    return model, processor


def shrink(image, max_pixels):
    """Downscale so width x height <= max_pixels (keeps aspect ratio).

    Image tokens grow with resolution; a full-size site photo can need
    thousands of tokens, which overflows a small GPU and makes Windows
    spill into slow shared memory.
    """
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(28, int(w * scale)), max(28, int(h * scale))))


@torch.inference_mode()
def generate(model, processor, images, prompts, max_new_tokens):
    """One (image, prompt) pair per sequence; images and prompts are parallel lists."""
    texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = processor(
        text=texts, images=images, padding=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    n_in = inputs["input_ids"].shape[1]
    out = out[:, n_in:]
    return processor.batch_decode(out, skip_special_tokens=True), n_in, out.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument(
        "--dtype",
        choices=list(DTYPES),
        default="auto",
        help="weight precision. 'auto' uses the model's own setting (bf16 for "
        "Qwen2-VL), which is slow on older GPUs such as the T4; try fp16 there.",
    )
    ap.add_argument(
        "--short-answers",
        action="store_true",
        help="ask for true/false per rule only, no reasons. Much faster on a "
        "slow GPU. It is a different prompt, so do not mix its results with "
        "runs made without this flag.",
    )
    ap.add_argument(
        "--mode",
        choices=["joint", "per_rule", "yes_no"],
        default="joint",
        help="joint: one question covering all 4 rules. per_rule: one question "
        "per rule (4 per image), observation first, then yes/no. yes_no: one "
        "short question per rule answered with a single Yes or No.",
    )
    ap.add_argument("--limit", type=int, default=None, help="only the first N images")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--load-4bit",
        action="store_true",
        help="4-bit quantised model for small GPUs. Results differ from full "
        "precision, so use it for testing, not for the reported baseline.",
    )
    args = ap.parse_args()

    out_path = Path(
        args.out
        or ROOT
        / "outputs"
        / f"baseline_{args.model.split('/')[-1]}{'_4bit' if args.load_4bit else ''}_{args.mode}_{args.split}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["image_id"] for line in f if line.strip()}
        print(f"Resuming: {len(done)} images already done.")

    ds = load_dataset(DATASET_ID, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    model, processor = load_model(args.model, args.max_pixels, args.load_4bit, args.dtype)
    prompt = build_prompt(short=args.short_answers)
    build_one = {"per_rule": build_rule_prompt, "yes_no": build_yesno_prompt}.get(args.mode)
    parse_one = {"per_rule": parse_rule_answer, "yes_no": parse_yesno_answer}.get(args.mode)
    if args.mode == "yes_no":
        args.max_new_tokens = min(args.max_new_tokens, 8)  # one word is enough
    if build_one:
        print("Prompts (one per rule):")
        for k in RULE_IDS:
            print(f"--- rule {k} ---\n{build_one(k)}")
        print()
    else:
        print("Prompt:\n" + prompt + "\n")

    ids = ds["image_id"]
    todo = [i for i in range(len(ds)) if str(ids[i]) not in done]
    with open(out_path, "a", encoding="utf-8") as f:
        for start in tqdm(range(0, len(todo), args.batch_size)):
            rows = [ds[i] for i in todo[start : start + args.batch_size]]
            images = [shrink(r["image"].convert("RGB"), args.max_pixels) for r in rows]
            t0 = time.time()
            if build_one:
                outs, n_in, n_out = generate(
                    model,
                    processor,
                    [img for img in images for _ in RULE_IDS],
                    [build_one(k) for _ in images for k in RULE_IDS],
                    args.max_new_tokens,
                )
                results = []
                n = len(RULE_IDS)
                for i in range(len(rows)):
                    chunk = outs[i * n : (i + 1) * n]
                    pred, reasons = {}, {}
                    for k, t in zip(RULE_IDS, chunk):
                        pred[k], reasons[k] = parse_one(t)
                    raw = "\n".join(f"[rule {k}] {t}" for k, t in zip(RULE_IDS, chunk))
                    results.append((pred, reasons, raw))
            else:
                outs, n_in, n_out = generate(
                    model, processor, images, [prompt] * len(images), args.max_new_tokens
                )
                results = [(*parse_answer(t), t) for t in outs]
            if start < 5 * args.batch_size:  # diagnostics for the first batches
                peak = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                tqdm.write(
                    f"image {images[0].size[0]}x{images[0].size[1]} | "
                    f"{n_in} input tokens | {n_out} new tokens | "
                    f"{time.time() - t0:.1f}s | peak GPU memory {peak:.2f} GB"
                )
            for row, (pred, reasons, text) in zip(rows, results):
                gt = ground_truth(row)
                f.write(
                    json.dumps(
                        {
                            "image_id": str(row["image_id"]),
                            "model": args.model,
                            "mode": args.mode,
                            "gt": {str(k): gt[k] for k in RULE_IDS},
                            "pred": {str(k): pred[k] for k in RULE_IDS},
                            "pred_reason": {str(k): reasons[k] for k in RULE_IDS},
                            "raw": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f.flush()
    print(f"Done. Predictions in {out_path}")


if __name__ == "__main__":
    main()
