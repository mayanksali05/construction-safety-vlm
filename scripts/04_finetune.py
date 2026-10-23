"""Plan step 2: QLoRA fine-tuning of Qwen2-VL-2B on the train split.

    # smoke test: 20 optimizer steps, checks memory and speed
    python scripts/04_finetune.py --out-dir outputs/ft_smoke --max-steps 20

    # full run
    python scripts/04_finetune.py --out-dir outputs/ft_qlora_r16

What is trained
  - The base model is frozen and stored in 4-bit (QLoRA).
  - Small LoRA adapters are added to the attention layers of the language
    model (q/k/v/o_proj). The vision encoder is not changed.

Training data (see src/ft_data.py)
  - Uses the train split only. 10% of it (chosen by a hash of image_id) is
    held out as a validation set and never trained on; 05/06 use it to pick
    decision thresholds. The test split is never touched here.
  - One example = one image + one rule. The question is the same yes/no
    prompt as the baseline (src/prompts.py), so before/after is a fair
    comparison.
  - Target answer: "Yes. <annotator's reason>" or "No."
  - Balanced: rare-rule violations are repeated and only some no-violation
    images are used, otherwise the model learns to always say "No".

Resuming
  The adapter and optimizer are saved to <out-dir>/last every --save-every
  steps. Run the same command again after a Colab disconnect and it carries
  on from there. The finished adapter goes to <out-dir>/final.
"""

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.ft_data import build_examples, split_indices  # noqa: E402
from src.prompts import build_yesno_prompt  # noqa: E402
from src.rules import DATASET_ID, RULE_NAMES  # noqa: E402
from src.vlm import END_OF_TURN, load_model, prompt_text, shrink  # noqa: E402


def make_batch(processor, rows, items, max_pixels):
    """Tokenise a batch and build labels that cover only the answer.

    items: list of (row index, rule id, answer). Every token before the
    answer (the image and the question) gets label -100, so the loss is
    computed only on "Yes. <reason><|im_end|>" / "No.<|im_end|>".
    """
    tok = processor.tokenizer
    images, texts, answers = [], [], []
    for (i, k, answer) in items:
        images.append(shrink(rows[i]["image"].convert("RGB"), max_pixels))
        target = answer + END_OF_TURN
        texts.append(prompt_text(processor, build_yesno_prompt(k)) + target)
        answers.append(tok(target, add_special_tokens=False)["input_ids"])
    enc = processor(text=texts, images=images, padding=True, return_tensors="pt")
    labels = torch.full_like(enc["input_ids"], -100)
    lengths = enc["attention_mask"].sum(dim=1).tolist()  # right padding
    for b, (ans, n) in enumerate(zip(answers, lengths)):
        tail = enc["input_ids"][b, n - len(ans) : n].tolist()
        if tail != ans:
            raise RuntimeError(
                "Answer tokens not found at the end of the sequence; label "
                f"masking would be wrong.\nexpected {ans}\nfound    {tail}"
            )
        labels[b, n - len(ans) : n] = enc["input_ids"][b, n - len(ans) : n]
    enc["labels"] = labels
    return enc


def save_state(model, optimizer, scheduler, scaler, step, out_dir):
    """Write <out-dir>/last atomically enough to survive a disconnect."""
    tmp, last = out_dir / "last_tmp", out_dir / "last"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
        },
        tmp / "trainer_state.pt",
    )
    if last.exists():
        shutil.rmtree(last)
    tmp.rename(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="fraction of steps")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28, help="same as the baseline")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16", help="fp16 on T4, bf16 on A100")
    ap.add_argument("--no-4bit", action="store_true", help="plain LoRA on 16-bit weights (needs more memory)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--neg-ratio", type=int, default=2)
    ap.add_argument("--min-pos", type=int, default=200)
    ap.add_argument("--max-repeat", type=int, default=5)
    ap.add_argument("--no-reasons", action="store_true", help='train on "Yes." / "No." only')
    ap.add_argument("--save-every", type=int, default=50, help="optimizer steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------- data ----------
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")
    meta = ds.remove_columns("image").to_list()
    fit_idx, val_idx = split_indices([m["image_id"] for m in meta], args.val_frac)
    examples, stats = build_examples(
        meta, fit_idx, neg_ratio=args.neg_ratio, min_pos=args.min_pos,
        max_repeat=args.max_repeat, with_reasons=not args.no_reasons, seed=args.seed,
    )
    print(f"Train split: {len(meta)} images -> {len(fit_idx)} for training, {len(val_idx)} held out for validation")
    for k, s in stats.items():
        print(f"  rule {k} {RULE_NAMES[k]:<22} violation images {s['violation_images']:>4} "
              f"x{s['copies_each']} = {s['violation_examples']:>4} | no-violation examples {s['no_violation_examples']:>4}")
    print(f"  total training examples per epoch: {len(examples)}")

    # Fixed order of micro-batches for the whole run, so a resumed run
    # continues exactly where it stopped.
    per_step = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(examples) / per_step)
    total_steps = math.ceil(args.epochs * steps_per_epoch)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    order = []
    epoch = 0
    while len(order) < total_steps * per_step:
        shuffled = examples[:]
        random.Random(args.seed + epoch).shuffle(shuffled)
        order += shuffled
        epoch += 1
    order = order[: total_steps * per_step]
    print(f"Optimizer steps: {total_steps} (batch {args.batch_size} x accumulation {args.grad_accum})")

    (out_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "data_stats": stats, "n_train_examples": len(examples),
                    "n_fit_images": len(fit_idx), "n_val_images": len(val_idx),
                    "total_steps": total_steps}, indent=2)
    )

    # ---------- model ----------
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict

    model, processor = load_model(args.model, args.dtype, load_4bit=not args.no_4bit)
    processor.tokenizer.padding_side = "right"
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    # q/k/v/o_proj exist only in the language model (the vision encoder
    # uses fused "qkv"), so the vision encoder stays frozen.
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(args.warmup * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if args.dtype == "fp16" else None

    start_step = 0
    last = out_dir / "last"
    if last.exists():
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(last / "adapter_model.safetensors")))
        state = torch.load(last / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        print(f"Resuming from step {start_step} of {total_steps}")
    if start_step >= total_steps:
        print("Already finished.")

    # ---------- training ----------
    model.train()
    device = next(p for p in model.parameters() if p.requires_grad).device
    log_path = out_dir / "train_log.csv"
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["step", "loss", "lr", "sec_per_step", "peak_gpu_gb"])

    running, t0 = [], time.time()
    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for m in range(args.grad_accum):
            s = (step * args.grad_accum + m) * args.batch_size
            items = order[s : s + args.batch_size]
            batch = make_batch(processor, ds, items, args.max_pixels).to(device)
            if step == start_step and m == 0:
                sup = batch["labels"][0][batch["labels"][0] != -100]
                print(f"First example: rule {items[0][1]}, {batch['input_ids'].shape[1]} tokens")
                print(f"  supervised text: {processor.tokenizer.decode(sup)!r}")
            with torch.autocast("cuda", dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16):
                loss = model(**batch).loss / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            step_loss += loss.item()
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        running.append(step_loss)

        done = step + 1
        if done % 10 == 0 or done == total_steps or done - start_step <= 3:
            sec = (time.time() - t0) / (done - start_step)
            peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            avg = sum(running) / len(running)
            eta = sec * (total_steps - done) / 60
            print(f"step {done}/{total_steps} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{sec:.1f}s/step | peak GPU {peak:.1f} GB | about {eta:.0f} min left", flush=True)
            writer.writerow([done, f"{avg:.5f}", f"{scheduler.get_last_lr()[0]:.3e}", f"{sec:.2f}", f"{peak:.2f}"])
            log.flush()
            running = []
        if done % args.save_every == 0 and done < total_steps:
            save_state(model, optimizer, scheduler, scaler, done, out_dir)
            print(f"  saved checkpoint at step {done}")

    log.close()
    save_state(model, optimizer, scheduler, scaler, total_steps, out_dir)
    model.save_pretrained(out_dir / "final")
    print(f"Done. Adapter saved in {out_dir / 'final'}")


if __name__ == "__main__":
    main()
