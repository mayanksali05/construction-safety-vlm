"""Per-rule precision / recall / F1 from a predictions file.

    python scripts/03_evaluate.py outputs/baseline_Qwen2-VL-2B-Instruct_test.jsonl

A violation is the positive class. If the model's answer for a rule could
not be read, it is counted as "no violation" and reported in the
`unparsed` column, so nothing is silently dropped.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rules import RULE_IDS, RULE_NAMES  # noqa: E402


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def evaluate(records):
    rows = []
    for k in RULE_IDS:
        tp = fp = fn = tn = unparsed = 0
        for rec in records:
            gt = bool(rec["gt"][str(k)])
            pred = rec["pred"][str(k)]
            if pred is None:
                unparsed += 1
                pred = False
            if gt and pred:
                tp += 1
            elif pred:
                fp += 1
            elif gt:
                fn += 1
            else:
                tn += 1
        p, r, f1 = prf(tp, fp, fn)
        rows.append(
            {
                "rule": f"{k} {RULE_NAMES[k]}",
                "support": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "unparsed": unparsed,
                "precision": p,
                "recall": r,
                "f1": f1,
            }
        )
    macro = {m: sum(r[m] for r in rows) / len(rows) for m in ("precision", "recall", "f1")}
    tp, fp, fn = (sum(r[c] for r in rows) for c in ("tp", "fp", "fn"))
    micro = dict(zip(("precision", "recall", "f1"), prf(tp, fp, fn)))
    return rows, macro, micro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    args = ap.parse_args()

    path = Path(args.predictions)
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        sys.exit("No predictions in file.")

    rows, macro, micro = evaluate(records)

    print(f"Model: {records[0].get('model', '?')}   Images: {len(records)}\n")
    head = f"{'Rule':<24}{'Support':>8}{'TP':>6}{'FP':>6}{'FN':>6}{'Unpars.':>9}{'P':>8}{'R':>8}{'F1':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['rule']:<24}{r['support']:>8}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
            f"{r['unparsed']:>9}{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}"
        )
    print("-" * len(head))
    for name, m in (("Macro average", macro), ("Micro average", micro)):
        print(f"{name:<59}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")

    out_csv = path.with_suffix(".metrics.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for name, m in (("macro", macro), ("micro", micro)):
            w.writerow({"rule": name, **m})
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
