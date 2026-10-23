"""Pick a decision threshold per rule on the validation set, apply it to test.

    python scripts/06_thresholds.py --val outputs/pyes_ft_qlora_r16_val.jsonl \
                                    --test outputs/pyes_ft_qlora_r16_test.jsonl
    python scripts/03_evaluate.py outputs/pyes_ft_qlora_r16_test.tuned.jsonl

For each rule, the threshold on P(yes) that gives the best F1 on the
validation images is chosen and then used, unchanged, on the test split.
The test labels are never used to choose anything.

Also prints average precision (AP) on test for each rule. AP does not
depend on any threshold, so it is a fair second measure, especially for the
rare rules where the validation set has only a handful of violations.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rules import RULE_IDS, RULE_NAMES  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def f1_at(scores, labels, t):
    tp = sum(1 for s, y in zip(scores, labels) if s >= t and y)
    fp = sum(1 for s, y in zip(scores, labels) if s >= t and not y)
    fn = sum(1 for s, y in zip(scores, labels) if s < t and y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def best_threshold(scores, labels):
    """Threshold with the highest F1. Candidates are the observed scores;
    on ties the higher threshold wins (fewer false alarms)."""
    if not any(labels):
        return 0.5, 0.0
    best = (0.5, -1.0)
    for t in sorted(set(scores)):
        f = f1_at(scores, labels, t)
        if f >= best[1]:
            best = (t, f)
    return best


def average_precision(scores, labels):
    """Area under the precision-recall curve (step-wise, as in sklearn)."""
    pos = sum(labels)
    if pos == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:  # tied scores together
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        recall = tp / pos
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
        i = j
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    val, test = load(args.val), load(args.test)
    thresholds = {}
    print(f"{'Rule':<24}{'val viol.':>10}{'threshold':>11}{'val F1':>8}{'test AP':>9}")
    for k in RULE_IDS:
        key = str(k)
        vs = [r["p_yes"][key] for r in val]
        vy = [bool(r["gt"][key]) for r in val]
        t, f = best_threshold(vs, vy)
        thresholds[key] = t
        test_ap = average_precision([r["p_yes"][key] for r in test], [int(bool(r["gt"][key])) for r in test])
        print(f"{str(k) + ' ' + RULE_NAMES[k]:<24}{sum(vy):>10}{t:>11.4f}{f:>8.3f}{test_ap:>9.3f}")

    out = Path(args.out or Path(args.test).with_suffix(".tuned.jsonl"))
    with open(out, "w", encoding="utf-8") as f:
        for r in test:
            r = dict(r)
            r["pred"] = {key: r["p_yes"][key] >= thresholds[key] for key in thresholds}
            r["thresholds"] = thresholds
            f.write(json.dumps(r) + "\n")
    Path(out).with_suffix(".thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(f"\nTest predictions with tuned thresholds: {out}")
    print(f"Next: python scripts/03_evaluate.py {out}")


if __name__ == "__main__":
    main()
