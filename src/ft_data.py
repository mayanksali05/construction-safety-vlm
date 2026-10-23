"""Training data for fine-tuning: validation hold-out and balanced sampling."""

import hashlib
import math
import random

from .data import has_violation, violation_reason
from .rules import RULE_IDS


def is_val(image_id, val_frac=0.1):
    """True if this train image belongs to the validation hold-out.

    Decided from a hash of the image id, so the split is the same on every
    machine and every run, with no file to keep in sync.
    """
    h = int(hashlib.md5(str(image_id).encode()).hexdigest(), 16)
    return (h % 1000) < int(round(val_frac * 1000))


def split_indices(image_ids, val_frac=0.1):
    """Row indices of the train split -> (fit, val)."""
    fit, val = [], []
    for i, image_id in enumerate(image_ids):
        (val if is_val(image_id, val_frac) else fit).append(i)
    return fit, val


def make_answer(violated, reason="", with_reasons=True, max_words=60):
    """The text the model is trained to produce.

    The verdict comes first ("Yes" / "No"), so evaluation can read the first
    word and score its probability. For violations the annotator's reason
    follows, which teaches the model to explain itself.
    """
    if not violated:
        return "No."
    reason = " ".join((reason or "").split()[:max_words])
    return f"Yes. {reason}" if (with_reasons and reason) else "Yes."


def build_examples(meta, indices, neg_ratio=2, min_pos=200, max_repeat=5,
                   with_reasons=True, seed=0):
    """Build (row index, rule id, answer) training examples.

    The raw data is very unbalanced (about 3% of image-rule pairs are
    violations). For each rule we therefore:
      - keep every violation image, repeated so that rare rules reach about
        `min_pos` examples (at most `max_repeat` copies each), and
      - sample `neg_ratio` non-violation images per violation example.

    meta: list of dataset rows without the image column, indexed by row.
    Returns (examples, stats).
    """
    rng = random.Random(seed)
    examples, stats = [], {}
    for k in RULE_IDS:
        field = f"rule_{k}_violation"
        pos = [i for i in indices if has_violation(meta[i].get(field))]
        pos_set = set(pos)
        neg = [i for i in indices if i not in pos_set]
        repeat = min(max_repeat, max(1, math.ceil(min_pos / len(pos)))) if pos else 0
        n_neg = min(len(neg), neg_ratio * len(pos) * repeat)
        for i in pos * repeat:
            reason = violation_reason(meta[i].get(field))
            examples.append((i, k, make_answer(True, reason, with_reasons)))
        for i in rng.sample(neg, n_neg):
            examples.append((i, k, make_answer(False)))
        stats[k] = {"violation_images": len(pos), "copies_each": repeat,
                    "violation_examples": len(pos) * repeat, "no_violation_examples": n_neg}
    return examples, stats
