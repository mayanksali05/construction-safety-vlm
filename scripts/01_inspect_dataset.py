"""Step 0: download ConstructionSite 10k and check it against the paper.

Run this first (on a machine with internet, e.g. the cluster login node):

    python scripts/01_inspect_dataset.py

It prints the columns, the split sizes and the number of images with a
violation per rule, and says whether each number matches the paper. It also
saves one example image and its annotation to outputs/sample/.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.data import ground_truth  # noqa: E402
from src.rules import (  # noqa: E402
    DATASET_ID,
    EXPECTED_SPLIT_SIZES,
    EXPECTED_VIOLATIONS,
    RULE_IDS,
    RULE_NAMES,
)


def check(label, got, expected):
    if expected is None:
        print(f"  {label}: {got}")
    else:
        flag = "OK" if got == expected else f"MISMATCH (paper: {expected})"
        print(f"  {label}: {got}  [{flag}]")
    return expected is None or got == expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sample"))
    args = ap.parse_args()

    ds = load_dataset(DATASET_ID)
    print("Splits:", list(ds.keys()))
    first = next(iter(ds.values()))
    print("\nColumns:")
    for name, feat in first.features.items():
        print(f"  {name}: {feat}")

    all_ok = True
    for split, d in ds.items():
        print(f"\n== {split} ==")
        all_ok &= check("images", len(d), EXPECTED_SPLIT_SIZES.get(split))
        # Drop the image column so counting does not decode 10k JPEGs.
        meta = d.remove_columns([c for c in d.column_names if c == "image"])
        counts = {k: 0 for k in RULE_IDS}
        any_violation = 0
        for ex in meta:
            gt = ground_truth(ex)
            for k in RULE_IDS:
                counts[k] += gt[k]
            any_violation += any(gt.values())
        for k in RULE_IDS:
            exp = EXPECTED_VIOLATIONS.get(split, {}).get(k)
            all_ok &= check(f"rule {k} ({RULE_NAMES[k]}) violations", counts[k], exp)
        print(f"  images with at least one violation: {any_violation}")

    # Save one example with a violation so the raw format can be seen.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = ds["test"] if "test" in ds else first
    meta = d.remove_columns(["image"]) if "image" in d.column_names else d
    idx = next((i for i, ex in enumerate(meta) if any(ground_truth(ex).values())), 0)
    ex = d[idx]
    image = ex.pop("image", None)
    if image is not None:
        image.convert("RGB").save(out / "sample.jpg")
    (out / "sample.json").write_text(
        json.dumps(ex, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved one example to {out}")

    print(
        "\nAll counts match the paper."
        if all_ok
        else "\nSome counts do NOT match the paper. Open outputs/sample/sample.json, "
        "look at the rule_k_violation fields, and fix has_violation() in "
        "src/data.py before running the baseline."
    )


if __name__ == "__main__":
    main()
