"""The four safety rules of ConstructionSite 10k.

Rule text is copied word for word from the dataset card:
https://huggingface.co/datasets/LouisChen15/ConstructionSite
Do not paraphrase it here; the model must see the same wording the
annotators used.
"""

DATASET_ID = "LouisChen15/ConstructionSite"

RULES = {
    1: (
        "Use of basic PPE when on foot at construction sites. (hard hats, "
        "properly worn clothes covering shoulders and legs, shoes that can "
        "cover toes, high-visibility retroreflective vests at night, face "
        "shield or safety glasses when cutting, welding, grinding, or "
        "drilling)."
    ),
    2: (
        "Use of safety harness when working from a height of three meters "
        "and the edges are without any edge protection."
    ),
    3: (
        "Adoption of edge protection or edge warning including guardrails, "
        "fences, for underground projects three meters in depth with steep "
        "retaining wall and for human to stand."
    ),
    4: (
        "Appearance of worker in the blind spots of the operator and within "
        "the operation radius of excavators in operation, or excavators with "
        "operators inside."
    ),
}

# Short names used only in tables and logs.
RULE_NAMES = {
    1: "PPE",
    2: "Harness at height",
    3: "Edge protection",
    4: "Excavator blind spot",
}

RULE_IDS = sorted(RULES)

# Images with a violation, per rule, as recorded in the project handoff note
# (taken from the ConstructionSite 10k paper). 01_inspect_dataset.py checks
# the downloaded data against these.
EXPECTED_VIOLATIONS = {
    "train": {1: 677, 2: 59, 3: 109, 4: 46},
    "test": {1: 323, 2: 25, 3: 63, 4: 24},
}
EXPECTED_SPLIT_SIZES = {"train": 7009, "test": 3004}
