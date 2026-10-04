"""Dataset helpers shared by the scripts."""

import json

from .rules import RULE_IDS


def has_violation(value):
    """True if a rule_k_violation field records a violation.

    The dataset card says the field is null when there is no violation and
    an object with `bounding_box` and `reason` when there is one. Depending
    on how the `datasets` library decodes it, "no violation" can also arrive
    as a dict whose values are all empty, or as a string. All are handled.
    """
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"null", "none", "nan"}:
            return False
        try:
            return has_violation(json.loads(text))
        except (ValueError, TypeError):
            return True  # non-empty free text
    if isinstance(value, dict):
        return any(_non_empty(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_non_empty(v) for v in value)
    return bool(value)


def _non_empty(v):
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    return True


def violation_reason(value):
    """The annotator's reason text, or '' if absent."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value
    if isinstance(value, dict):
        return value.get("reason") or ""
    return ""


def ground_truth(example):
    """{rule_id: bool} for one dataset row."""
    return {k: has_violation(example.get(f"rule_{k}_violation")) for k in RULE_IDS}
