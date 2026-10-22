"""Prompt and answer parsing for the zero-shot baseline.

NOTE: this prompt is our own. It is not the prompt used in the
ConstructionSite 10k paper. Before comparing our numbers with the paper's
GPT-4o / LLaVA numbers, check the paper's prompt and evaluation protocol.
"""

import json
import re

from .rules import RULES, RULE_IDS


def build_prompt(short=False):
    lines = [
        "You are a construction site safety inspector. Look at the image "
        "and check it against these four safety rules.",
        "",
    ]
    for k in RULE_IDS:
        lines.append(f"Rule {k}: {RULES[k]}")
    lines += [
        "",
        "For each rule, decide whether the image shows a violation of that "
        "rule. If the rule does not apply to this image, it is not violated.",
        "",
    ]
    if short:
        lines += [
            "Answer with JSON only, in exactly this form, and nothing else:",
            '{"rule_1": true or false, "rule_2": true or false, '
            '"rule_3": true or false, "rule_4": true or false}',
            "true means the rule is violated.",
        ]
        return "\n".join(lines)
    lines += [
        "Answer with JSON only, in exactly this form:",
        '{"rule_1": {"violated": true or false, "reason": "short reason"}, '
        '"rule_2": {"violated": true or false, "reason": "short reason"}, '
        '"rule_3": {"violated": true or false, "reason": "short reason"}, '
        '"rule_4": {"violated": true or false, "reason": "short reason"}}',
    ]
    return "\n".join(lines)


_TRUE = {"true", "yes", "1", "violated", "violation"}
_FALSE = {"false", "no", "0", "not violated", "no violation", "none", "null"}


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    return None


def parse_answer(text):
    """Model text -> ({rule_id: bool or None}, {rule_id: reason}).

    None means the answer for that rule could not be read.
    """
    pred = {k: None for k in RULE_IDS}
    reasons = {k: "" for k in RULE_IDS}

    obj = None
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
        except ValueError:
            obj = None

    if isinstance(obj, dict):
        for k in RULE_IDS:
            item = obj.get(f"rule_{k}", obj.get(str(k)))
            if isinstance(item, dict):
                pred[k] = _to_bool(item.get("violated"))
                reasons[k] = str(item.get("reason") or "")
            else:
                pred[k] = _to_bool(item)

    # Fallback for broken JSON: read `"rule_k": {"violated": true` directly.
    for k in RULE_IDS:
        if pred[k] is None:
            m = re.search(
                rf'rule[_ ]?{k}\b.{{0,40}}?violated"?\s*[:=]\s*"?(true|false|yes|no)',
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not m:  # short form: "rule_k": true
                m = re.search(
                    rf'rule[_ ]?{k}"?\s*[:=]\s*"?(true|false|yes|no)\b',
                    text,
                    flags=re.IGNORECASE,
                )
            if m:
                pred[k] = _to_bool(m.group(1))
    return pred, reasons


def build_rule_prompt(rule_id):
    """One rule per question. The model describes what it sees BEFORE it
    gives the verdict, and answers yes/no to a plainly worded question.

    Reason: with the joint prompt, Qwen2-VL-2B wrote "violated": false for
    every rule even when its own reason described a violation.
    """
    return "\n".join(
        [
            "You are a construction site safety inspector.",
            "",
            f"Safety rule: {RULES[rule_id]}",
            "",
            "Look at the image.",
            "Step 1: in one sentence, describe what in the image is relevant "
            "to this rule. If nothing in the image is relevant to it, say so.",
            "Step 2: decide whether the image shows this rule being broken.",
            "",
            "Answer with JSON only, in exactly this form:",
            '{"observation": "one sentence", "rule_broken": "yes" or "no"}',
        ]
    )


def parse_rule_answer(text):
    """Model text -> (bool or None, observation)."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return _to_bool(obj.get("rule_broken")), str(obj.get("observation") or "")
        except ValueError:
            pass
    m = re.search(r'rule_broken"?\s*[:=]\s*"?(yes|no|true|false)\b', text, flags=re.IGNORECASE)
    return (_to_bool(m.group(1)) if m else None), ""


# Short yes/no prompts, one rule per question.
# Question 1 is prompt 1 from Sahraoui (2025), "Automated Hazard Detection in
# Construction Sites Using Large Language and Vision-Language Models"
# (arXiv:2511.15720), which reports that Qwen2-VL-2B does better with short,
# single-rule, yes/no prompts. That thesis covers the PPE rule only, and does
# not print the exact rule line it appends, so:
#   - the "Rule: ..." line here is our own wording of that line, and
#   - questions 2-4 are our own, written in the same style.
YESNO_QUESTIONS = {
    1: "In the image, is anyone not wearing basic PPE?",
    2: "In the image, is anyone working at height near an unprotected edge without a safety harness?",
    3: "In the image, is there a deep excavation without edge protection or edge warning?",
    4: "In the image, is any worker in the blind spot or within the operation radius of an excavator?",
}


def build_yesno_prompt(rule_id):
    return f"{YESNO_QUESTIONS[rule_id]}\nRule: {RULES[rule_id]}\nAnswer only: Yes or No."


def parse_yesno_answer(text):
    """Model text -> (bool or None, ''). Uses the first yes/no word."""
    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return (m.group(1).lower() == "yes" if m else None), ""
