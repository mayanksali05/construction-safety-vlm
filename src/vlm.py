"""Model helpers shared by fine-tuning (04) and probability scoring (05).

The baseline script (02) keeps its own copies so its results stay
reproducible exactly as they were run.
"""

import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Text that closes the assistant turn in the Qwen2-VL chat template.
END_OF_TURN = "<|im_end|>"


def load_model(model_id, dtype="fp16", load_4bit=True):
    """Qwen2-VL and its processor.

    load_4bit=True stores the frozen weights in 4-bit NF4 (QLoRA). dtype is
    the compute precision: fp16 on a T4 (no fast bf16), bf16 on A100/H100.
    """
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    kwargs = {"torch_dtype": DTYPES[dtype], "device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPES[dtype],
        )
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def shrink(image, max_pixels):
    """Downscale so width x height <= max_pixels (keeps aspect ratio)."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(28, int(w * scale)), max(28, int(h * scale))))


def prompt_text(processor, prompt):
    """Chat-formatted user turn with one image, ending where the answer starts."""
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def yes_no_ids(tokenizer):
    """Token ids of "Yes" and "No" as the first word of an answer."""
    ids = {}
    for word in ("Yes", "No"):
        toks = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            raise ValueError(f"'{word}' is not a single token: {toks}")
        ids[word] = toks[0]
    return ids["Yes"], ids["No"]
