import torch

CHOICE_LABELS = ["A", "B", "C", "D"]


def format_hellaswag_example(example, include_answer=True):
    gold_label = int(example["label"])

    text = f"Context: {example['ctx']}\n"

    for idx, ending in enumerate(example["endings"]):
        text += f"{CHOICE_LABELS[idx]}. {ending}\n"

    if include_answer:
        text += f"Answer: {CHOICE_LABELS[gold_label]}\n"

    return text


def build_prompt(test_example, demo_examples):
    prompt = (
        "Choose the most plausible continuation for the context. "
        "Answer with A, B, C, or D.\n\n"
    )

    for demo in demo_examples:
        prompt += format_hellaswag_example(demo, include_answer=True)
        prompt += "\n"

    prompt += format_hellaswag_example(test_example, include_answer=False)
    prompt += "Answer:"

    return prompt


def extract_answer(text):
    text = text.strip()

    if len(text) == 0:
        return None

    first_char = text[0].upper()

    if first_char in CHOICE_LABELS:
        return CHOICE_LABELS.index(first_char)

    return None


def generate_answer(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text


def score_answer(prompt, answer, model, tokenizer):
    full_text = prompt + answer

    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    labels = inputs["input_ids"].clone()
    labels[:, :prompt_inputs["input_ids"].shape[-1]] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    return -outputs.loss.item()