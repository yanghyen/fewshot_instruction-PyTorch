import torch

def format_piqa_example(example, include_answer=True):
    answer = "A" if example["label"] == 0 else "B"
    
    text = (
        f"Question: {example['goal']}\n"
        f"A. {example['sol1']}\n"
        f"B. {example['sol2']}\n"
    )
    
    if include_answer:
        text += f"Answer: {answer}\n"
        
    return text

def build_prompt(test_example, demo_examples):
    prompt = "Choose the more appropriate solution. Answer with A or B. \n\n"
    
    for demo in demo_examples:
        prompt += format_piqa_example(demo, include_answer=True)
        prompt += "\n"
        
    prompt += format_piqa_example(test_example, include_answer=False)
    prompt += "Answer:"
    
    return prompt

def extract_answer(text):
    text = text.strip()
    
    if len(text) == 0:
        return None
    
    first_char = text[0].upper()
    
    if first_char == "A":
        return 0
    elif first_char == "B":
        return 1
    else:
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
        
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]      # prompt까지 decode하지 않고 생성된 답변만 가져옴 
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