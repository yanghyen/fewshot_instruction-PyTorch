from transformers import tokenizer
from models.load_model import *

def generate_response(prompt):
    inputs = tokenizer(
        prompt, 
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=10,
        do_sample=False
    )
    
    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )