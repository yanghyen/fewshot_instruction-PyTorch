from models.load_model import *

prompt = "What is the capital of Korea?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=20
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)