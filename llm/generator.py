import torch

def generate_model_code(tokenizer, model, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """Generate cognitive model code from the LLM given a structured prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
