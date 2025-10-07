from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_llama(model_name: str):
    print(f"[GeCCo] Loading LLaMA model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
    )
    return tokenizer, model
