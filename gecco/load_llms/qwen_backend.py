from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_qwen(model_name: str):
    print(f"[GeCCo] Loading Qwen model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model
