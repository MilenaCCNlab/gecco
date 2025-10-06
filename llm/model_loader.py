from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(model_name: str):
    """Load model + tokenizer (basic interface for now)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    return tokenizer, model
