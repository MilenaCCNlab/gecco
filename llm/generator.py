# llm/generator.py
import torch

def generate(model, tokenizer=None, prompt=None, cfg=None):
    """
    Unified text generation function for any supported backend.
    Handles both OpenAI GPT and Hugging Face-style models cleanly.
    """
    if model is None:
        raise ValueError("Model not initialized correctly.")
    provider = cfg.llm.provider.lower()

    # -----------------------------
    # OpenAI / GPT-style generation
    # -----------------------------
    if "openai" in provider or "gpt" in provider:
        max_out = cfg.llm.max_output_tokens
        reasoning_effort = getattr(cfg.llm, "reasoning_effort", "medium")
        text_verbosity = getattr(cfg.llm, "text_verbosity", "low")

        print(
            f"[GeCCo] Using GPT model '{cfg.llm.base_model}' "
            f"(reasoning={reasoning_effort}, verbosity={text_verbosity}, max_output_tokens={max_out})"
        )

        resp = model.responses.create(
            model=cfg.llm.base_model,
            reasoning={"effort": "low"},
            input=[
                {"role": "developer", "content": cfg.llm.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        decoded = resp.output_text.strip()

        return decoded

    # -----------------------------
    # Hugging Face-style generation
    # -----------------------------
    else:
        max_new = getattr(cfg.llm, "max_output_tokens", getattr(cfg.llm, "max_tokens", 2048))

        print(
            f"[GeCCo] Using HF model '{cfg.llm.base_model}' "
            f"(max_new_tokens={max_new}, temperature={cfg.llm.temperature})"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=cfg.llm.temperature,
            do_sample=True,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)
