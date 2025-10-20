class FeedbackGenerator:
    """
    Base feedback handler for guiding the LLM between iterations.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.history = []  # store per-iteration summaries if needed

    def record_iteration(self, iteration_idx, results):
        """
        Store results from a given iteration (list of dicts with param_names, etc.)
        """
        self.history.append({
            "iteration": iteration_idx,
            "results": results,
        })

    def get_feedback(self, best_model, tried_param_sets):
        """
        Construct feedback string for the next prompt.
        Default: discourage reuse of past parameter combinations.
        """

        previous_parameters = "\n".join([", ".join(s) for s in tried_param_sets])   # Summarize all tried parameter sets
        default_prompt = (
            f"Your best model so far:\n "
            f" {best_model}.\n"
            f"The parameter combinations tried so far:\n{previous_parameters}\n\n"
            "Avoid repeating these exact combinations, "
            "and explore alternative parameter configurations or mechanisms.\n"
        )

        if self.cfg.feedback.prompt:
            # Allow user to use {best_model} and {previous_parameters} in their custom prompt
            feedback = self.cfg.feedback.prompt.format(
                best_model=best_model,
                previous_parameters=previous_parameters
            )
        else:
            feedback = default_prompt

        return feedback


class LLMFeedbackGenerator(FeedbackGenerator):
    """
    Optional subclass: let an LLM summarize model search performance and propose directions.
    """

    def __init__(self, cfg, model, tokenizer):
        super().__init__(cfg)
        self.model = model
        self.tokenizer = tokenizer

    def get_feedback(self, best_model, tried_param_sets):
        """
        Construct feedback string for the next prompt using an LLM.
        """
        # Summarize all tried parameter sets
        previous_parameters = "\n".join([", ".join(s) for s in tried_param_sets])

        default_prompt = (
            f"The best model so far was:\n "
            f" {best_model}.\n"
            f"The following parameter combinations have already been explored:\n{previous_parameters}\n\n"
            "Please suggest high-level guidance for generating new model variants "
            "that differ conceptually but might still perform well."
        )

        if self.cfg.feedback.prompt is not None:
            prompt = self.cfg.feedback.prompt.format(
                best_model=best_model,
                previous_parameters=previous_parameters
            )
        else:
            prompt = default_prompt

        # from llm.generator import generate
        feedback_text = self.generate(prompt)
        return feedback_text.strip()

    def generate(self, prompt):
        """
        Unified text generation function for any supported backend.
        """
        if self.model is None:
            raise ValueError("Model not initialized correctly.")
        provider = self.cfg.llm.provider.lower()

        # -----------------------------
        # OpenAI / GPT-style generation
        # -----------------------------
        if "openai" in provider or "gpt" in provider:
            max_out = self.cfg.llm.max_output_tokens
            reasoning_effort = getattr(self.cfg.llm, "reasoning_effort", "medium")
            text_verbosity = getattr(self.cfg.llm, "text_verbosity", "low")

            print(
                f"[GeCCo] Using GPT model '{self.cfg.llm.base_model}' "
                f"(reasoning={reasoning_effort}, verbosity={text_verbosity}, max_output_tokens={max_out})"
            )

            resp = self.model.responses.create(
                model=self.cfg.llm.base_model,
                reasoning={"effort": "low"},
                input=[
                    {"role": "developer", "content": self.cfg.llm.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            decoded = resp.output_text.strip()

            return decoded
        # -----------------------------
        # Hugging Face-style generation
        # -----------------------------
        else:
            max_new = getattr(self.cfg.llm, "max_output_tokens", getattr(self.cfg.llm, "max_tokens", 2048))

            print(
                f"[GeCCo] Using HF model '{self.cfg.llm.base_model}' "
                f"(max_new_tokens={max_new}, temperature={self.cfg.llm.temperature})"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=self.cfg.llm.temperature,
                do_sample=True,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=self.cfg.llm.temperature,
                do_sample=True,
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)