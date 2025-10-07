# engine/feedback.py

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

    def get_feedback(self, best_iter, tried_param_sets):
        """
        Construct feedback string for the next prompt.
        Default: discourage reuse of past parameter combinations.
        """
        feedback = (
            f"The best model so far (iteration {best_iter}).\n\n"
            "These are parameter combinations tried so far:\n"
        )

        for param_set in tried_param_sets:
            feedback += f"- {', '.join(param_set)}\n"

        feedback += (
            "\nAvoid repeating these exact combinations, "
            "and explore alternative parameter configurations or mechanisms.\n"
        )

        return feedback


class LLMFeedbackGenerator(FeedbackGenerator):
    """
    Optional subclass: let an LLM summarize model search performance and propose directions.
    """

    def __init__(self, cfg, llm, tokenizer):
        super().__init__(cfg)
        self.llm = llm
        self.tokenizer = tokenizer

    def get_feedback(self, best_iter, tried_param_sets):
        # Summarize all tried parameter sets
        summary = "\n".join([", ".join(s) for s in tried_param_sets])

        prompt = (
            f"In a model search loop, the best model so far was found at iteration {best_iter}.\n"
            f"The following parameter combinations have already been explored:\n{summary}\n\n"
            "Please suggest high-level guidance for generating new model variants "
            "that differ conceptually but might still perform well."
        )

        from llm.generator import generate
        feedback_text = generate(self.llm, self.tokenizer, prompt, self.cfg)
        return feedback_text.strip()
