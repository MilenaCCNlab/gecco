# 🧠 GeCCo: Generative Cognitive Model Composer

**Author:** [Milena Rmus](https://github.com/MilenaCCNlab)  
**Last updated:** October 2025

---

## 📘 Overview

**GeCCo (Generative Cognitive Composer)** is a research framework for **automatically generating, fitting, and evaluating cognitive models** using large language models (LLMs).

Given behavioral data (e.g., from decision-making tasks), GeCCo prompts an LLM to generate candidate cognitive models as executable Python functions, fits them to data, evaluates them via information criteria (AIC/BIC), and iteratively improves model quality using structured feedback.

---

## 🧩 Key Features

- 🧠 **LLM-guided model generation** — models are produced as interpretable Python functions.
- ⚙️ **Flexible configuration** — YAML-based configs define task columns, metrics, and data formats.
- 📊 **Automatic fitting and evaluation** — supports multi-start optimization with AIC/BIC scoring.
- 🔁 **Iterative model search loop** — integrates structured or LLM-generated feedback.
- 🧮 **Task-agnostic** — supports any dataset by specifying relevant input columns.
- 🧱 **Modular architecture** — clean separation between generation, fitting, and evaluation.

---

## 📂 Repository Structure

gecco/
├── config/
│   ├── schema.py             # Config loader + validation
│   └── two_step.yaml         # Example config for two-step task
│
├── core/
│   ├── evaluation.py         # AIC/BIC and related metrics
│   ├── data_structures.py    # FitResult and ModelSpec definitions
│   └── __init__.py
│
├── engine/
│   ├── run_fit.py            # Fits an LLM-generated model to data
│   ├── model_search.py       # Iterative search loop for generating and evaluating models
│   └── feedback.py           # (Optional) custom feedback logic for LLM prompts
│
├── llm/
│   ├── generator.py          # Handles LLM prompting and text generation
│   └── prompt_builder.py     # Builds task- and iteration-specific prompts
│
├── utils/
│   ├── extraction.py         # Extracts code blocks, parameter names, and bounds from LLM output
│   └── misc.py               # Misc. utilities (safe exec, logging, etc.)
│
├── examples/
│   └── two_step_demo.py      # Example script showing full GeCCo workflow
│
└── results/
    ├── models/               # Saved model definitions per iteration
    └── bics/                 # BIC results for each model


## ⚙️ Configuration

All experiment and data parameters are specified in a YAML file (e.g., `config/two_step.yaml`):

```yaml
data:
  id_column: participant
  input_columns: [action_1, state, action_2, reward]

evaluation:
  metric: BIC
  n_starts: 10

loop:
  max_iterations: 5

llm:
  models_per_iteration: 3
```
## 🚀 Usage

1. **Prepare your dataset**  
   Your input DataFrame must contain all columns listed under `data.input_columns` in the config.

2. **Run the demo script**
   ```bash
   cd gecco/examples
   python two_step_demo.py
    ```

## 🧪 How It Works

- **Prompting** – A structured prompt (defined in `prompt_builder.py`) is sent to the LLM, instructing it to generate 1–3 candidate model functions (`cognitive_model1`, `cognitive_model2`, ...).

- **Code Extraction** – GeCCo extracts clean function definitions, parameter names, and parameter bounds from the model’s docstring using regex.

- **Model Fitting** – Each model is compiled and fit to each participant’s data using maximum likelihood estimation via `scipy.optimize.minimize`.

- **Evaluation** – Fit quality is assessed using the metric specified in the config (default: BIC).

- **Feedback Loop** – The system optionally provides feedback to the LLM about which models performed best or which parameter sets to avoid in subsequent iterations.

---

## 💬 Customizing Feedback

**Examples include:**
- **LLM-generated feedback**
- **Rule-based feedback**
- **User-defined textual hints**

Then import and plug into the main loop in `model_search.py`.

---

## 🧰 Requirements

- **Python ≥ 3.10**
- `torch`, `transformers`, `numpy`, `pandas`, `scipy`
- *Optional:* `vllm`, `unsloth`, `accelerate` for large-model inference



