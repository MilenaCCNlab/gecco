"""
Cognitive Library Learning v3 - Configuration
==============================================

Shared configuration for all learning steps.
"""

import os
import sys

# Add gecco to path
sys.path.insert(0, "/home/aj9225/gecco-1")

# Directories
BASE_DIR = "/home/aj9225/gecco-1"
RESULTS_DIR = f"{BASE_DIR}/results/two_step_psychiatry_individual_stai_class_individual"
MODELS_DIR = f"{RESULTS_DIR}/models"
BICS_DIR = f"{RESULTS_DIR}/bics"
DATA_PATH = f"{BASE_DIR}/data/two_step_gillan_2016.csv"
OUTPUT_DIR = f"{RESULTS_DIR}/cognitive_library_auto"

# LLM Configuration
MODEL_NAME = "gemini-3-pro-preview"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_gemini_client():
    """Load the Gemini client using GECCO's backend."""
    from google import genai
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment or .env file.")
    
    client = genai.Client()
    return client


def call_gemini(client, prompt: str, model_name: str = MODEL_NAME) -> str:
    """Call Gemini API and return the response text."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text


def extract_code_block(response: str, language: str = "python") -> str:
    """Extract code block from LLM response."""
    import re
    pattern = rf"```{language}\s*(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try without language specifier
    pattern = r"```\s*(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def load_model_files() -> dict:
    """Load all participant model files."""
    import re
    models = {}
    for filename in sorted(os.listdir(MODELS_DIR)):
        if filename.startswith("best_model_") and filename.endswith(".txt"):
            match = re.search(r"participant(\d+)", filename)
            if match:
                pid = f"p{match.group(1)}"
                with open(os.path.join(MODELS_DIR, filename)) as f:
                    content = f.read()
                # Extract code from markdown
                code_match = re.search(r"```(?:python|plaintext)?\s*(.*?)```", content, re.DOTALL)
                if code_match:
                    models[pid] = code_match.group(1).strip()
                else:
                    models[pid] = content.strip()
    return models


def load_stored_bics() -> dict:
    """Load stored BIC values."""
    import json
    import re
    bics = {}
    for filename in os.listdir(BICS_DIR):
       
        if not filename.endswith('.json'):
            continue
        
        match = re.search(r"participant(\d+)", filename)
        if not match:
            continue
        pid = f"p{match.group(1)}"
        # load bic from best_bic_0_participant{id}.json, which contains bic as {"bic": 307.1535993317867}
        if filename.startswith(f"best_bic_0_participant{match.group(1)}"):
            bics[pid] = data["bic"]
            continue
        else:
            with open(os.path.join(BICS_DIR, filename)) as f:
                data = json.load(f)
            # For other files, find the minimum bic
            for model in data:
                if pid not in bics or model["metric_value"] < bics[pid]:
                    bics[pid] = model["metric_value"]
    return bics


if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    models = load_model_files()
    print(f"Found {len(models)} participant models")
    
    bics = load_stored_bics()
    print(f"Found {len(bics)} BIC values")
