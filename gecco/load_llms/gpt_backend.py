import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def load_gpt(model_name: str):
    print(f"[GeCCo] Initializing OpenAI GPT model: {model_name}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment or .env file.")
    return OpenAI(api_key=api_key)
