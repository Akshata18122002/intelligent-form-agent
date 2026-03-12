import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(override=False)


class LLMClient:
    def __init__(self):
        # Prefer explicit OpenRouter env var, fall back to OPENAI_API_KEY if set.
        api_key = os.getenv("OPENROUTER_API_KEY") 

        if not api_key:
            raise RuntimeError(
                "No API key found. Please set OPENROUTER_API_KEY in your .env file."
            )

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        self.model = model

    def generate(self, prompt: str):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content