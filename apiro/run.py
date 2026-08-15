import argparse
import sys
import logging
from pathlib import Path

# `requests` was only imported inside generate(), so generate_with_logprobs()
# raised NameError the moment anything called it.
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("run")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

class OllamaLLMClient:
    def __init__(self, url, model):
        self.url = url
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 180}},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def generate_with_logprobs(self, prompt: str) -> tuple[str, list]:
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 180},
            "logprobs": True,
        }
        resp = requests.post(f"{self.url}/api/generate", json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", ""), data.get("logprobs", [])

    def chat(self, prompt: str) -> str:
        return self.generate(prompt)

if __name__ == "__main__":
    print("Please use scripts/investigate.py or scripts/app.py as the entry point.")
    sys.exit(0)
