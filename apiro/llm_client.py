"""
apiro/llm_client.py — OllamaLLMClient
======================================

Single shared implementation of the LLM client used by the CLI, the web
app, and the evaluation scripts. This used to be copy-pasted, nearly
identically, into apiro/run.py, scripts/investigate.py,
scripts/run_pmc_eval.py, and scripts/run_niah_eval.py — consolidated here
so a change (timeout, retry behaviour, model options) only has to happen
once.

Any object with a `.chat(prompt: str) -> str` method satisfies the
interface the rest of the codebase expects from an LLM client (see
apiro/graph/stub_llm.py for the test doubles). `.generate()` is the same
call under a different name, kept because some call sites historically
used it directly.
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


class OllamaLLMClient:
    """Thin wrapper around Ollama's /api/generate endpoint."""

    def __init__(self, url: str, model: str, temperature: float = 0.2, num_predict: int = 180, timeout: int = 90):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.num_predict},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def chat(self, prompt: str) -> str:
        return self.generate(prompt)
