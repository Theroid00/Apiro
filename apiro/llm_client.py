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

    def __init__(
        self,
        url: str,
        model: str,
        temperature: float = 0.2,
        seed: int = 7,
        num_predict: int = 180,
        json_num_predict: int = 768,
        timeout: int = 90,
        scheduler=None,
    ):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.num_predict = num_predict
        self.json_num_predict = json_num_predict
        self.timeout = timeout
        self.scheduler = scheduler

    def _generate(self, prompt: str, *, json_mode: bool = False) -> str:
        def request():
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "num_predict": (
                        self.json_num_predict if json_mode else self.num_predict
                    ),
                },
            }
            if json_mode:
                payload["format"] = "json"
            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response

        resp = (
            self.scheduler.call("generation", request)
            if self.scheduler is not None
            else request()
        )
        return resp.json().get("response", "")

    def generate(self, prompt: str) -> str:
        return self._generate(prompt)

    def generate_json(self, prompt: str) -> str:
        return self._generate(prompt, json_mode=True)

    def chat(self, prompt: str) -> str:
        return self.generate(prompt)
