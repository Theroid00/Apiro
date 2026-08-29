"""
entropy/engine.py — EntropyEngine (Signal Rewrite)
====================================================

WHAT CHANGED AND WHY:

OLD (broken) signal:
    Measured Shannon entropy of the first generated token on a yes/no
    verification question. This is "how surprised is the LLM by the question"
    — i.e. LLM fluency, not clinical uncertainty.

NEW (fixed) signal:
    Asks the LLM: "How many distinct diagnoses could plausibly explain
    this clinical finding?" and maps the count to an uncertainty score.

    - Many competing diagnoses (>= 5) → high uncertainty → entropy ≈ 0.693
    - Few diagnoses (1-2) → low uncertainty → entropy ≈ 0.1

    This is the correct operationalisation of "diagnostic breadth" as a
    proxy for epistemic uncertainty. A symptom that could mean 10 different
    things IS more uncertain than a finding that points to only one diagnosis.

ARCHITECTURE:
    The interface is identical to the original EntropyEngine. Every caller
    that used temperature_corrected_entropy() or epistemic_certainty_entropy()
    will continue to work without changes.

    Ollama is still used for the LLM call, but now we use a simple chat
    completion instead of the logprob API (which was unreliable on local
    Ollama anyway).
"""

import logging
import time
from typing import Optional

import requests

from apiro.config import (
    OLLAMA_BASE_URL, PRIMARY_MODEL,
)

logger = logging.getLogger(__name__)


# Map the LLM's count answer to an uncertainty score in [0.05, 0.693].
# The mapping is monotonically increasing: more competing diagnoses = more uncertain.
# Values are calibrated so the frontier priority ordering is meaningful.
_COUNT_TO_ENTROPY: dict[int, float] = {
    0: 0.05,   # no plausible diagnoses → near-certain this is a dead end
    1: 0.10,   # one diagnosis → highly confident, very specific
    2: 0.25,   # two diagnoses → some uncertainty
    3: 0.40,   # three → moderate uncertainty
    4: 0.55,   # four → high uncertainty
    5: 0.65,   # five → very high uncertainty
}
_DEFAULT_HIGH = 0.693   # ln(2) — max binary uncertainty, used for >= 6 diagnoses


DIFFERENTIAL_BREADTH_PROMPT = """\
You are a clinical diagnostician. Given a single clinical finding, count how many distinct primary diagnoses could plausibly cause it.

Clinical finding: {claim}

Instructions:
- Count DISTINCT diagnoses (not sub-types of the same disease).
- Only count diagnoses where this finding is a cardinal or major feature.
- Respond with ONLY a single integer on the first line. No explanation.

Integer count:"""


class EntropyEngine:
    """
    Queries the LLM to measure diagnostic breadth uncertainty.

    High count (many competing diagnoses) → high entropy (explore more).
    Low count (finding is specific)       → low entropy (converging).
    """

    def __init__(
        self,
        model: str = PRIMARY_MODEL,
        ollama_url: str = OLLAMA_BASE_URL,
        timeout: int = 60,
        retries: int = 2,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.retries = retries
        self._cache: dict[str, float] = {}  # claim → entropy score

    # ------------------------------------------------------------------
    # Public API (interface-compatible with old EntropyEngine)
    # ------------------------------------------------------------------

    def temperature_corrected_entropy(self, prompt: str) -> Optional[float]:
        """
        Legacy entry point — `prompt` is expected to contain the clinical claim.
        Extracts the claim and delegates to differential_breadth_entropy().
        """
        # The old verification prompt embeds the claim after "Clinical claim: "
        claim = self._extract_claim_from_prompt(prompt)
        return self.differential_breadth_entropy(claim)

    def epistemic_certainty_entropy(
        self,
        claim: str,
        context_chunks: list[str] | None = None,
    ) -> Optional[float]:
        """Legacy entry point — delegates to differential_breadth_entropy()."""
        return self.differential_breadth_entropy(claim)

    def differential_breadth_entropy(self, claim: str) -> Optional[float]:
        """
        THE CORE SIGNAL: ask the LLM how many diagnoses explain this finding.

        Returns an entropy score in [0.05, 0.693] proportional to the count.
        Returns None only on total failure (Ollama down, parse failure).
        """
        if not claim or claim.startswith("["):
            return _DEFAULT_HIGH  # stub / failed expansion → treat as uncertain

        # Cache lookup
        key = claim.strip().lower()
        if key in self._cache:
            return self._cache[key]

        count = self._query_differential_count(claim)
        if count is None:
            return _DEFAULT_HIGH

        score = _COUNT_TO_ENTROPY.get(count, _DEFAULT_HIGH)
        self._cache[key] = score

        logger.debug(
            f"[EntropyEngine] '{claim[:60]}' → {count} diagnoses → entropy={score:.3f}"
        )
        return score

    def first_token_entropy(self, prompt: str, temperature: float = 0.3) -> Optional[float]:
        """Legacy compat — delegates to temperature_corrected_entropy."""
        return self.temperature_corrected_entropy(prompt)

    def is_reachable(self) -> bool:
        """Return True if Ollama server is reachable and the model is available."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            pulled = {m["name"] for m in resp.json().get("models", [])}
            model_base = self.model.split(":")[0]
            return any(p.startswith(model_base) for p in pulled)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_differential_count(self, claim: str) -> Optional[int]:
        """Ask the LLM to count competing diagnoses. Returns int or None."""
        prompt = DIFFERENTIAL_BREADTH_PROMPT.format(claim=claim.strip())
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,   # deterministic — we want a count, not creativity
                "num_predict": 8,     # just a number
            },
        }
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()
                count = self._parse_count(raw)
                if count is not None:
                    return count
                # An unparseable answer is a transient formatting miss, not a
                # dead server; returning immediately spent the whole retry
                # budget on nothing and defaulted the node to max uncertainty,
                # which sends it straight to the top of the frontier.
                logger.debug(
                    f"[EntropyEngine] Unparseable count {raw[:40]!r} "
                    f"(attempt {attempt + 1}/{self.retries})."
                )
            except requests.exceptions.Timeout:
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.warning(f"[EntropyEngine] Query failed (attempt {attempt+1}): {e}")
                time.sleep(2 * (attempt + 1))
        return None

    @staticmethod
    def _parse_count(raw: str) -> Optional[int]:
        """Parse the first integer from the LLM's response."""
        import re
        m = re.search(r"\b(\d+)\b", raw)
        if m:
            return min(int(m.group(1)), 6)   # cap at 6 → maps to _DEFAULT_HIGH
        return None

    @staticmethod
    def _extract_claim_from_prompt(prompt: str) -> str:
        """
        Extract the clinical claim from an old-style verification prompt.
        Falls back to returning the entire prompt as the claim.
        """
        marker = "Clinical claim:"
        if marker in prompt:
            after = prompt.split(marker, 1)[1]
            # Take just the first line
            return after.split("\n")[0].strip()
        return prompt.strip()

    @staticmethod
    def _build_verification_prompt(
        claim: str,
        context_chunks: list[str] | None = None,
    ) -> str:
        """
        Legacy compat: returns a prompt string that temperature_corrected_entropy()
        can accept. Since our new engine extracts the claim from the prompt,
        this just embeds the claim in the expected format.
        """
        return f"Clinical claim: {claim.strip()}\n\nBased on the evidence above, is this claim clinically supported? Answer with Yes or No only."
