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
import math
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

# ---------------------------------------------------------------------------
# Posterior uncertainty — the signal for depth >= 1 nodes
# ---------------------------------------------------------------------------
#
# WHY A SECOND SIGNAL EXISTS
#
# differential_breadth_entropy asks "how many distinct primary diagnoses could
# plausibly cause this finding?". That is the right question for a depth-0
# axiom, which IS a finding ("Potassium 5.6 mmol/L"). It is the wrong question
# for a depth >= 1 node, because by then the claim is already a full diagnostic
# hypothesis:
#
#     "Pulmonary embolism with associated pleuritic chest pain and cough..."
#
# The honest answer to "how many diagnoses explain that?" is one. So the signal
# collapses. Measured over 3,782 generated hypotheses in the 2026-08-30 run:
#
#     H = 0.10  ("one diagnosis")   64.3%
#     H >= 0.65 ("many")            29.6%
#     everything else                6.1%
#
# Two-thirds of nodes carry an identical score. The frontier is ordered by that
# score, synthesis ranks by it, and saturation is measured on it — so the
# entropy-guided traversal was, for most nodes, not guided by entropy. It was
# measuring *whether a claim is phrased as a diagnosis*, not how uncertain the
# engine is.
#
# For a hypothesis the quantity that actually varies is posterior uncertainty:
# given THIS patient, how confident are we that THIS is the primary diagnosis?
# That is a verbalized-confidence elicitation, and it varies continuously.
#
# NOT YET VALIDATED. This is a fix for a measured degeneracy, not a measured
# accuracy improvement. config.ENTROPY_SIGNAL switches back to "breadth".

HYPOTHESIS_CONFIDENCE_PROMPT = """\
You are a clinical diagnostician. Given a patient and a candidate diagnosis, \
state how confident you are that this is the patient's PRIMARY diagnosis.

=== PATIENT ===
{case_context}

=== CANDIDATE DIAGNOSIS ===
{claim}

Instructions:
- Answer with ONLY an integer from 0 to 100.
- 100 means certain this is the primary diagnosis; 0 means certainly not.
- Consider how well it explains ALL the findings, not just some of them.
- Judge this specific patient, not how common the disease is in general.

Confidence (0-100):"""


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

    def hypothesis_uncertainty(
        self,
        claim: str,
        case_context: str = "",
    ) -> Optional[float]:
        """Posterior uncertainty that `claim` is the primary diagnosis.

        Elicits a 0-100 confidence and maps it onto the same [0.05, 0.693]
        range the breadth signal uses, so every consumer (frontier ordering,
        synthesis ranking, saturation) keeps working unchanged.

        The map is the binary Shannon entropy of the stated confidence,
        rescaled: H(p) = -p·log2(p) - (1-p)·log2(1-p), which is 0 at p = 0 or
        1 and maximal at p = 0.5. A hypothesis the model is confident about
        (p -> 1) and one it has ruled out (p -> 0) are both *low* uncertainty;
        a coin-flip is high. That is the correct shape for an
        uncertainty-chasing frontier, and unlike the breadth signal it is
        continuous — its value moves with the model's actual belief instead of
        collapsing onto "one diagnosis".

        Args:
            claim: The candidate diagnosis.
            case_context: The patient. Without it the question degenerates back
                into a general-knowledge one, so callers should always pass it.

        Returns:
            Entropy in [0.05, 0.693], or None on total failure.
        """
        if not claim or claim.startswith("["):
            return _DEFAULT_HIGH

        key = f"hyp::{case_context[:200]}::{claim.strip().lower()}"
        if key in self._cache:
            return self._cache[key]

        result = self.score_hypothesis(claim, case_context)
        if result is None:
            return _DEFAULT_HIGH
        score, _confidence = result
        self._cache[key] = score
        return score

    def score_hypothesis(
        self,
        claim: str,
        case_context: str = "",
    ) -> Optional[tuple[float, float]]:
        """Return ``(entropy, confidence)`` for a candidate diagnosis.

        Both halves are needed by different consumers, and deriving one from
        the other is impossible: the entropy map is symmetric about p = 0.5, so
        H = 0.234 means either 5% or 95% confidence.

        That matters. The frontier wants *uncertainty* — chase the coin-flips.
        Synthesis wants *belief* — lead with what the engine thinks is true. If
        synthesis ranked by ascending entropy alone it would put a confidently
        ruled-out hypothesis (p -> 0, H -> 0.05) at the top of the
        differential, which is the opposite of the intent.

        Returns:
            ``(entropy in [0.05, 0.693], confidence in [0, 1])``, or None on
            total failure.
        """
        raw = self._query_confidence(claim, case_context)
        if raw is None:
            return None
        p = min(max(raw / 100.0, 1e-6), 1 - 1e-6)
        binary_entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))  # [0, 1]
        entropy = round(0.05 + binary_entropy * (_DEFAULT_HIGH - 0.05), 4)
        logger.debug(
            f"[EntropyEngine] '{claim[:50]}' → confidence={raw} → H={entropy:.3f}"
        )
        return entropy, round(p, 4)

    def _query_confidence(self, claim: str, case_context: str) -> Optional[int]:
        """Ask for a 0-100 confidence. Returns None on failure."""
        prompt = HYPOTHESIS_CONFIDENCE_PROMPT.format(
            case_context=(case_context or "(not provided)").strip()[:3000],
            claim=claim.strip(),
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 8},
        }
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/generate", json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()
                value = self._parse_confidence(raw)
                if value is not None:
                    return value
                logger.debug(
                    f"[EntropyEngine] Unparseable confidence {raw[:40]!r} "
                    f"(attempt {attempt + 1}/{self.retries})."
                )
            except requests.exceptions.Timeout:
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.warning(f"[EntropyEngine] Confidence query failed "
                               f"(attempt {attempt + 1}): {e}")
                time.sleep(2 * (attempt + 1))
        return None

    @staticmethod
    def _parse_confidence(raw: str) -> Optional[int]:
        """First integer in [0, 100] from the reply."""
        import re
        for match in re.finditer(r"\b(\d{1,3})\b", raw):
            value = int(match.group(1))
            if 0 <= value <= 100:
                return value
        return None

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
