"""
graph/contradiction.py (Signal Rewrite v2 — perf pass)
------------------------------------------------------
Detects logical contradictions between two clinical claims using a two-stage
fast-filter + LLM judge.

WHY TWO-STAGE:

Stage 1 — Fast filter (no LLM):
    NegEx + keyword antonym detection. Catches the 95% of pairs that are
    obviously NOT contradictions (different topics) and the obvious ones that
    ARE (explicit negation, drug conflicts). Zero network calls.
    → If fast-filter says CONTRADICTION or NEUTRAL with high confidence → done.
    → If fast-filter is ambiguous AND the claims are highly related → Stage 2.

Stage 2 — LLM judge (only when needed):
    "Can a single patient have both of these findings simultaneously?"
    Only called when Stage 1 flags a HIGH-SIMILARITY pair as potentially
    contradictory. Similarity is measured by shared medical keywords.

RESULT:
    In practice, the LLM judge fires on <5 pairs per traversal (pairs that
    share the same drug, body part, or disease entity but differ in assertion).
    Total LLM calls for contradiction: ~5 instead of ~90 per iteration.

PERFORMANCE NOTES (this pass):
    - Keyword extraction is memoized with a module-level functools.lru_cache
      (maxsize=8192) returning a frozenset. In a BFS batch the same claim text
      is compared against dozens of neighbors, so this collapses N regex+set
      builds per claim into a single cached one.
    - _shares_medical_entities now does a direct frozenset intersection using
      the cached keyword sets (with an isdisjoint fast path).
    - _cache_key uses direct integer tuple ordering for a cheap, stable key.

INTERFACE:
    Identical to the original ContradictionDetector. No callers need changes.
"""

import re
import logging
import time
import functools
from dataclasses import dataclass
from typing import Literal, Optional

import requests

from apiro.config import OLLAMA_BASE_URL, PRIMARY_MODEL

logger = logging.getLogger(__name__)

_CACHE_MAX = 4096

# Confidence assigned to a deterministic (keyword/negation) contradiction.
# This MUST sit strictly above CONTRADICTION_THRESHOLD_EF. It used to be
# exactly 0.92, and every consumer tests `score > 0.92`, so no keyword
# contradiction could ever fire: the deterministic half of the guardrail —
# the entire point of the Hybrid Apiro design — was dead code.
FAST_FILTER_CONTRADICTION_SCORE = 0.95

# Max concurrent LLM-judge calls issued by check_batch().
_BATCH_WORKERS = 8

_RAW_SEED_SUFFIX = re.compile(
    r"[—\-–?]\s*(symptom|lab|vital|imaging|history|medication|procedure)\s*$",
    re.IGNORECASE,
)

_ABSTRACTION_GROUPS: dict[str, str] = {
    "symptom":    "observation",
    "vital":      "observation",
    "lab":        "measurement",
    "imaging":    "measurement",
    "history":    "context",
    "medication": "context",
    "procedure":  "context",
}

NEGEX_PATTERNS = re.compile(
    r"\b("
    r"no\b|not\b|without|denies|denied|absent|absence of|"
    r"negative for|rules? out|ruled out|free of|"
    r"never|unlikely|cannot|can't|doesn't|does not|"
    r"no evidence of|no sign of|no history of"
    r")\b",
    re.IGNORECASE,
)

# Pairs of antonym keywords. If claim_a has a word from set A and claim_b has
# the corresponding word from set B (or vice versa), they are likely contradictory.
_ANTONYM_PAIRS: list[tuple[set, set]] = [
    ({"indicated", "safe", "beneficial", "recommended", "first-line"},
     {"contraindicated", "avoid", "dangerous", "do not use", "prohibited"}),
    ({"elevated", "increased", "high", "raised", "positive"},
     {"normal", "within normal limits", "absent", "negative", "low", "decreased"}),
    ({"present", "confirmed", "diagnosed", "detected", "demonstrates"},
     {"absent", "not present", "ruled out", "excluded", "no evidence"}),
    ({"fever", "pyrexia", "febrile", "temperature elevated"},
     {"afebrile", "no fever", "apyrexial", "temperature normal"}),
    ({"hypokalemia", "low potassium"}, {"hyperkalemia", "high potassium"}),
    ({"hypothyroidism", "low thyroid"}, {"hyperthyroidism", "high thyroid"}),
    ({"hypoglycemia", "low glucose", "low blood sugar"}, {"hyperglycemia", "high glucose", "high blood sugar"}),
    ({"anemia", "low hemoglobin"}, {"polycythemia", "high hemoglobin"}),
    ({"leukopenia", "low wbc"}, {"leukocytosis", "high wbc"}),
    ({"acidosis", "low ph"}, {"alkalosis", "high ph"}),
    ({"left", "left-sided"}, {"right", "right-sided"}),
]

_CLINICAL_STOPWORDS = {
    "patient", "clinical", "finding", "findings", "presents", "presenting", "presents with",
    "showing", "history", "result", "results", "normal", "abnormal", "elevated", "decreased",
    "increased", "reduced", "low", "high", "positive", "negative", "absent", "present", "with",
    "without", "have", "has", "had", "shows", "showed", "was", "were", "are", "is", "about",
    "above", "after", "before", "during", "been", "being", "cause", "caused", "primary", "secondary",
    "acute", "chronic", "mild", "severe", "moderate", "suspected", "probable", "likely", "unlikely",
    "sign", "signs", "symptom", "symptoms", "disease", "diseases", "disorder", "disorders", "condition",
    "conditions", "findings", "finding", "test", "tests", "examination", "exam", "report", "reported",
    "reveal", "reveals", "revealed", "show", "shows", "shown", "evidence", "level", "levels", "value",
    "values", "result-old"
}

# Medical abbreviations that survive the length filter in keyword extraction.
_ABBREVIATIONS = {"gist", "tsh", "acs", "pe", "bp", "egf", "crp", "nmo", "ms", "achr", "mg"}

# Precompiled word tokenizer for keyword extraction (avoids recompiling per call).
_KEYWORD_TOKENIZER = re.compile(r"\b[a-zA-Z\-]{2,}\b")

CONTRADICTION_JUDGE_PROMPT = """\
You are a clinical logician. Given two clinical findings about the same patient, determine if they logically EXCLUDE each other.

Finding A: {claim_a}
Finding B: {claim_b}

Question: Can a single patient simultaneously have BOTH of these findings?

Rules:
- Answer YES if both findings can coexist in the same patient (even if unrelated or from different organ systems).
- Answer NO only if one finding logically rules out the other (true medical contradiction).
- Do NOT answer NO just because the findings are from different organ systems.

Answer with YES or NO only."""


@functools.lru_cache(maxsize=8192)
def _get_keywords(text: str) -> frozenset[str]:
    """
    Extract the set of medical entity keywords from a claim.

    Memoized: in BFS batches the same claim text is compared against many
    neighbors, so caching collapses repeated regex + set allocations into a
    single computation per distinct claim. Returns a frozenset so the result
    is immutable and hashable (safe to share across callers and to cache).
    """
    kws: set[str] = set()
    for w in _KEYWORD_TOKENIZER.findall(text.lower()):
        if w in _ABBREVIATIONS:
            kws.add(w)
        elif len(w) >= 4 and w not in _CLINICAL_STOPWORDS:
            kws.add(w)
    return frozenset(kws)


@dataclass
class NLIResult:
    label: Literal["contradiction", "entailment", "neutral"]
    score: float
    negation_detected: bool


class ContradictionDetector:
    """
    Two-stage contradiction detection:
    1. Fast NegEx + antonym keyword filter (no LLM, instant)
    2. LLM judge — only for high-similarity pairs with negation present

    Interface identical to original ContradictionDetector.
    """

    def __init__(
        self,
        model: str = PRIMARY_MODEL,
        ollama_url: str = OLLAMA_BASE_URL,
        timeout: int = 30,
        retries: int = 2,
        scheduler=None,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.retries = retries
        self.scheduler = scheduler
        self._cache: dict[tuple[int, int], NLIResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._llm_calls = 0
        logger.info(f"[ContradictionDetector] Two-stage LLM judge: {model} @ {ollama_url}")

    @staticmethod
    def _seed_type(claim: str) -> str | None:
        m = _RAW_SEED_SUFFIX.search(claim)
        return m.group(1).lower() if m else None

    @classmethod
    def should_check(cls, claim_a: str, claim_b: str) -> bool:
        """Checks hypothesis-hypothesis, same-group seed pairs, or cross-group escalations."""
        type_a = cls._seed_type(claim_a)
        type_b = cls._seed_type(claim_b)

        if type_a is None and type_b is None:
            return True

        # Cross-group escalation: if the claims are related, check them
        if cls._shares_medical_entities(claim_a, claim_b):
            return True

        if type_a is not None and type_b is not None:
            group_a = _ABSTRACTION_GROUPS.get(type_a)
            group_b = _ABSTRACTION_GROUPS.get(type_b)
            return group_a is not None and group_a == group_b

        return False

    def _has_negation(self, text: str) -> bool:
        return bool(NEGEX_PATTERNS.search(text))

    def _cache_key(self, claim_a: str, claim_b: str) -> tuple[int, int]:
        # Fast integer tuple ordering: hash once, order the two ints so the
        # pair is symmetric (check(a, b) and check(b, a) share a slot).
        ha = hash(claim_a)
        hb = hash(claim_b)
        return (ha, hb) if ha <= hb else (hb, ha)

    def cache_info(self) -> dict:
        total = self._cache_hits + self._cache_misses
        rate = self._cache_hits / total if total else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "llm_calls": self._llm_calls,
            "size": len(self._cache),
            "hit_rate": round(rate, 3),
        }

    def check(self, claim_a: str, claim_b: str) -> NLIResult:
        """
        Two-stage contradiction check.
        Stage 1: fast keyword/negation filter — no LLM.
        Stage 2: LLM judge — only when Stage 1 is ambiguous AND claims share entities.
        """
        negation_detected = self._has_negation(claim_a) or self._has_negation(claim_b)

        # Cache lookup
        key = self._cache_key(claim_a, claim_b)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1

        result = self._fast_filter(claim_a, claim_b, negation_detected)
        if result is not None:
            # Fast filter gave a confident answer — no LLM needed
            self._store_cache(key, result)
            return result

        # Stage 2: LLM judge — only if claims share medical entities
        if self._shares_medical_entities(claim_a, claim_b):
            result = self._llm_judge(claim_a, claim_b, negation_detected)
            logger.debug(
                f"[ContradictionDetector] LLM judge: {result.label} | "
                f"'{claim_a[:35]}' vs '{claim_b[:35]}'"
            )
        else:
            # Different topics entirely — cannot be a true logical contradiction
            result = NLIResult(label="neutral", score=0.9, negation_detected=negation_detected)

        self._store_cache(key, result)
        return result

    def check_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """
        Check many pairs at once.

        Stage 1 (fast filter + cache) is resolved inline; only the pairs that
        genuinely need the LLM judge are dispatched, and those run concurrently.
        Previously this was a plain list comprehension, so a batch of 40 pairs
        that escalated 5 of them paid 5 sequential round-trips to Ollama on the
        critical path of every expansion.
        """
        if not pairs:
            return []

        results: list[Optional[NLIResult]] = [None] * len(pairs)
        pending: list[int] = []

        for i, (a, b) in enumerate(pairs):
            key = self._cache_key(a, b)
            if key in self._cache:
                self._cache_hits += 1
                results[i] = self._cache[key]
                continue
            negation = self._has_negation(a) or self._has_negation(b)
            fast = self._fast_filter(a, b, negation)
            if fast is not None:
                self._cache_misses += 1
                self._store_cache(key, fast)
                results[i] = fast
            elif not self._shares_medical_entities(a, b):
                # Different topics entirely — cannot be a true contradiction.
                self._cache_misses += 1
                neutral = NLIResult(label="neutral", score=0.9, negation_detected=negation)
                self._store_cache(key, neutral)
                results[i] = neutral
            else:
                pending.append(i)

        if pending:
            import concurrent.futures

            def _judge(idx: int) -> tuple[int, NLIResult]:
                a, b = pairs[idx]
                negation = self._has_negation(a) or self._has_negation(b)
                return idx, self._llm_judge(a, b, negation)

            workers = min(_BATCH_WORKERS, len(pending))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for idx, result in pool.map(_judge, pending):
                    self._cache_misses += 1
                    self._store_cache(self._cache_key(*pairs[idx]), result)
                    results[idx] = result

        # Length and order must mirror `pairs` exactly — callers zip the result
        # back against their own (new_node, existing) metadata list.
        return [
            r if r is not None else NLIResult(label="neutral", score=0.5, negation_detected=False)
            for r in results
        ]

    # ── Stage 1: Fast filter ──────────────────────────────────────────────────

    @staticmethod
    def _contains_term(text_lower: str, term: str) -> bool:
        """
        Whole-word/phrase containment.

        Plain `term in text` matched inside unrelated words and produced
        spurious contradictions: "low" fires on "blood flow" and "allow",
        "high" on "highly", "left" on "cleft", "mg" on "mg/dL". Since a
        fast-filter hit now carries enough score to soft-prune a node, the
        match has to respect word boundaries.
        """
        return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text_lower) is not None

    def _fast_filter(
        self, claim_a: str, claim_b: str, negation_detected: bool
    ) -> Optional[NLIResult]:
        """
        Returns NLIResult if we can decide confidently without an LLM call.
        Returns None if ambiguous (caller must escalate to LLM).
        """
        a, b = claim_a.lower(), claim_b.lower()

        # Check antonym keyword pairs
        for set_pos, set_neg in _ANTONYM_PAIRS:
            a_pos = any(self._contains_term(a, kw) for kw in set_pos)
            b_pos = any(self._contains_term(b, kw) for kw in set_pos)
            a_neg = any(self._contains_term(a, kw) for kw in set_neg)
            b_neg = any(self._contains_term(b, kw) for kw in set_neg)

            # One claim positive, the other negative on the same dimension
            if (a_pos and b_neg) or (a_neg and b_pos):
                # Only flag as contradiction if they share medical entities
                # (prevents "elevated troponin" vs "normal blood pressure" false hits)
                if self._shares_medical_entities(claim_a, claim_b):
                    return NLIResult(
                        label="contradiction",
                        score=FAST_FILTER_CONTRADICTION_SCORE,
                        negation_detected=negation_detected,
                    )

        # No strong keyword signal found — ambiguous, escalate to LLM if needed
        return None

    @classmethod
    def _shares_medical_entities(cls, claim_a: str, claim_b: str) -> bool:
        """
        True if the two claims share at least one medical entity keyword.

        Uses the memoized _get_keywords helper and a direct frozenset
        intersection. `isdisjoint` short-circuits on the first shared keyword
        and avoids materializing the full intersection set.
        """
        kws_a = _get_keywords(claim_a)
        kws_b = _get_keywords(claim_b)
        return not kws_a.isdisjoint(kws_b)

    # ── Stage 2: LLM judge ────────────────────────────────────────────────────

    def _llm_judge(
        self, claim_a: str, claim_b: str, negation_detected: bool
    ) -> NLIResult:
        """Ask the LLM if the two findings can coexist in the same patient."""
        self._llm_calls += 1
        prompt = CONTRADICTION_JUDGE_PROMPT.format(
            claim_a=claim_a.strip(),
            claim_b=claim_b.strip(),
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 5},
        }
        for attempt in range(self.retries):
            try:
                def request():
                    response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json=payload,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    return response

                resp = (
                    self.scheduler.call(
                        "contradiction", request, is_retry=attempt > 0
                    )
                    if self.scheduler is not None
                    else request()
                )
                raw = resp.json().get("response", "").strip().upper()
                first_word = re.split(r"\s|\.", raw)[0].strip(".,!?\"'")

                if first_word == "NO":
                    return NLIResult(
                        label="contradiction",
                        score=0.95,
                        negation_detected=negation_detected,
                    )
                else:
                    return NLIResult(
                        label="neutral",
                        score=0.95,
                        negation_detected=negation_detected,
                    )

            except requests.exceptions.Timeout:
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.warning(f"[ContradictionDetector] LLM judge failed (attempt {attempt+1}): {e}")
                time.sleep(2 * (attempt + 1))

        # Total failure → safe default: do not prune
        logger.error("[ContradictionDetector] All LLM attempts failed — defaulting to neutral.")
        return NLIResult(label="neutral", score=0.5, negation_detected=negation_detected)

    def _store_cache(self, key: tuple, result: NLIResult) -> None:
        if len(self._cache) >= _CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
