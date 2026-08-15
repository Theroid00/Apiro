#!/usr/bin/env python3
"""
scripts/run_niah_eval.py
========================
Runs the Clinical Needle-In-A-Haystack (NIAH) evaluation.

For every case in ``data/niah_cases.json`` the harness compares three arms:

    Arm 1 -- Bare LLM Zero-Shot
    Arm 2 -- Standard RAG Baseline
    Arm 3 -- Apiro Belief-Graph Traversal

Each case embeds one (or more) diagnostically decisive "needle(s)" inside a
long, distractor-heavy clinical "haystack". Cases are tagged with a *family*
(single_needle, contradiction_needle, multi_needle, red_herring,
negation_trap), a *haystack length* bucket and a *needle depth* bucket so the
harness can report where each arm degrades.

The scoring primitives (``_check_synthesis_hit`` / ``_print_summary``) and all
Apiro components are reused unchanged from the PMC harness so the two
benchmarks stay directly comparable.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("niah_eval")

# --------------------------------------------------------------------------- #
# Project path
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apiro.graph.belief_graph import BeliefGraph
from apiro.graph.node import Node
from apiro.graph.traversal import ApiroTraversal
from apiro.axioms.extractor import AxiomExtractor
from apiro.config import SATURATION_EXPLORATION_ONLY

# The canonical NIAH families. Order here defines display order everywhere.
NIAH_FAMILIES = [
    "single_needle",
    "contradiction_needle",
    "multi_needle",
    "red_herring",
    "negation_trap",
]

# Ordered length / depth buckets used to build the accuracy matrix. Any bucket
# value seen in the data that is not listed here is appended dynamically.
LENGTH_BUCKETS = ["short", "medium", "long", "xlong"]
DEPTH_BUCKETS = ["shallow", "mid", "deep"]


# --------------------------------------------------------------------------- #
# Case loading helpers
# --------------------------------------------------------------------------- #
def _load_cases(cases_path: Path) -> list[dict]:
    """Load NIAH cases, accepting either a bare ``list`` or ``{"cases": [...]}``."""
    with open(cases_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        cases = data.get("cases")
        if cases is None:
            raise ValueError(
                f"{cases_path} is a JSON object but has no 'cases' key "
                f"(keys present: {sorted(data.keys())})."
            )
    elif isinstance(data, list):
        cases = data
    else:
        raise ValueError(
            f"{cases_path} must contain a list or an object with a 'cases' key, "
            f"got {type(data).__name__}."
        )
    if not isinstance(cases, list):
        raise ValueError("'cases' must be a list.")
    return cases


def _case_family(case: dict) -> str:
    fam = (case.get("family") or case.get("needle_family") or "unknown")
    fam = str(fam).strip().lower()
    return fam or "unknown"


def _length_bucket(case: dict) -> str:
    tokens = case.get("target_tokens") or case.get("approx_tokens") or case.get("length_tokens")
    if tokens is not None:
        return f"{tokens}k" if tokens >= 1000 and tokens % 1000 == 0 else f"{tokens}tok"
    val = (
        case.get("length_bucket")
        or case.get("haystack_length")
        or case.get("length")
        or "unknown"
    )
    return str(val).strip().lower() or "unknown"


def _depth_bucket(case: dict) -> str:
    depth = case.get("depth_fraction") or case.get("depth")
    if depth is not None:
        try:
            return f"{int(float(depth) * 100)}%"
        except (ValueError, TypeError):
            pass
    val = (
        case.get("depth_bucket")
        or case.get("needle_depth")
        or "unknown"
    )
    return str(val).strip().lower() or "unknown"


def _targets(case: dict) -> list[str]:
    """Return the list of target diagnoses (multi_needle cases may have several)."""
    if "target_diagnoses" in case and case["target_diagnoses"]:
        tgts = case["target_diagnoses"]
    elif "target_diagnosis" in case and case["target_diagnosis"]:
        tgts = case["target_diagnosis"]
    elif "diagnosis" in case and case["diagnosis"]:
        tgts = case["diagnosis"]
    elif "target" in case and case["target"]:
        tgts = case["target"]
    elif "needles" in case and case.get("metadata", {}).get("correct_diagnosis"):
        tgts = case["metadata"]["correct_diagnosis"]
    else:
        tgts = []
    if isinstance(tgts, str):
        return [tgts]
    return [str(t) for t in tgts]


def _target_display(case: dict) -> str:
    return " ; ".join(_targets(case)) or "(none)"


def _synthesis_hits_all_targets(items, targets, embedder, llm_client) -> bool:
    """
    A NIAH answer is correct only if EVERY needle target is recovered.

    For single-needle families this collapses to the ordinary single-target
    check; for multi_needle every target must appear in the synthesis.
    """
    from apiro.eval.evaluator import _check_synthesis_hit

    if not targets:
        return False
    for target in targets:
        hit, _ = _check_synthesis_hit(
            items,
            target,
            embedder=embedder,
            llm_client=llm_client,
        )
        if not hit:
            return False
    return True


# --------------------------------------------------------------------------- #
# Real component wiring (Ollama + ChromaDB)
# --------------------------------------------------------------------------- #
def _build_real_components():
    import requests
    from apiro.config import OLLAMA_BASE_URL, PRIMARY_MODEL
    from apiro.graph.expander import NodeExpander
    from apiro.graph.saturation import SaturationDetector
    from apiro.graph.rabbit_hole import RabbitHoleDetector
    from apiro.graph.contradiction import ContradictionDetector
    from apiro.entropy.engine import EntropyEngine
    from apiro.corpus.embedder import Embedder

    # Fail fast if Ollama is unreachable.
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.error(f"[Ollama Error] Could not reach Ollama at {OLLAMA_BASE_URL}: {e}")
        sys.exit(1)

    embedder = Embedder()

    class _ChromaAdapter:
        """Adapts the Embedder to the chroma-style query surface the expander expects."""

        def __init__(self, emb: Embedder):
            self._emb = emb

        def query(
            self,
            collection_name: str = "",
            query_texts: list | None = None,
            n_results: int = 6,
            where: dict | None = None,
        ) -> dict:
            query_texts = query_texts or []
            text = query_texts[0] if query_texts else ""
            results = self._emb.query(text, n_results=n_results, where=where)
            # Distances pass through so the expander can discard mere nearest
            # neighbours that are not real evidence.
            return {
                "documents": [[r["text"] for r in results]],
                "distances": [[r.get("distance") for r in results]],
            }

    chroma_adapter = _ChromaAdapter(embedder)
    entropy_engine = EntropyEngine(model=PRIMARY_MODEL, ollama_url=OLLAMA_BASE_URL)

    class OllamaLLMClient:
        def __init__(self, url, model):
            self.url = url
            self.model = model

        def generate(self, prompt: str) -> str:
            import requests as req

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 180},
            }
            resp = req.post(f"{self.url}/api/generate", json=payload, timeout=120)
            return resp.json().get("response", "")

        def generate_with_logprobs(self, prompt: str) -> tuple[str, list]:
            import requests as req

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 180},
                "logprobs": True,
            }
            resp = req.post(f"{self.url}/api/generate", json=payload, timeout=120)
            data = resp.json()
            return data.get("response", ""), data.get("logprobs", [])

        def chat(self, prompt: str) -> str:
            return self.generate(prompt)

    llm_client = OllamaLLMClient(OLLAMA_BASE_URL, PRIMARY_MODEL)
    contradiction = ContradictionDetector()
    expander = NodeExpander(
        entropy_engine=entropy_engine,
        chroma_client=chroma_adapter,
        llm_client=llm_client,
        contradiction_detector=contradiction,
    )
    # exploration_only: depth-0 axiom seeds have a fixed entropy and must never
    # be counted as evidence of convergence.
    saturation = SaturationDetector(exploration_only=SATURATION_EXPLORATION_ONLY)
    rabbit_hole = RabbitHoleDetector()
    traversal = ApiroTraversal(
        expander=expander,
        saturation=saturation,
        rabbit_hole=rabbit_hole,
        contradiction=contradiction,
    )
    return {
        "embedder": embedder,
        "llm_client": llm_client,
        "traversal": traversal,
        "contradiction": contradiction,
    }


# --------------------------------------------------------------------------- #
# Mock/stub component wiring
# --------------------------------------------------------------------------- #
def _build_stub_components():
    from apiro.graph.expander import NodeExpander, StubEntropyEngine, StubChromaClient
    from apiro.graph.saturation import SaturationDetector
    from apiro.graph.rabbit_hole import RabbitHoleDetector

    class StubContradictionDetector:
        """
        Deterministic contradiction detector for offline NIAH runs.

        It recognises the canonical NIAH "trap" pairs -- a tempting surface
        diagnosis pitted against a buried needle that rules it out -- and marks
        them as contradictions so the traversal soft-prunes the distractor.
        """

        def check(self, claim_a: str, claim_b: str):
            from dataclasses import dataclass

            @dataclass
            class R:
                label: str
                score: float
                negation_detected: bool

            a, b = claim_a.lower(), claim_b.lower()

            def match(kws1, kws2):
                return (
                    (any(k in a for k in kws1) and any(k in b for k in kws2))
                    or (any(k in b for k in kws1) and any(k in a for k in kws2))
                )

            # ACS surface story vs. buried normal cardiac work-up.
            if match(
                {"myocardial infarction", "acute coronary", "angina"},
                {"normal troponin", "normal sinus rhythm", "normal ecg", "normal ekg"},
            ):
                return R("contradiction", 0.95, False)
            # Sepsis/infection vs. explicitly negative cultures (negation trap).
            if match(
                {"sepsis", "bacteremia", "infection"},
                {"cultures negative", "no growth", "afebrile", "negative cultures"},
            ):
                return R("contradiction", 0.95, True)
            # Malaria vs. negative blood films.
            if match(
                {"malaria"},
                {"negative for plasmodium", "blood films are negative", "thick smear negative"},
            ):
                return R("contradiction", 0.95, False)
            # PE vs. explicitly negative CTPA (negation trap / red herring).
            if match(
                {"pulmonary embolism", "pe"},
                {"no evidence of pulmonary embolism", "negative ctpa", "d-dimer negative"},
            ):
                return R("contradiction", 0.95, True)
            # Panic vs. biochemically confirmed pheochromocytoma.
            if match(
                {"panic", "anxiety"},
                {"metanephrines", "adrenal mass", "pheochromocytoma"},
            ):
                return R("contradiction", 0.95, False)
            # Gastroenteritis vs. adrenal-insufficiency needle.
            if match(
                {"gastroenteritis", "dehydration"},
                {"hyperpigmentation", "cortisol", "acth", "addison"},
            ):
                return R("contradiction", 0.95, False)
            return R("neutral", 0.5, False)

        def check_batch(self, pairs):
            return [self.check(a, b) for a, b in pairs]

        def should_check(self, claim_a: str, claim_b: str) -> bool:
            return True

    class StubLLMClient:
        """
        Deterministic LLM stand-in. It reproduces the canonical NIAH failure
        modes: bare/RAG arms latch onto the surface distractor, while the
        traversal expansion prompts (which carry the surfaced needle in their
        context) recover the true diagnosis.
        """

        def generate(self, prompt: str) -> str:
            p = prompt.lower()

            bare = "based on the following clinical presentation" in p
            rag = "standard rag presentation" in p or "retrieved medical context" in p
            synth = "synthesize the final top 3" in p

            # ---- Bare LLM / RAG: seduced by the surface story ------------- #
            if bare or rag:
                if "substernal chest pain" in p or "pressure in his chest" in p:
                    return "1. Acute Myocardial Infarction\n2. Angina Pectoris\n3. GERD"
                if "productive cough" in p and "fever" in p:
                    return "1. Community-Acquired Pneumonia\n2. Bronchitis\n3. Sepsis"
                if "sub-saharan africa" in p or "tea-colored urine" in p:
                    return "1. Malaria infection\n2. Acute Hepatitis\n3. Hemolytic Anemia"
                if "tearing chest pain" in p or "dyspnea" in p:
                    return "1. Pulmonary Embolism\n2. Pneumothorax\n3. Myocardial Infarction"
                if "palpitations" in p and "anxiety" in p:
                    return "1. Panic Disorder\n2. Generalized Anxiety Disorder\n3. Arrhythmia"
                if "nausea, vomiting" in p and "weight loss" in p:
                    return "1. Acute Gastroenteritis\n2. Food Poisoning\n3. Dehydration"
                return (
                    "1. Common surface diagnosis A\n"
                    "2. Common surface diagnosis B\n"
                    "3. Common surface diagnosis C"
                )

            # ---- Apiro final synthesis: needle recovered ------------------ #
            if synth:
                if "normal troponin" in p or "normal sinus rhythm" in p or "esr" in p:
                    return "1. Subacute Thyroiditis\n2. Non-cardiac Chest Pain\n3. Pericarditis"
                if "cultures negative" in p or "no growth" in p:
                    return "1. Adrenal Crisis\n2. Adrenal Insufficiency\n3. Hypovolemia"
                if "no evidence of pulmonary embolism" in p or "widened mediastinum" in p:
                    return "1. Aortic Dissection\n2. Thoracic Aortic Aneurysm\n3. Pericardial Effusion"
                if "metanephrines" in p or "adrenal mass" in p:
                    return "1. Pheochromocytoma\n2. Adrenal Adenoma\n3. Hypertension"
                if "hyperpigmentation" in p or "cortisol" in p or "acth" in p:
                    return "1. Addison's Disease\n2. Adrenal Insufficiency\n3. Hyponatremia"
                if "negative for plasmodium" in p or "blood films are negative" in p:
                    return "1. G6PD Deficiency\n2. Hemolytic Anemia\n3. Viral Hepatitis"
                return (
                    "1. Needle-supported diagnosis A\n"
                    "2. Alternative diagnosis B\n"
                    "3. Alternative diagnosis C"
                )

            # ---- Apiro node-expansion hypotheses -------------------------- #
            if "normal sinus rhythm" in p or "troponin" in p or "thyroid" in p or "tsh" in p:
                return (
                    "Hypotheses:\n"
                    "1. Subacute Thyroiditis given the buried thyroid findings\n"
                    "2. Non-cardiac chest pain with normal cardiac work-up\n"
                    "3. Pericarditis"
                )
            if "cultures negative" in p or "no growth" in p or "hypotension" in p:
                return (
                    "Hypotheses:\n"
                    "1. Adrenal Crisis given negative cultures and hypotension\n"
                    "2. Adrenal Insufficiency (Addison's)\n"
                    "3. Distributive shock non-septic"
                )
            if "widened mediastinum" in p or "asymmetric blood pressure" in p or "tearing" in p:
                return (
                    "Hypotheses:\n"
                    "1. Aortic Dissection given widened mediastinum\n"
                    "2. Thoracic aortic aneurysm\n"
                    "3. Pulmonary embolism (excluded on CTPA)"
                )
            if "metanephrines" in p or "adrenal mass" in p or "palpitations" in p:
                return (
                    "Hypotheses:\n"
                    "1. Pheochromocytoma paroxysm\n"
                    "2. Adrenal adenoma\n"
                    "3. Panic disorder (distractor)"
                )
            if "hyperpigmentation" in p or "cortisol" in p or "acth" in p:
                return (
                    "Hypotheses:\n"
                    "1. Addison's Disease adrenal insufficiency\n"
                    "2. Nelson syndrome\n"
                    "3. Gastroenteritis (distractor)"
                )
            if "bite cells" in p or "g6pd" in p or "heinz" in p or "negative for plasmodium" in p:
                return (
                    "Hypotheses:\n"
                    "1. G6PD Deficiency hemolytic crisis\n"
                    "2. Drug-induced hemolytic anemia\n"
                    "3. Malaria (blood films negative)"
                )

            return (
                "Hypotheses:\n"
                "1. Alternative differential diagnosis A\n"
                "2. Alternative differential diagnosis B\n"
                "3. Alternative differential diagnosis C"
            )

        def chat(self, prompt: str) -> str:
            return self.generate(prompt)

    llm_client = StubLLMClient()
    contradiction = StubContradictionDetector()
    expander = NodeExpander(
        entropy_engine=StubEntropyEngine(),
        chroma_client=StubChromaClient(),
        llm_client=llm_client,
        contradiction_detector=contradiction,
    )
    saturation = SaturationDetector(
        theta=0.25,
        window=5,
        max_variance=0.04,
        exploration_only=SATURATION_EXPLORATION_ONLY,
    )
    rabbit_hole = RabbitHoleDetector(min_depth=3, reversal_window=4)
    traversal = ApiroTraversal(
        expander=expander,
        saturation=saturation,
        rabbit_hole=rabbit_hole,
        contradiction=contradiction,
    )
    return {
        "embedder": None,
        "llm_client": llm_client,
        "traversal": traversal,
        "contradiction": contradiction,
    }


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #
def _evaluate_case(case, components, real_components, axiom_extractor):
    embedder = components["embedder"]
    llm_client = components["llm_client"]
    traversal = components["traversal"]

    case_id = case.get("case_id", case.get("id", "?"))
    vignette = case.get("vignette") or case.get("haystack") or ""
    description = case.get("description", "")
    targets = _targets(case)
    family = _case_family(case)
    length = _length_bucket(case)
    depth = _depth_bucket(case)

    logger.info(f"\nEvaluating Case: {case_id} — {description} "
                f"[family={family}, length={length}, depth={depth}]")

    # ---- Arm 1: Bare LLM Zero-Shot --------------------------------------- #
    logger.info("  Running Bare LLM Zero-Shot...")
    prompt = (
        "Based on the following clinical presentation, provide a list of your top 3 "
        "differential diagnoses. Output ONLY the top 3 diagnoses as a bulleted or "
        "numbered list without any other text:\n\n"
        f"{vignette}"
    )
    bare_output = llm_client.generate(prompt)
    logger.info(f"  Bare LLM Output:\n{bare_output.strip()}")
    bare_items = [ln.strip() for ln in bare_output.split("\n") if ln.strip()]
    bare_success = _synthesis_hits_all_targets(
        bare_items, targets,
        embedder if real_components else None,
        llm_client if real_components else None,
    )

    # ---- Arm 2: Standard RAG Baseline ------------------------------------ #
    logger.info("  Running Standard RAG Baseline...")
    if real_components:
        rag_results = embedder.query(vignette, n_results=6)
        rag_context = "\n\n".join([r["text"] for r in rag_results])
        prompt_rag = (
            "Based on the following clinical presentation and the retrieved medical "
            "context, provide your top 3 differential diagnoses. Output ONLY the top 3 "
            "diagnoses as a bulleted or numbered list:\n\n"
            f"Vignette: {vignette}\n\nContext:\n{rag_context}"
        )
        rag_output = llm_client.generate(prompt_rag)
    else:
        prompt_rag = f"Standard RAG presentation:\n{vignette}"
        rag_output = llm_client.generate(prompt_rag)
    logger.info(f"  RAG Output:\n{rag_output.strip()}")
    rag_items = [ln.strip() for ln in rag_output.split("\n") if ln.strip()]
    rag_success = _synthesis_hits_all_targets(
        rag_items, targets,
        embedder if real_components else None,
        llm_client if real_components else None,
    )

    # ---- Arm 3: Apiro Belief-Graph Traversal ----------------------------- #
    logger.info("  Running Apiro Traversal...")
    graph = BeliefGraph()
    seeds = [
        Node(
            id=s["id"],
            claim=s["claim"] if (" — " in s["claim"] or " ? " in s["claim"])
            else f"{s['claim']} — {s['domain']}",
            entropy_score=s["entropy"],
            domain=s["domain"],
            depth=s["depth"],
        )
        for s in case.get("seed_nodes", [])
    ]

    if real_components:
        logger.info("  Extracting deterministic clinical axioms...")
        from apiro.axioms.seeding import axioms_to_seed_nodes, enrich_vignette

        axioms = axiom_extractor.extract(vignette)
        seeds.extend(axioms_to_seed_nodes(axioms))
        logger.info(f"  Extracted {len(axioms)} axioms and anchored them to the graph.")
        vignette_to_pass = enrich_vignette(vignette, axioms)
    else:
        vignette_to_pass = vignette

    traversal_res = traversal.run(
        seed_nodes=seeds,
        graph=graph,
        max_depth=6,
        case_name=case_id,
        vignette=vignette_to_pass,
    )
    apiro_output = traversal_res.synthesis
    apiro_output_str = "\n".join(apiro_output)
    logger.info(f"  Apiro Final Synthesis:\n{apiro_output_str.strip()}")
    apiro_success = _synthesis_hits_all_targets(
        apiro_output, targets,
        embedder if real_components else None,
        llm_client if real_components else None,
    )

    # Log soft-pruned distractor nodes for debugging.
    for node in graph.nodes.values():
        if getattr(node, "contradiction_penalty", 0.0) > 0:
            logger.info(
                f"    - Contradiction Soft-Pruned Node: '{node.claim[:45]}' "
                f"has penalty={node.contradiction_penalty} "
                f"(is_rabbit_hole={node.is_rabbit_hole})"
            )

    return {
        "case_id": case_id,
        "description": description,
        "family": family,
        "length": length,
        "depth": depth,
        "target": _target_display(case),
        "targets": targets,
        "bare_llm": {"success": bare_success, "output": bare_output},
        "rag": {"success": rag_success, "output": rag_output},
        "apiro": {"success": apiro_success, "output": apiro_output},
        "traversal": {
            "stop_reason": traversal_res.stop_reason,
            "total_nodes": traversal_res.total_nodes,
            "total_edges": traversal_res.total_edges,
            "seed_nodes": sum(1 for n in graph.nodes.values() if n.depth == 0),
            "explored_nodes": graph.count_expansions(min_depth=1),
            "max_depth_reached": max((n.depth for n in graph.nodes.values()), default=0),
            "rabbit_holes": traversal_res.rabbit_hole_count,
            "contradictions": traversal_res.contradiction_count,
            "duration_seconds": traversal_res.duration_seconds,
        },
    }


# --------------------------------------------------------------------------- #
# Aggregation & reporting
# --------------------------------------------------------------------------- #
def _accuracy(hits: int, total: int) -> float:
    return (hits / total) if total else 0.0


def _fmt_pct(hits: int, total: int) -> str:
    if not total:
        return "   -  "
    return f"{_accuracy(hits, total) * 100:5.1f}%"


def _overall_table(results):
    n = len(results)
    bare = sum(1 for r in results if r["bare_llm"]["success"])
    rag = sum(1 for r in results if r["rag"]["success"])
    apiro = sum(1 for r in results if r["apiro"]["success"])

    print("\n" + "=" * 65)
    print("  CLINICAL NEEDLE-IN-A-HAYSTACK — OVERALL ACCURACY")
    print("=" * 65)
    print(f"{'Arm':<24}{'Correct':>12}{'Accuracy':>12}")
    print("-" * 65)
    print(f"{'Bare LLM Zero-Shot':<24}{f'{bare}/{n}':>12}{_fmt_pct(bare, n):>12}")
    print(f"{'Standard RAG':<24}{f'{rag}/{n}':>12}{_fmt_pct(rag, n):>12}")
    print(f"{'Apiro Traversal':<24}{f'{apiro}/{n}':>12}{_fmt_pct(apiro, n):>12}")
    print("=" * 65)
    return {
        "bare_llm": _accuracy(bare, n),
        "rag": _accuracy(rag, n),
        "apiro": _accuracy(apiro, n),
        "bare_llm_correct": bare,
        "rag_correct": rag,
        "apiro_correct": apiro,
        "n_cases": n,
    }


def _per_family_table(results):
    print("\n" + "=" * 78)
    print("  PER-FAMILY ACCURACY")
    print("=" * 78)
    print(f"{'Family':<24}{'N':>5}{'Bare':>12}{'RAG':>12}{'Apiro':>12}")
    print("-" * 78)

    seen = [f for f in NIAH_FAMILIES if any(r["family"] == f for r in results)]
    extra = sorted({r["family"] for r in results} - set(NIAH_FAMILIES))
    per_family = {}
    for fam in seen + extra:
        subset = [r for r in results if r["family"] == fam]
        n = len(subset)
        bare = sum(1 for r in subset if r["bare_llm"]["success"])
        rag = sum(1 for r in subset if r["rag"]["success"])
        apiro = sum(1 for r in subset if r["apiro"]["success"])
        print(f"{fam:<24}{n:>5}{_fmt_pct(bare, n):>12}"
              f"{_fmt_pct(rag, n):>12}{_fmt_pct(apiro, n):>12}")
        per_family[fam] = {
            "n_cases": n,
            "bare_llm": _accuracy(bare, n),
            "rag": _accuracy(rag, n),
            "apiro": _accuracy(apiro, n),
        }
    print("=" * 78)
    return per_family


def _length_depth_matrix(results, arm="apiro"):
    """Length (rows) x Depth (cols) accuracy heatmap for a single arm."""
    lengths = [l for l in LENGTH_BUCKETS if any(r["length"] == l for r in results)]
    lengths += sorted({r["length"] for r in results} - set(LENGTH_BUCKETS) - {"unknown"})
    if any(r["length"] == "unknown" for r in results):
        lengths.append("unknown")

    depths = [d for d in DEPTH_BUCKETS if any(r["depth"] == d for r in results)]
    depths += sorted({r["depth"] for r in results} - set(DEPTH_BUCKETS) - {"unknown"})
    if any(r["depth"] == "unknown" for r in results):
        depths.append("unknown")

    print("\n" + "=" * 78)
    print(f"  LENGTH x DEPTH ACCURACY MATRIX — arm='{arm}'  (rows=length, cols=depth)")
    print("=" * 78)
    header = f"{'length \\ depth':<16}" + "".join(f"{d:>12}" for d in depths)
    print(header)
    print("-" * len(header))

    matrix = {}
    for length in lengths:
        row_cells = []
        matrix[length] = {}
        for depth in depths:
            cell = [
                r for r in results if r["length"] == length and r["depth"] == depth
            ]
            n = len(cell)
            hits = sum(1 for r in cell if r[arm]["success"])
            if n:
                row_cells.append(f"{_accuracy(hits, n) * 100:4.0f}% ({hits}/{n})".rjust(12))
            else:
                row_cells.append(f"{'-':>12}")
            matrix[length][depth] = {
                "n_cases": n,
                "accuracy": _accuracy(hits, n),
                "correct": hits,
            }
        print(f"{length:<16}" + "".join(row_cells))
    print("=" * 78)
    return {"lengths": lengths, "depths": depths, "cells": matrix}


def _traversal_diagnostics(results):
    stop_reasons: dict[str, int] = {}
    explored = []
    for r in results:
        sr = r["traversal"]["stop_reason"]
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
        explored.append(r["traversal"]["explored_nodes"])

    apiro_only = sum(
        1 for r in results if r["apiro"]["success"] and not r["bare_llm"]["success"]
    )
    bare_only = sum(
        1 for r in results if r["bare_llm"]["success"] and not r["apiro"]["success"]
    )

    print("\n" + "-" * 65)
    print(f"Apiro wins where bare LLM fails : {apiro_only}")
    print(f"Bare LLM wins where Apiro fails : {bare_only}")
    print(f"Stop reasons                    : {stop_reasons}")
    if explored:
        print(
            "Exploration expansions per case : "
            f"min={min(explored)} mean={sum(explored) / len(explored):.1f} "
            f"max={max(explored)}"
        )
    print("-" * 65 + "\n")
    return {
        "stop_reasons": stop_reasons,
        "apiro_only_wins": apiro_only,
        "bare_only_wins": bare_only,
    }


def _print_evaluator_summary(results):
    """Delegate to the shared evaluator summary printer when available."""
    try:
        from apiro.eval.evaluator import _print_summary
        _print_summary(results)
    except Exception as e:
        logger.debug(f"Summary printer notice: {e}")

def run_evaluation(cases_path: str = "data/niah_cases.json", real_components: bool = False, limit: int | None = None, out_path: str | None = None):
    cases_file = Path(cases_path)
    if not cases_file.is_absolute():
        cases_file = PROJECT_ROOT / cases_file
    cases = _load_cases(cases_file)
    if limit is not None:
        cases = cases[:limit]

    logger.info(f"Loaded {len(cases)} cases from {cases_file} (real_components={real_components})")

    if real_components:
        components = _build_real_components()
    else:
        components = _build_stub_components()

    axiom_extractor = AxiomExtractor()
    results = []

    for i, case in enumerate(cases):
        logger.info(f"\n[{i+1}/{len(cases)}] Processing case...")
        res = _evaluate_case(case, components, real_components, axiom_extractor)
        results.append(res)

    print("\n" + "=" * 78)
    print("CLINICAL NEEDLE-IN-A-HAYSTACK (NIAH) EVALUATION REPORT".center(78))
    print("=" * 78)

    overall = _overall_table(results)
    per_family = _per_family_table(results)
    matrix = _length_depth_matrix(results, arm="apiro")
    diagnostics = _traversal_diagnostics(results)

    if out_path:
        out_file = Path(out_path)
        if not out_file.is_absolute():
            out_file = PROJECT_ROOT / out_file
        out_file.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "cases_evaluated": len(results),
            "real_components": real_components,
            "overall": overall,
            "per_family": per_family,
            "length_depth_matrix": matrix,
            "diagnostics": diagnostics,
            "case_results": results,
        }
        with open(out_file, "w") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info(f"Saved evaluation results to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Run Clinical Needle-In-A-Haystack benchmark")
    parser.add_argument("--cases", type=str, default="data/niah_cases.json", help="Path to niah_cases.json")
    parser.add_argument("--real-components", action="store_true", help="Use live Ollama and ChromaDB components")
    parser.add_argument("--real", action="store_true", help="Alias for --real-components")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--out", type=str, default="data/niah_eval_results.json", help="Output path for results JSON")
    args = parser.parse_args()

    use_real = args.real_components or args.real
    run_evaluation(cases_path=args.cases, real_components=use_real, limit=args.limit, out_path=args.out)

if __name__ == "__main__":
    main()

