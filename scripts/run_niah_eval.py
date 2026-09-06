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
import re
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
from apiro.config import N_DIFFERENTIAL, SATURATION_EXPLORATION_ONLY
from apiro.parsing import ABSTENTION_SENTINEL, detect_abstention, parse_differential

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


def _first_present(case: dict, *keys):
    """First key whose value is not None.

    `a or b` treats 0 and 0.0 as absent. `depth_fraction` is legitimately 0.0
    for a needle placed at the very top of the haystack, so the old `or` chain
    reported the shallowest depth bucket — the control condition for the
    long-context claim — as "unknown" and dropped it from the matrix.
    """
    for key in keys:
        value = case.get(key)
        if value is not None:
            return value
    return None


def _length_bucket(case: dict) -> str:
    tokens = _first_present(case, "target_tokens", "approx_tokens", "length_tokens")
    if tokens is not None:
        # A round thousand renders as "2k", not "2000k". The old expression
        # appended "k" to the raw token count, so the published length x depth
        # matrix labelled 8,000-token contexts as "8000k" — eight million
        # tokens — in every report and in data/niah_eval_results.json.
        if tokens >= 1000 and tokens % 1000 == 0:
            return f"{tokens // 1000}k"
        return f"{tokens}tok"
    val = (
        case.get("length_bucket")
        or case.get("haystack_length")
        or case.get("length")
        or "unknown"
    )
    return str(val).strip().lower() or "unknown"


def _depth_bucket(case: dict) -> str:
    depth = _first_present(case, "depth_fraction", "depth")
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
    """Live Ollama + ChromaDB stack.

    The wiring (and a verbatim copy of the chroma adapter) used to live here
    and again in run_pmc_eval.py. It is now shared — see
    apiro/eval/harness.py for why that mattered.
    """
    from apiro.eval.harness import build_real_components

    components = build_real_components(llm_timeout=120)
    return {
        "embedder": components.embedder,
        "llm_client": components.llm_client,
        "traversal_factory": components.create_traversal,
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
    traversal = (
        components["traversal_factory"](
            n_diagnoses=N_DIFFERENTIAL,
            allow_abstention=components.get("allow_abstention", False),
        )
        if real_components
        else components["traversal"]
    )

    case_id = case.get("case_id", case.get("id", "?"))
    vignette = case.get("vignette") or case.get("haystack") or case.get("context") or ""
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
        f"Based on the following clinical presentation, provide a list of your top "
        f"{N_DIFFERENTIAL} differential diagnoses, most likely first. Output ONLY the "
        f"{N_DIFFERENTIAL} diagnosis names, one per line, no numbering, no "
        f"explanation.\n"
        f"If the note does not contain enough information to identify a diagnosis, "
        f"reply with exactly '{ABSTENTION_SENTINEL}' and nothing else.\n\n"
        f"{vignette}"
    )
    bare_output = llm_client.generate(prompt)
    logger.info(f"  Bare LLM Output:\n{bare_output.strip()}")
    # Same parser and the same candidate budget as the Apiro arm. Splitting the
    # raw reply on newlines handed the baselines an uncapped candidate list
    # (~7 lines per case) while Apiro was graded on 3 parsed slots — so a
    # baseline could hit the target on its seventh line and still be scored
    # correct, which is not the question the benchmark claims to answer.
    bare_items = parse_differential(bare_output, limit=N_DIFFERENTIAL)
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
            f"Based on the following clinical presentation and the retrieved medical "
            f"context, provide your top {N_DIFFERENTIAL} differential diagnoses, most "
            f"likely first. Output ONLY the {N_DIFFERENTIAL} diagnosis names, one per "
            f"line, no numbering, no explanation.\n"
            f"If the note does not contain enough information to identify a diagnosis, "
            f"reply with exactly '{ABSTENTION_SENTINEL}' and nothing else.\n\n"
            f"Vignette: {vignette}\n\nContext:\n{rag_context}"
        )
        rag_output = llm_client.generate(prompt_rag)
    else:
        prompt_rag = f"Standard RAG presentation:\n{vignette}"
        rag_output = llm_client.generate(prompt_rag)
    logger.info(f"  RAG Output:\n{rag_output.strip()}")
    rag_items = parse_differential(rag_output, limit=N_DIFFERENTIAL)
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

    if not seeds:
        # build_niah_cases.py does not emit a `seed_nodes` field, and stub mode
        # skips axiom extraction, so the Apiro arm was handed an empty frontier:
        # the traversal returned stop_reason="no_frontier" on iteration 1 and
        # synthesised from an empty graph. That is a guaranteed loss dressed up
        # as a result, so seed the presentation itself and say so — the same
        # fallback apiro.axioms.seeding.build_seeds() applies.
        logger.warning(
            f"  Case {case_id} has no seed nodes and no axioms were extracted "
            f"(stub mode). Seeding the raw presentation so the traversal has a "
            f"frontier."
        )
        seeds = [Node(
            id="ax_fallback_0",
            claim=f"The patient presents with the following clinical picture: "
                  f"{vignette.strip()[:400]}",
            entropy_score=0.01,
            domain="pathophysiology",
            depth=0,
            metadata={"axiom_weight": 1.0, "polarity": "affirmed", "fallback": True},
        )]

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
        # Carried through so the significance report can state the effective
        # sample size. Cases sharing a diagnosis are near-duplicates.
        "diagnosis": case.get("diagnosis", "unknown"),
        # The curated wrong answer, where the family ships one. This is what
        # makes distractor selection directly measurable rather than inferred.
        "wrong_diagnosis": (case.get("metadata") or {}).get("wrong_diagnosis"),
        # Matched-pair linkage (--paired) and counterfactual linkage
        # (--counterfactual). pair_role is clean/adversarial or control/trap.
        "pair_id": case.get("pair_id"),
        "pair_role": case.get("pair_role"),
        "perturbation": case.get("perturbation"),
        "unanswerable": bool((case.get("metadata") or {}).get("unanswerable")),
        "prior_diagnosis": (case.get("metadata") or {}).get("prior_diagnosis"),
        "target": _target_display(case),
        "targets": targets,
        "bare_llm": {"success": bare_success, "output": bare_output,
                     "candidates": bare_items,
                     "abstained": detect_abstention(bare_output)},
        "rag": {"success": rag_success, "output": rag_output,
                "candidates": rag_items,
                "abstained": detect_abstention(rag_output)},
        "apiro": {"success": apiro_success, "output": apiro_output,
                  "candidates": list(apiro_output),
                  # The engine's differential is already parsed, so scan the
                  # joined candidates; an empty differential is not by itself
                  # an abstention (it may be an unparseable answer).
                  "abstained": detect_abstention("\n".join(apiro_output))},
        # Candidates offered per arm. If these are not equal, the accuracy
        # comparison is confounded by output formatting rather than reasoning.
        "n_candidates": {
            "bare_llm": len(bare_items),
            "rag": len(rag_items),
            "apiro": len(apiro_output),
        },
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


_NUMERIC_BUCKET_RE = re.compile(r"^(\d+(?:\.\d+)?)")


def _bucket_sort_key(label: str) -> tuple[float, str]:
    """Sort key that orders generated buckets numerically, not lexically.

    The generated buckets are strings like "2k", "8k", "25%", "100%". A plain
    `sorted()` puts "100%" before "25%" and would put "16k" before "2k", so the
    published matrix read as if accuracy were being swept in an arbitrary order.
    A numeric prefix sorts on its value; anything else sorts last, by name.
    """
    match = _NUMERIC_BUCKET_RE.match(label)
    if not match:
        return (float("inf"), label)
    scale = 1000.0 if label[match.end():].startswith("k") else 1.0
    return (float(match.group(1)) * scale, label)


def _ordered_buckets(values: set[str], canonical: list[str]) -> list[str]:
    """Canonical buckets first (in their declared order), then the rest sorted
    numerically, with "unknown" always last."""
    ordered = [b for b in canonical if b in values]
    extra = values - set(canonical) - {"unknown"}
    ordered += sorted(extra, key=_bucket_sort_key)
    if "unknown" in values:
        ordered.append("unknown")
    return ordered


def _length_depth_matrix(results, arm="apiro"):
    """Length (rows) x Depth (cols) accuracy heatmap for a single arm."""
    lengths = _ordered_buckets({r["length"] for r in results}, LENGTH_BUCKETS)
    depths = _ordered_buckets({r["depth"] for r in results}, DEPTH_BUCKETS)

    print("\n" + "=" * 78)
    print(f"  LENGTH x DEPTH ACCURACY MATRIX — arm='{arm}'  (rows=length, cols=depth)")
    print("=" * 78)
    # NOTE: the corner label is built outside the f-string. A literal backslash
    # inside an f-string expression is a SyntaxError before Python 3.12, and
    # pyproject declares requires-python = ">=3.10" — this file did not parse
    # at all on the project's own minimum interpreter.
    corner = "length \\ depth"
    header = f"{corner:<16}" + "".join(f"{d:>12}" for d in depths)
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


def _distractor_selection_table(results, embedder=None, llm_client=None):
    """How often does each arm name the curated WRONG diagnosis?

    This is the architecture's actual claim, measured directly. Accuracy says
    whether the right answer was found; this says whether the *designed* wrong
    answer was chosen instead — which is what "rejects distractors instead of
    rationalizing them" means operationally.

    It is also the most statistically efficient endpoint available here: it
    conditions on nothing (every case with a curated distractor contributes)
    and it needs one traversal per case, so at equal compute it reaches
    significance sooner than aggregate accuracy does.

    Only families that ship an explicit wrong-diagnosis label contribute;
    contradiction_needle does.
    """
    from apiro.eval.harness import make_matcher
    from apiro.eval.metrics import mcnemar_exact, wilson_interval

    scored = [r for r in results if r.get("wrong_diagnosis")]
    if not scored:
        return None

    matcher = make_matcher(embedder=embedder, llm_client=llm_client)
    labels = {"apiro": "Apiro", "rag": "Standard RAG", "bare_llm": "Bare LLM"}

    picked = {}
    for arm in ("apiro", "rag", "bare_llm"):
        flags = []
        for r in scored:
            top = (r[arm].get("candidates") or [None])[0]
            flags.append(bool(top) and matcher(str(top), r["wrong_diagnosis"]))
        picked[arm] = flags

    n = len(scored)
    print("\n" + "=" * 78)
    print("  PRIMARY ENDPOINT — DISTRACTOR SELECTION")
    print(f"  Top-ranked answer is the curated WRONG diagnosis. Lower is better. N = {n}")
    print("=" * 78)
    for arm in ("apiro", "rag", "bare_llm"):
        hits = sum(picked[arm])
        low, high = wilson_interval(hits, n)
        print(f"  {labels[arm]:<16}{hits:>3}/{n:<4}{hits / n * 100:>7.1f}%"
              f"   95% CI [{low * 100:5.1f}%, {high * 100:5.1f}%]")

    stats = {"n_cases": n, "rates": {a: sum(f) / n for a, f in picked.items()}}
    print("-" * 78)
    for challenger, baseline in (("apiro", "rag"), ("apiro", "bare_llm")):
        # "Avoided the distractor" is the success direction here.
        mcn = mcnemar_exact(
            [not f for f in picked[challenger]],
            [not f for f in picked[baseline]],
        )
        verdict = "significant" if mcn["significant_at_05"] else "NOT significant"
        print(f"  {labels[challenger]} avoids it where {labels[baseline]} does not: "
              f"{mcn['a_only']} vs {mcn['b_only']} of {mcn['n_discordant']} "
              f"discordant, p = {mcn['p_value']:.4f} ({verdict})")
        stats[f"{challenger}_vs_{baseline}"] = mcn
    print("=" * 78)
    return stats


def _counterfactual_table(results):
    """Bias Trap Rate — the sharpest test of whether evidence beats priors.

    Control and trap share a presenting syndrome, so the statistical prior
    points at the same diagnosis for both; only the buried discriminative
    evidence differs, and in the trap it implies a different disease. A model
    running on priors is right on the control and wrong on the trap.

    Conditioning on the control being correct isolates the question: of the
    cases an arm could do, how often did flipping the evidence fail to flip the
    answer? Unlike a distractor that leaves the answer unchanged, a trap cannot
    be passed by ignoring the note.

    Design after MedEinst (arXiv:2601.06636).
    """
    from apiro.eval.metrics import bias_trap_rate, compare_bias_traps

    pairs = {}
    for r in results:
        if r.get("pair_id") and r.get("pair_role") in ("control", "trap"):
            pairs.setdefault(r["pair_id"], {})[r["pair_role"]] = r
    complete = [v for v in pairs.values() if "control" in v and "trap" in v]
    if not complete:
        return None

    labels = {"apiro": "Apiro", "rag": "Standard RAG", "bare_llm": "Bare LLM"}
    outcomes = {
        arm: ([bool(p["control"][arm]["success"]) for p in complete],
              [bool(p["trap"][arm]["success"]) for p in complete])
        for arm in ("apiro", "rag", "bare_llm")
    }

    print("\n" + "=" * 78)
    print("  PRIMARY ENDPOINT — BIAS TRAP RATE (counterfactual pairs)")
    print(f"  P(wrong on trap | right on control). Lower is better. "
          f"{len(complete)} pairs")
    print("=" * 78)
    print(f"{'Arm':<16}{'control':>10}{'trap':>10}{'trapped':>10}"
          f"{'trap rate':>12}{'95% CI':>20}")
    print("-" * 78)
    stats = {"n_pairs": len(complete), "arms": {}}
    for arm in ("apiro", "rag", "bare_llm"):
        t = bias_trap_rate(*outcomes[arm])
        rate = "n/a" if t["trap_rate"] is None else f"{t['trap_rate'] * 100:.1f}%"
        lo, hi = t["trap_rate_ci"]
        print(f"{labels[arm]:<16}{t['control_accuracy'] * 100:>9.1f}%"
              f"{t['trap_accuracy'] * 100:>9.1f}%"
              f"{t['n_trapped']:>6}/{t['n_control_correct']:<3}"
              f"{rate:>12}{f'[{lo * 100:.0f}%, {hi * 100:.0f}%]':>20}")
        stats["arms"][arm] = t
    print("-" * 78)
    for challenger, baseline in (("apiro", "rag"), ("apiro", "bare_llm")):
        cmp_ = compare_bias_traps(*outcomes[challenger], *outcomes[baseline])
        if not cmp_["n_comparable"]:
            print(f"  {labels[challenger]} vs {labels[baseline]}: no pair both got "
                  f"right on the control — nothing comparable.")
            continue
        mcn = cmp_["mcnemar"]
        verdict = "significant" if mcn["significant_at_05"] else "NOT significant"
        print(f"  {labels[challenger]} vs {labels[baseline]}, on the "
              f"{cmp_['n_comparable']} pairs BOTH got right on the control:")
        print(f"      escaped the trap: {cmp_['a_escaped']} vs {cmp_['b_escaped']};"
              f"  p = {mcn['p_value']:.4f} ({verdict})")
        stats[f"{challenger}_vs_{baseline}"] = cmp_
    print("=" * 78)
    return stats


def _abstention_table(results):
    """Fabrication rate on cases whose discriminative evidence was removed.

    On an unanswerable note the only correct behaviour is to decline. Naming a
    diagnosis anyway is a confident fabrication — the failure this architecture
    exists to prevent, and the one Pillar 3 has so far only inferred from a
    heuristic confidence score rather than measured.

    Answering when it need not have is reported separately: over-abstention is
    a usefulness cost, not a safety failure, and collapsing the two into one
    accuracy number hides which is happening.

    Design after MedAbstain (arXiv:2601.12471).
    """
    from apiro.eval.metrics import abstention_metrics, mcnemar_exact

    if not any(r.get("unanswerable") for r in results):
        return None

    labels = {"apiro": "Apiro", "rag": "Standard RAG", "bare_llm": "Bare LLM"}
    should = [bool(r.get("unanswerable")) for r in results]

    print("\n" + "=" * 78)
    print("  ABSTENTION — cases with the discriminative evidence removed")
    print(f"  Fabrication = named a diagnosis when none was supported. "
          f"Lower is better.")
    print("=" * 78)
    print(f"{'Arm':<16}{'fabricated':>13}{'fab. rate':>12}{'95% CI':>20}"
          f"{'over-abstain':>15}")
    print("-" * 78)
    stats, flags = {}, {}
    for arm in ("apiro", "rag", "bare_llm"):
        abst = [bool(r[arm].get("abstained")) for r in results]
        flags[arm] = abst
        a = abstention_metrics(
            abst, should,
            correct_when_answered=[bool(r[arm]["success"]) for r in results],
        )
        rate = "n/a" if a["fabrication_rate"] is None else f"{a['fabrication_rate'] * 100:.1f}%"
        over = "n/a" if a["over_abstention_rate"] is None else f"{a['over_abstention_rate'] * 100:.1f}%"
        lo, hi = a["fabrication_rate_ci"]
        print(f"{labels[arm]:<16}{a['fabricated']:>8}/{a['n_unanswerable']:<4}"
              f"{rate:>12}{f'[{lo * 100:.0f}%, {hi * 100:.0f}%]':>20}{over:>15}")
        stats[arm] = a
    print("-" * 78)
    unanswerable_idx = [i for i, u in enumerate(should) if u]
    for challenger, baseline in (("apiro", "rag"), ("apiro", "bare_llm")):
        # Success direction: correctly declining on an unanswerable case.
        mcn = mcnemar_exact(
            [flags[challenger][i] for i in unanswerable_idx],
            [flags[baseline][i] for i in unanswerable_idx],
        )
        verdict = "significant" if mcn["significant_at_05"] else "NOT significant"
        print(f"  {labels[challenger]} declines where {labels[baseline]} fabricates: "
              f"{mcn['a_only']} vs {mcn['b_only']} of {mcn['n_discordant']} "
              f"discordant, p = {mcn['p_value']:.4f} ({verdict})")
        stats[f"{challenger}_vs_{baseline}"] = mcn
    print("=" * 78)
    return stats


def _resilience_table(results):
    """Matched-pair distractor resilience, when the case set was built --paired.

    Each pair is the same haystack, needle and depth with and without an
    adversarial sentence, so the pair is its own control and the variance of
    case difficulty — large, and unrelated to the thesis — cancels out.

    Retention, P(correct on adversarial | correct on clean), is the headline:
    of the cases an arm could solve, how many survived the distractor?

    Costs two runs per pair and discards pairs either arm failed clean, so it
    is less statistically efficient than distractor selection above. It buys
    interpretability instead: it separates resilience from raw capability,
    which no single-condition metric can.
    """
    from apiro.eval.metrics import compare_robustness, distractor_robustness

    pairs = {}
    for r in results:
        if r.get("pair_id") and r.get("pair_role"):
            pairs.setdefault(r["pair_id"], {})[r["pair_role"]] = r
    complete = [v for v in pairs.values() if "clean" in v and "adversarial" in v]
    if not complete:
        return None

    labels = {"apiro": "Apiro", "rag": "Standard RAG", "bare_llm": "Bare LLM"}
    outcomes = {
        arm: (
            [bool(p["clean"][arm]["success"]) for p in complete],
            [bool(p["adversarial"][arm]["success"]) for p in complete],
        )
        for arm in ("apiro", "rag", "bare_llm")
    }

    print("\n" + "=" * 78)
    print("  MATCHED-PAIR DISTRACTOR RESILIENCE")
    print(f"  Same haystack, needle and depth ± one adversarial sentence. "
          f"{len(complete)} pairs")
    print("=" * 78)
    print(f"{'Arm':<16}{'clean':>9}{'adversarial':>13}{'degradation':>13}"
          f"{'broken':>9}{'retention':>11}")
    print("-" * 78)
    stats = {"n_pairs": len(complete), "arms": {}}
    for arm in ("apiro", "rag", "bare_llm"):
        rob = distractor_robustness(*outcomes[arm])
        ret = "n/a" if rob["retention"] is None else f"{rob['retention'] * 100:.0f}%"
        print(f"{labels[arm]:<16}{rob['clean_accuracy'] * 100:>8.1f}%"
              f"{rob['adversarial_accuracy'] * 100:>12.1f}%"
              f"{rob['degradation'] * 100:>+12.1f}pp"
              f"{rob['broken']:>9}{ret:>11}")
        stats["arms"][arm] = rob
    print("-" * 78)
    for challenger, baseline in (("apiro", "rag"), ("apiro", "bare_llm")):
        cmp_ = compare_robustness(*outcomes[challenger], *outcomes[baseline])
        if not cmp_["n_comparable"]:
            print(f"  {labels[challenger]} vs {labels[baseline]}: no pair solved "
                  f"clean by both — nothing comparable.")
            continue
        mcn = cmp_["mcnemar"]
        verdict = "significant" if mcn["significant_at_05"] else "NOT significant"
        print(f"  {labels[challenger]} vs {labels[baseline]}, on the "
              f"{cmp_['n_comparable']} pairs BOTH solved clean:")
        print(f"      survived the distractor: {cmp_['a_survived']} vs "
              f"{cmp_['b_survived']};  p = {mcn['p_value']:.4f} ({verdict})")
        stats[f"{challenger}_vs_{baseline}"] = cmp_
    print("=" * 78)
    return stats


def _significance_table(results):
    """Paired significance of each arm against the two baselines.

    The overall table reports three accuracies over the same 25 cases and the
    README turns the gap between two of them into a headline ("+28% over
    RAG"). Three accuracies on shared cases is a *paired* design, so the
    supporting test is McNemar on the discordant cases, not an eyeball
    comparison of two independent proportions. Reported here so the number and
    its uncertainty travel together.
    """
    from apiro.eval.metrics import mcnemar_exact, paired_bootstrap_delta_ci, wilson_interval

    n = len(results)
    outcomes = {
        arm: [bool(r[arm]["success"]) for r in results]
        for arm in ("apiro", "rag", "bare_llm")
    }
    labels = {"apiro": "Apiro", "rag": "Standard RAG", "bare_llm": "Bare LLM"}

    # Effective sample size. McNemar and the bootstrap both assume independent
    # cases; cases built from the same diagnosis share needles, distractors and
    # phrasing, so a p-value computed over many near-duplicates is narrower
    # than the evidence supports.
    diagnoses = {r.get("diagnosis", "unknown") for r in results}
    per_dx = n / max(1, len(diagnoses))
    print("\n" + "=" * 78)
    print(f"  CASE INDEPENDENCE: {n} cases over {len(diagnoses)} distinct "
          f"diagnoses ({per_dx:.1f} each)")
    if per_dx > 8:
        print("  WARNING: cases sharing a diagnosis are near-duplicates. The")
        print("           p-values below assume independence and are therefore")
        print("           optimistic. Widen NEEDLE_BANK in build_niah_cases.py.")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("  ACCURACY WITH 95% CONFIDENCE INTERVALS (Wilson)")
    print("=" * 78)
    for arm in ("apiro", "rag", "bare_llm"):
        hits = sum(outcomes[arm])
        low, high = wilson_interval(hits, n)
        print(f"  {labels[arm]:<16}{hits:>3}/{n:<4}"
              f"{hits / n * 100 if n else 0:>7.1f}%"
              f"   95% CI [{low * 100:5.1f}%, {high * 100:5.1f}%]")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("  PAIRED COMPARISONS (exact McNemar on discordant cases)")
    print("=" * 78)
    stats = {}
    for challenger, baseline in (("apiro", "rag"), ("apiro", "bare_llm"), ("rag", "bare_llm")):
        delta = paired_bootstrap_delta_ci(outcomes[challenger], outcomes[baseline])
        mcn = mcnemar_exact(outcomes[challenger], outcomes[baseline])
        verdict = "significant" if mcn["significant_at_05"] else "NOT significant"
        print(f"  {labels[challenger]} vs {labels[baseline]}")
        print(f"      delta = {delta['delta'] * 100:+6.1f} pp   "
              f"95% CI [{delta['ci_low'] * 100:+.1f}, {delta['ci_high'] * 100:+.1f}] pp")
        print(f"      McNemar: {mcn['a_only']} won / {mcn['b_only']} lost of "
              f"{mcn['n_discordant']} discordant, p = {mcn['p_value']:.4f} "
              f"({verdict} at alpha = 0.05)")
        stats[f"{challenger}_vs_{baseline}"] = {"delta_ci": delta, "mcnemar": mcn}
    print("=" * 78)
    stats["_independence"] = {
        "n_cases": n,
        "n_distinct_diagnoses": len(diagnoses),
        "cases_per_diagnosis": round(per_dx, 2),
    }
    return stats


def _signal_health_table(log_dir="data"):
    """Is the engine's entropy score discriminating, or has it collapsed?

    Read from the traversal logs the run just wrote. The frontier ordering, the
    synthesis ranking and the saturation window all read this one number, so a
    degenerate signal silently disables three mechanisms at once while every
    stage still appears to run. It cost a full benchmark round to notice.
    """
    import glob
    from apiro.eval.metrics import signal_health

    values = []
    for path in glob.glob(f"{log_dir}/traversal_log_*.jsonl"):
        try:
            for line in open(path):
                event = json.loads(line)
                if event.get("event") == "node_expanded" and event.get("entropy") is not None:
                    values.append(event["entropy"])
        except Exception:  # noqa: BLE001 - a malformed log must not fail the report
            continue
    if not values:
        return None

    h = signal_health(values)
    print("\n" + "=" * 78)
    print("  ENTROPY SIGNAL HEALTH")
    print("=" * 78)
    print(f"  generated hypotheses scored : {h['n']}")
    print(f"  distinct values             : {h['n_distinct']}")
    print(f"  most common value           : {h['modal_value']} "
          f"({h['modal_share'] * 100:.1f}% of all nodes)")
    print(f"  mean / stdev                : {h['mean']:.3f} / {h['stdev']:.3f}")
    print(f"  normalized entropy          : {h['normalized_entropy']:.3f} "
          f"(1.0 = fully spread, 0.0 = constant)")
    if h["degenerate"]:
        print()
        print("  *** DEGENERATE ***  One value covers "
              f"{h['modal_share'] * 100:.0f}% of nodes.")
        print("  The frontier ordering, the synthesis ranking and the saturation")
        print("  window all read this score. Treat any traversal result below as")
        print("  measuring a system whose central mechanism was not active.")
    print("=" * 78)
    return h


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


# NOTE: a `_print_evaluator_summary()` helper used to live here. It was never
# called, and it passed the per-case results *list* to
# apiro.eval.evaluator._print_summary(), which expects the aggregate summary
# *dict* — so had anything called it, it would have printed an all-zero table
# behind a swallowed exception. The reporting this harness actually uses is
# _overall_table / _per_family_table / _length_depth_matrix below.


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

    # Offer abstention to the Apiro arm only when the case set actually
    # contains cases with no answer. On an all-answerable set the option costs
    # accuracy and buys nothing measurable — it produced a 50% over-abstention
    # rate on the 2026-08-30 run.
    has_unanswerable = any((c.get("metadata") or {}).get("unanswerable") for c in cases)
    components["allow_abstention"] = has_unanswerable
    expander = getattr(components.get("traversal"), "expander", None)
    if expander is not None:
        expander.allow_abstention = has_unanswerable
    logger.info(
        f"Abstention {'ENABLED' if has_unanswerable else 'disabled'} "
        f"({'unanswerable cases present' if has_unanswerable else 'every case is answerable'})."
    )

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
    # On-thesis endpoints first: they test what the architecture claims, and
    # distractor selection is also the most statistically efficient of the
    # three (see docs/BENCHMARKING.md).
    signal = _signal_health_table()
    counterfactual = _counterfactual_table(results)
    abstention = _abstention_table(results)
    distractor = _distractor_selection_table(
        results,
        embedder=components["embedder"],
        llm_client=components["llm_client"] if real_components else None,
    )
    resilience = _resilience_table(results)
    significance = _significance_table(results)
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
            "signal_health": signal,
            "counterfactual_traps": counterfactual,
            "abstention": abstention,
            "distractor_selection": distractor,
            "resilience": resilience,
            "significance": significance,
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
