#!/usr/bin/env python3
"""
scripts/run_pmc_eval.py
=======================
Real-world PMC case-report benchmark. Compares Apiro's belief-graph traversal
against a bare zero-shot LLM and a standard RAG baseline on the ten scrubbed
PubMed Central case reports in ``data/pmc_cases.json``.

(The file was renamed from run_distractor_eval.py; the docstring still carried
the old name.)

CAVEAT ON THE CASE SET: the ``target_diagnosis`` fields in
``data/pmc_cases.json`` were produced by an unconstrained LLM in
``generate_pmc_cases.py`` and four of the ten are multi-paragraph prose rather
than a diagnosis label. Grading a differential against a paragraph is not a
fair test of any arm. Regenerate the set with the current
``generate_pmc_cases.py`` (which now post-processes the label) before treating
these accuracies as meaningful.

Usage:
    python scripts/run_pmc_eval.py --real
    python scripts/run_pmc_eval.py --real --limit 5 --out data/pmc_eval_results.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("distractor_eval")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apiro.graph.belief_graph import BeliefGraph
from apiro.graph.node import Node
from apiro.graph.traversal import ApiroTraversal
from apiro.axioms.extractor import AxiomExtractor
from apiro.config import N_DIFFERENTIAL, SATURATION_EXPLORATION_ONLY
from apiro.parsing import parse_differential

def _significance_block(results) -> dict:
    """Wilson intervals per arm plus paired McNemar tests between the arms.

    The PMC set is N = 10. An accuracy printed to one decimal place from ten
    binary outcomes implies a precision it does not have; the interval says so
    explicitly, and McNemar says whether any gap between two arms survives the
    fact that they were scored on the same ten cases.
    """
    from apiro.eval.metrics import mcnemar_exact, paired_bootstrap_delta_ci, wilson_interval

    n = len(results)
    outcomes = {
        arm: [bool(r[arm]["success"]) for r in results]
        for arm in ("apiro", "rag", "bare_llm")
    }
    labels = {"apiro": "Apiro", "rag": "Standard RAG", "bare_llm": "Bare LLM"}

    print("-" * 65)
    print("95% confidence intervals (Wilson):")
    for arm in ("apiro", "rag", "bare_llm"):
        hits = sum(outcomes[arm])
        low, high = wilson_interval(hits, n)
        print(f"  {labels[arm]:<14}{hits}/{n}  "
              f"[{low * 100:.1f}%, {high * 100:.1f}%]")

    print("Paired comparisons (exact McNemar):")
    stats = {}
    for challenger, baseline in (("apiro", "rag"), ("apiro", "bare_llm"), ("rag", "bare_llm")):
        delta = paired_bootstrap_delta_ci(outcomes[challenger], outcomes[baseline])
        mcn = mcnemar_exact(outcomes[challenger], outcomes[baseline])
        verdict = "significant" if mcn["significant_at_05"] else "NOT significant"
        print(f"  {labels[challenger]} vs {labels[baseline]}: "
              f"delta={delta['delta'] * 100:+.1f}pp, "
              f"discordant={mcn['n_discordant']}, "
              f"p={mcn['p_value']:.4f} ({verdict})")
        stats[f"{challenger}_vs_{baseline}"] = {"delta_ci": delta, "mcnemar": mcn}
    return stats


def run_evaluation(real_components: bool, limit: int | None = None, out_path: str | None = None):
    # Load distractor cases
    cases_path = PROJECT_ROOT / "data" / "pmc_cases.json"
    with open(cases_path) as f:
        cases = json.load(f)

    # Initialize components
    if real_components:
        # Shared wiring — see apiro/eval/harness.py. This block used to be a
        # verbatim copy of the one in run_niah_eval.py, and the two copies had
        # already drifted apart on the LLM timeout, so the two benchmarks were
        # not in fact reporting numbers from an identical stack.
        from apiro.eval.harness import build_real_components

        components = build_real_components(llm_timeout=120)
        embedder = components.embedder
        llm_client = components.llm_client
        contradiction = components.contradiction
        traversal = components.traversal
    else:
        # Import stub components
        from apiro.graph.expander import NodeExpander, StubEntropyEngine, StubChromaClient
        from apiro.graph.saturation import SaturationDetector
        from apiro.graph.rabbit_hole import RabbitHoleDetector
        
        # Stub Contradiction Detector simulating soft-pruning on specific distractor pairs
        class StubContradictionDetector:
            def check(self, claim_a: str, claim_b: str):
                from dataclasses import dataclass
                @dataclass
                class R:
                    label: str
                    score: float
                    negation_detected: bool
                a, b = claim_a.lower(), claim_b.lower()
                
                def match(kws1, kws2):
                    return (any(k in a for k in kws1) and any(k in b for k in kws2)) or \
                           (any(k in b for k in kws1) and any(k in a for k in kws2))

                # Case 1: ACS vs normal troponin/ECG
                if match({"myocardial infarction", "angina"}, {"normal troponin", "normal sinus rhythm"}):
                    return R("contradiction", 0.95, False)
                # Case 2: malaria vs negative blood films
                if match({"malaria"}, {"negative for plasmodium", "blood films are negative"}):
                    return R("contradiction", 0.95, False)
                # Case 3: thyroiditis ACS mimic vs normal troponin/ECG
                if match({"myocardial infarction", "coronary syndrome", "angina"}, {"normal sinus rhythm", "troponin", "tender", "thyroid"}):
                    return R("contradiction", 0.95, False)
                # Case 4: pulmonary embolism vs normal CTPA
                if match({"pulmonary embolism", "pe"}, {"no evidence of pulmonary embolism", "widened mediastinum", "aortic dissection"}):
                    return R("contradiction", 0.95, False)
                # Case 5: panic disorder vs elevated metanephrines/adrenal mass
                if match({"panic", "anxiety"}, {"metanephrines", "adrenal mass", "pheochromocytoma"}):
                    return R("contradiction", 0.95, False)
                # Case 6: gastroenteritis vs hyperpigmentation/cortisol/ACTH
                if match({"gastroenteritis", "dehydration"}, {"hyperpigmentation", "cortisol", "acth", "addison"}):
                    return R("contradiction", 0.95, False)
                # Case 7: parkinson/alzheimer vs NPH/lumbar puncture
                if match({"parkinson", "alzheimer"}, {"hydrocephalus", "nph", "lumbar puncture"}):
                    return R("contradiction", 0.95, False)
                # Case 8: appendicitis vs lead poisoning/normal CT
                if match({"appendicitis"}, {"lead poisoning", "lead level", "normal abdominal ct"}):
                    return R("contradiction", 0.95, False)
                # Case 9: multiple sclerosis vs NMO/normal brain MRI
                if match({"multiple sclerosis", "ms"}, {"neuromyelitis", "nmo", "aquaporin", "aqp4", "normal brain mri"}):
                    return R("contradiction", 0.95, False)
                # Case 10: stroke vs myasthenia gravis/normal brain CT
                if match({"stroke", "bell's palsy"}, {"myasthenia", "mg", "acetylcholine", "achr", "normal brain ct"}):
                    return R("contradiction", 0.95, False)
                return R("neutral", 0.5, False)
            def check_batch(self, pairs):
                return [self.check(a, b) for a, b in pairs]
            def should_check(self, claim_a: str, claim_b: str) -> bool:
                return True

        # Stub LLM Client simulating bare LLM vs RAG vs Apiro
        class StubLLMClient:
            def generate(self, prompt: str) -> str:
                # Bare LLM prompt detection
                if "Based on the following clinical presentation" in prompt:
                    if "dinner" in prompt or "substernal chest pain" in prompt:
                        return "1. Acute Myocardial Infarction\n2. Angina Pectoris\n3. Gastroesophageal Reflux Disease"
                    elif "sub-Saharan Africa" in prompt or "tea-colored urine" in prompt:
                        return "1. Malaria infection\n2. Acute Hepatitis\n3. Hemolytic Anemia"
                    elif "neck and chest pain radiating to the jaw" in prompt:
                        return "1. Acute Coronary Syndrome\n2. Angina Pectoris\n3. Myocardial Infarction"
                    elif "tearing chest pain" in prompt:
                        return "1. Pulmonary Embolism\n2. Pneumothorax\n3. Myocardial Infarction"
                    elif "headaches, profuse sweating, palpitations, and intense anxiety" in prompt:
                        return "1. Panic Disorder\n2. Generalized Anxiety Disorder\n3. Cardiovascular Arrhythmia"
                    elif "nausea, vomiting, abdominal pain, and weight loss" in prompt:
                        return "1. Acute Gastroenteritis\n2. Food Poisoning\n3. Dehydration"
                    elif "magnetic (gate-like) gait, and urinary urgency" in prompt:
                        return "1. Parkinson's Disease\n2. Alzheimer's Disease\n3. Vascular Dementia"
                    elif "colicky abdominal pain, constipation, and joint pain" in prompt:
                        return "1. Acute Appendicitis\n2. Bowel Obstruction\n3. Diverticulitis"
                    elif "bilateral vision loss and painful eye movements" in prompt:
                        return "1. Multiple Sclerosis\n2. Optic Neuritis\n3. Cerebral Venous Sinus Thrombosis"
                    elif "diplopia (double vision), ptosis" in prompt:
                        return "1. Acute Ischemic Stroke\n2. Bell's Palsy\n3. Transient Ischemic Attack"

                # RAG prompt detection
                if "Standard RAG presentation" in prompt or "retrieved medical context" in prompt:
                    if "dinner" in prompt or "substernal chest pain" in prompt:
                        return "1. Acute Myocardial Infarction\n2. Coronary Artery Disease\n3. Gastroesophageal Reflux Disease"
                    elif "sub-Saharan Africa" in prompt or "tea-colored urine" in prompt:
                        return "1. Malaria infection\n2. Hemolytic Anemia due to Malaria\n3. Acute Hepatitis"
                    elif "neck and chest pain radiating to the jaw" in prompt:
                        return "1. Acute Coronary Syndrome\n2. Angina Pectoris\n3. Subacute Thyroiditis"
                    elif "tearing chest pain" in prompt:
                        return "1. Pulmonary Embolism\n2. Pneumothorax\n3. Aortic Dissection"
                    elif "headaches, profuse sweating, palpitations, and intense anxiety" in prompt:
                        return "1. Panic Disorder\n2. Generalized Anxiety\n3. Cardiovascular Arrhythmia"
                    elif "nausea, vomiting, abdominal pain, and weight loss" in prompt:
                        return "1. Acute Gastroenteritis\n2. Food Poisoning\n3. Dehydration"
                    elif "magnetic (gate-like) gait, and urinary urgency" in prompt:
                        return "1. Parkinson's Disease\n2. Alzheimer's Disease\n3. Vascular Dementia"
                    elif "colicky abdominal pain, constipation, and joint pain" in prompt:
                        return "1. Acute Appendicitis\n2. Bowel Obstruction\n3. Diverticulitis"
                    elif "bilateral vision loss and painful eye movements" in prompt:
                        return "1. Multiple Sclerosis\n2. Optic Neuritis\n3. Cerebral Venous Sinus Thrombosis"
                    elif "diplopia (double vision), ptosis" in prompt:
                        return "1. Acute Ischemic Stroke\n2. Bell's Palsy\n3. Transient Ischemic Attack"
                
                # Expand node prompts
                p_lower = prompt.lower()
                if "synthesize the final top 3" in prompt:
                    # Match most-specific unique keywords FIRST to avoid false hits on 'chest pain'
                    if "thyroid" in p_lower or "tsh" in p_lower:
                        return "1. Subacute Thyroiditis\n2. Hyperthyroidism\n3. De Quervain Thyroiditis"
                    elif "aortic" in p_lower or "mediastinum" in p_lower:
                        return "1. Aortic Dissection\n2. Hypertensive Emergency\n3. Thoracic Aortic Aneurysm"
                    elif "metanephrines" in p_lower or "pheochromocytoma" in p_lower:
                        return "1. Pheochromocytoma\n2. Adrenal Adenoma\n3. Hypertension"
                    elif "cortisol" in p_lower or "acth" in p_lower:
                        return "1. Addison's Disease\n2. Adrenal Insufficiency\n3. Hyponatremia"
                    elif "hydrocephalus" in p_lower or "ventriculomegaly" in p_lower:
                        return "1. Normal Pressure Hydrocephalus\n2. Communicating Hydrocephalus\n3. Dementia"
                    elif "basophilic stippling" in p_lower or "blood lead" in p_lower:
                        return "1. Lead Poisoning\n2. Microcytic Anemia\n3. Sideroblastic Anemia"
                    elif "aquaporin" in p_lower or "letm" in p_lower or "aqp4" in p_lower:
                        return "1. Neuromyelitis Optica\n2. Longitudinal Myelitis\n3. NMOSD"
                    elif "acetylcholine receptor" in p_lower or "decremental" in p_lower:
                        return "1. Myasthenia Gravis\n2. Lambert-Eaton Syndrome\n3. Neuromuscular Junction Disorder"
                    elif "esophageal" in p_lower or "dysphagia" in p_lower:
                        return "1. Esophageal Spasm\n2. Gastroesophageal Reflux Disease\n3. Achalasia"
                    elif "g6pd" in p_lower or "bite cells" in p_lower or "heinz" in p_lower:
                        return "1. G6PD Deficiency\n2. Drug-induced Hemolytic Anemia\n3. Autoimmune Hemolytic Anemia"
                
                if "severe substernal chest pain" in p_lower and "thyroid" not in p_lower and "tsh" not in p_lower:
                    return "Hypotheses:\n1. Acute Myocardial Infarction is the primary cause\n2. Diffuse Esophageal Spasm should be ruled out\n3. Gastroesophageal Reflux Disease"
                if "thyroid gland" in p_lower or "tsh" in p_lower or "esr" in p_lower or ("neck" in p_lower and "thyroid" in p_lower):
                    return "Hypotheses:\n1. Subacute Thyroiditis de Quervain\n2. Hyperthyroidism Graves disease\n3. Thyroiditis autoimmune"
                if ("normal sinus rhythm" in p_lower or "troponin" in p_lower) and "thyroid" not in p_lower and "tsh" not in p_lower:
                    return "Hypotheses:\n1. Non-cardiac chest pain\n2. Esophageal Spasm causing spasm pain\n3. Reflux disease"
                if "bite cells" in p_lower or "nitrofurantoin" in p_lower:
                    return "Hypotheses:\n1. G6PD Deficiency hemolytic crisis\n2. Drug-induced hemolytic anemia\n3. Heinz body anemia"
                if "neck and chest pain" in p_lower:
                    return "Hypotheses:\n1. Acute Coronary Syndrome is possible\n2. Subacute Thyroiditis causing neck radiating pain\n3. Pharyngitis"
                if "thyroid gland" in p_lower or "tsh" in p_lower:
                    return "Hypotheses:\n1. Subacute Thyroiditis\n2. Graves disease hyperthyroidism\n3. Thyroid cyst"
                if "tearing chest pain" in p_lower:
                    return "Hypotheses:\n1. Pulmonary Embolism is possible\n2. Aortic Dissection causing tearing pain\n3. Tension pneumothorax"
                if "asymmetric blood pressure" in p_lower or "mediastinum" in p_lower:
                    return "Hypotheses:\n1. Aortic Dissection\n2. Thoracic aortic aneurysm\n3. Subclavian steal syndrome"
                if "anxiety" in p_lower or "palpitations" in p_lower:
                    return "Hypotheses:\n1. Panic Disorder attack\n2. Pheochromocytoma paroxysm\n3. Cardiac arrhythmia"
                if "metanephrines" in p_lower or "adrenal mass" in p_lower:
                    return "Hypotheses:\n1. Pheochromocytoma\n2. Adrenal adenoma\n3. Cushing disease"
                if "nausea, vomiting, abdominal pain" in p_lower:
                    return "Hypotheses:\n1. Acute Gastroenteritis\n2. Addison's Disease presenting as gastrointestinal crisis\n3. Bowel obstruction"
                if "hyperpigmentation" in p_lower or "cortisol" in p_lower:
                    return "Hypotheses:\n1. Addison's Disease adrenal insufficiency\n2. Nelson syndrome\n3. Congenital adrenal hyperplasia"
                if "cognitive decline" in p_lower or "gait" in p_lower:
                    return "Hypotheses:\n1. Parkinson's Disease or Parkinsonism\n2. Normal Pressure Hydrocephalus gait triad\n3. Alzheimer's Disease"
                if "lumbar puncture" in p_lower or "ventriculomegaly" in p_lower:
                    return "Hypotheses:\n1. Normal Pressure Hydrocephalus\n2. Obstructive hydrocephalus\n3. Pseudotumor cerebri"
                if "colicky abdominal pain" in p_lower:
                    return "Hypotheses:\n1. Acute Appendicitis abdominal pathology\n2. Lead Poisoning paint scraping history\n3. Nephrolithiasis"
                if "basophilic stippling" in p_lower or "lead" in p_lower:
                    return "Hypotheses:\n1. Lead Poisoning plumbing occupational\n2. Sideroblastic anemia\n3. Thalassemia minor"
                if "vision loss" in p_lower or "paraparesis" in p_lower:
                    return "Hypotheses:\n1. Multiple Sclerosis demyelinating\n2. Neuromyelitis Optica spectrum disorder\n3. Acute optic neuritis"
                if "aquaporin" in p_lower or "letm" in p_lower:
                    return "Hypotheses:\n1. Neuromyelitis Optica\n2. Transverse myelitis idiopathic\n3. MS plaque spinal cord"
                if "diplopia" in p_lower or "slurred speech" in p_lower:
                    return "Hypotheses:\n1. Acute Ischemic Stroke cerebrovascular\n2. Myasthenia Gravis fatigable weakness\n3. Bell's Palsy facial nerve"
                if "acetylcholine" in p_lower or "stimulation" in p_lower:
                    return "Hypotheses:\n1. Myasthenia Gravis neuromuscular\n2. Lambert-Eaton myasthenic syndrome\n3. Botulism toxin"
                
                return "Hypotheses:\n1. Alternative differential diagnosis A\n2. Alternative differential diagnosis B\n3. Alternative differential diagnosis C"

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
            theta=0.25, window=5, max_variance=0.04,
            exploration_only=SATURATION_EXPLORATION_ONLY,
        )
        rabbit_hole = RabbitHoleDetector(min_depth=3, reversal_window=4)
        traversal = ApiroTraversal(
            expander=expander,
            saturation=saturation,
            rabbit_hole=rabbit_hole,
            contradiction=contradiction,
        )


    # Axiom Extractor Setup
    axiom_extractor = AxiomExtractor()

    # Begin evaluation
    logger.info("=" * 65)
    logger.info(f"Running Distractor-Resilience Evaluation (Mode: {'REAL' if real_components else 'MOCK'})")
    logger.info("=" * 65)

    results = []

    # The case count used to be hard-coded to 5, so the "accuracy" figures in
    # the project log were computed over an arbitrary slice of the dataset with
    # no way to widen it.
    selected = cases if limit is None else cases[:limit]
    logger.info(f"Evaluating {len(selected)}/{len(cases)} cases.")

    for case in selected:
        case_id = case["case_id"]
        vignette = case["vignette"]
        target = case["target_diagnosis"]
        logger.info(f"\nEvaluating Case: {case_id} — {case['description']}")
        
        from apiro.eval.evaluator import _check_synthesis_hit

        # 1. Bare LLM Zero-Shot
        logger.info("  Running Bare LLM Zero-Shot...")
        prompt = (
            f"Based on the following clinical presentation, provide a list of your top "
            f"{N_DIFFERENTIAL} differential diagnoses, most likely first. Output ONLY the "
            f"{N_DIFFERENTIAL} diagnosis names, one per line, no numbering, no "
            f"explanation:\n\n"
            f"{vignette}"
        )
        bare_output = llm_client.generate(prompt)
        logger.info(f"  Bare LLM Output:\n{bare_output.strip()}")

        # Same parser and candidate budget as the Apiro arm — see the note in
        # run_niah_eval.py. Splitting on newlines gave the baselines an
        # uncapped candidate list while Apiro was held to 3 parsed slots.
        bare_items = parse_differential(bare_output, limit=N_DIFFERENTIAL)
        bare_success, _ = _check_synthesis_hit(
            bare_items,
            target,
            embedder=embedder if real_components else None,
            llm_client=llm_client if real_components else None
        )

        # 2. Standard RAG Baseline
        logger.info("  Running Standard RAG Baseline...")
        if real_components:
            rag_results = embedder.query(vignette, n_results=6)
            rag_context = "\n\n".join([r["text"] for r in rag_results])
            prompt_rag = (
                f"Based on the following clinical presentation and the retrieved medical "
                f"context, provide your top {N_DIFFERENTIAL} differential diagnoses, most "
                f"likely first. Output ONLY the {N_DIFFERENTIAL} diagnosis names, one per "
                f"line, no numbering, no explanation:\n\n"
                f"Vignette: {vignette}\n\nContext:\n{rag_context}"
            )
            rag_output = llm_client.generate(prompt_rag)
        else:
            prompt_rag = f"Standard RAG presentation:\n{vignette}"
            rag_output = llm_client.generate(prompt_rag)
        
        logger.info(f"  RAG Output:\n{rag_output.strip()}")
        
        rag_items = parse_differential(rag_output, limit=N_DIFFERENTIAL)
        rag_success, _ = _check_synthesis_hit(
            rag_items,
            target,
            embedder=embedder if real_components else None,
            llm_client=llm_client if real_components else None
        )

        # 3. Apiro Traversal
        logger.info("  Running Apiro Traversal...")
        graph = BeliefGraph()
        seeds = [
            Node(
                id=s["id"],
                claim=s["claim"] if " — " in s["claim"] or " ? " in s["claim"] else f"{s['claim']} — {s['domain']}",
                entropy_score=s["entropy"],
                domain=s["domain"],
                depth=s["depth"]
            )
            for s in case["seed_nodes"]
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
            vignette=vignette_to_pass
        )
        
        apiro_output = traversal_res.synthesis
        apiro_output_str = "\n".join(apiro_output)
        logger.info(f"  Apiro Final Synthesis:\n{apiro_output_str.strip()}")
        
        apiro_success, _ = _check_synthesis_hit(
            apiro_output,
            target,
            embedder=embedder if real_components else None,
            llm_client=llm_client if real_components else None
        )

        # Log soft-pruned nodes
        for node in graph.nodes.values():
            if getattr(node, "contradiction_penalty", 0.0) > 0:
                logger.info(
                    f"    - Contradiction Soft-Pruned Node: '{node.claim[:45]}' "
                    f"has penalty={node.contradiction_penalty} (is_rabbit_hole={node.is_rabbit_hole})"
                )

        results.append({
            "case_id": case_id,
            "description": case["description"],
            "target": target,
            "bare_llm": {
                "success": bare_success,
                "output": bare_output
            },
            "rag": {
                "success": rag_success,
                "output": rag_output
            },
            "apiro": {
                "success": apiro_success,
                "output": apiro_output
            },
            # Traversal diagnostics. Without these there is no way to tell a
            # wrong answer apart from an engine that never ran: before the
            # saturation fix every case stopped after ~5 seed expansions and
            # the summary table looked exactly the same.
            "traversal": {
                "stop_reason":     traversal_res.stop_reason,
                "total_nodes":     traversal_res.total_nodes,
                "total_edges":     traversal_res.total_edges,
                "seed_nodes":      sum(1 for n in graph.nodes.values() if n.depth == 0),
                "explored_nodes":  graph.count_expansions(min_depth=1),
                "max_depth_reached": max((n.depth for n in graph.nodes.values()), default=0),
                "rabbit_holes":    traversal_res.rabbit_hole_count,
                "contradictions":  traversal_res.contradiction_count,
                "duration_seconds": traversal_res.duration_seconds,
            },
        })

    # Summary Table
    print("\n" + "=" * 65)
    print("  DISTRACTOR-RESILIENCE EVALUATION SUMMARY")
    print("=" * 65)
    
    bare_wins = sum(1 for r in results if r["bare_llm"]["success"])
    rag_wins = sum(1 for r in results if r["rag"]["success"])
    apiro_wins = sum(1 for r in results if r["apiro"]["success"])
    
    for r in results:
        t = r["traversal"]
        print(f"Case {r['case_id']}: {r['description']}")
        print(f"  Target Diagnosis : {r['target']}")
        print(f"  Bare LLM Success : {'[PASS]' if r['bare_llm']['success'] else '[FAIL] (Hallucinated distractor)'}")
        print(f"  RAG Success      : {'[PASS]' if r['rag']['success'] else '[FAIL] (Hallucinated distractor)'}")
        print(f"  Apiro Success    : {'[PASS]' if r['apiro']['success'] else '[FAIL]'}")
        print(f"  Apiro Traversal  : stop={t['stop_reason']} nodes={t['total_nodes']} "
              f"(seeds={t['seed_nodes']}, explored={t['explored_nodes']}, "
              f"max_depth={t['max_depth_reached']}) {t['duration_seconds']}s")
        print("-" * 65)

    n = len(results) or 1
    print(f"Bare LLM Total Success: {bare_wins}/{len(results)} ({bare_wins/n*100:.1f}%)")
    print(f"RAG Baseline Success  : {rag_wins}/{len(results)} ({rag_wins/n*100:.1f}%)")
    print(f"Apiro Total Success   : {apiro_wins}/{len(results)} ({apiro_wins/n*100:.1f}%)")

    # At N = 10 a one-case swing moves accuracy by ten points, so the interval
    # and the paired test matter more here than the point estimate does.
    significance = _significance_block(results)

    # Head-to-head against the baseline Apiro is meant to beat.
    apiro_only = sum(1 for r in results if r["apiro"]["success"] and not r["bare_llm"]["success"])
    bare_only  = sum(1 for r in results if r["bare_llm"]["success"] and not r["apiro"]["success"])
    print("-" * 65)
    print(f"Apiro wins where bare LLM fails : {apiro_only}")
    print(f"Bare LLM wins where Apiro fails : {bare_only}")

    stop_reasons: dict[str, int] = {}
    explored = []
    for r in results:
        stop_reasons[r["traversal"]["stop_reason"]] = stop_reasons.get(r["traversal"]["stop_reason"], 0) + 1
        explored.append(r["traversal"]["explored_nodes"])
    print(f"Stop reasons                    : {stop_reasons}")
    if explored:
        print(f"Exploration expansions per case : "
              f"min={min(explored)} mean={sum(explored)/len(explored):.1f} max={max(explored)}")
    print("=" * 65 + "\n")

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "mode": "real" if real_components else "mock",
                "n_cases": len(results),
                "bare_llm_accuracy": bare_wins / n,
                "rag_accuracy": rag_wins / n,
                "apiro_accuracy": apiro_wins / n,
                "significance": significance,
                "stop_reasons": stop_reasons,
                "results": results,
            }, f, indent=2)
        print(f"Detailed results written to {out}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distractor-resilience evaluation")
    parser.add_argument("--real", action="store_true", help="Use real components (Ollama + ChromaDB)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N cases (default: all).")
    parser.add_argument("--out", type=str, default=None,
                        help="Write detailed per-case results to this JSON path.")
    args = parser.parse_args()
    run_evaluation(args.real, limit=args.limit, out_path=args.out)
