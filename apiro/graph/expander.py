"""
graph/expander.py
-----------------
Given a node (a clinical hypothesis), generates 3 child hypotheses using:
  1. RAG — retrieve relevant medical context from ChromaDB
  2. LLM — generate child hypotheses conditioned on that context

THE PIPELINE (per the spec's NodeExpander.expand()):
  1. Query ChromaDB for top-6 relevant chunks for this node's claim
  2. Build a prompt: system message + retrieved context + parent claim
  3. Call the LLM → parse exactly 3 hypotheses (one per line)
  4. For each hypothesis:
     a. Compute entropy  (via EntropyEngine from Phase 1)
     b. Classify domain  (via DomainClassifier — simple keyword mapping here)
     c. Run contradiction check vs ALL existing nodes
     d. Create Node + Edge, add to graph
  5. Return list of new nodes

INTEGRATION NOTES:
  - EntropyEngine and BeliefGraph come from Phase 1 (this package)
  - ChromaDB client is passed in — we don't own its lifecycle
  - LLM client is passed in — allows easy swap (OpenAI ↔ Anthropic ↔ local)
  - DomainClassifier is inline here (simple keyword rules) — can be extracted later

STUB FALLBACKS:
  We provide StubEntropyEngine and StubChromaClient so this module is testable
  WITHOUT an Ollama instance. The real objects have the same interface.
"""

import re
import logging

from apiro.graph.node import Node
from apiro.graph.edge import Edge
from apiro.parsing import (
    ABSTENTION_SENTINEL,
    DIFFERENTIAL_SENTINEL,
    detect_abstention,
    parse_claims,
    parse_differential,
)
from apiro.config import (
    ENTROPY_SIGNAL,
    RAG_DOMAIN_FILTER,
    N_CHILD_HYPOTHESES,
    N_DIFFERENTIAL,
    CONTRADICTION_THRESHOLD,
    RAG_TOP_K,
    RAG_MIN_CHUNKS_FOR_GROUNDING,
    RAG_MAX_DISTANCE,
)

logger = logging.getLogger(__name__)



# ── Domain classifier ─────────────────────────────────────────────────────────
# Simple keyword-based domain tagger. Good enough for Phase 2.
# Can be replaced with a classifier model later with zero interface changes.

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "genetics":        ["gene", "genetic", "mutation", "allele", "chromosom", "hereditary", "inherited"],
    "pharmacology":    ["drug", "medication", "dose", "prescribe", "administer", "mg", "contraindicated", "antibiotic", "statin"],
    "imaging":         ["ct", "mri", "x-ray", "ultrasound", "scan", "radiograph", "echo", "echocardiogram"],
    "lab":             ["blood", "serum", "plasma", "troponin", "creatinine", "bilirubin", "wbc", "rbc", "platelet", "culture"],
    "pathophysiology": ["mechanism", "pathway", "cascade", "ischemia", "inflammation", "necrosis", "apoptosis", "fibrosis"],
    "treatment":       ["surgery", "procedure", "therapy", "treatment", "intervention", "resect", "catheter", "stent"],
    "comorbidity":     ["comorbid", "concurrent", "coexisting", "secondary", "complication", "alongside"],
}

# Prototype sentences for semantic fallback classification.
# One representative sentence per domain, embedded at first call.
_DOMAIN_PROTOTYPES: dict[str, str] = {
    "genetics":        "Gene mutation and chromosomal inheritance pattern in hereditary disease.",
    "pharmacology":    "Drug dose, medication administration, contraindication and antibiotic treatment.",
    "imaging":         "CT scan, MRI, ultrasound and radiographic imaging findings.",
    "lab":             "Blood serum levels, troponin, creatinine, electrolytes and lab measurements.",
    "pathophysiology": "Disease mechanism, inflammatory cascade, ischemia and cellular necrosis pathway.",
    "treatment":       "Surgical intervention, procedure, therapy and catheter stent placement.",
    "comorbidity":     "Concurrent complication, secondary condition and coexisting disease.",
}

_domain_prototype_embeddings: dict | None = None   # lazy-loaded


def classify_domain(text: str, embedder=None) -> str:
    """
    Hybrid domain classification:
      Pass 1 — fast keyword matching (covers obvious cases).
      Pass 2 — if no keyword hits, use sentence-transformer cosine similarity
               against prototype sentences (handles edge cases like 'electrolyte
               imbalance' → 'lab' instead of defaulting to 'pathophysiology').
    """
    global _domain_prototype_embeddings

    text_lower = text.lower()
    scores = {
        domain: sum(1 for kw in keywords if kw in text_lower)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    # ── Semantic fallback ─────────────────────────────────────────────────────
    if embedder is not None:
        try:
            if _domain_prototype_embeddings is None:
                protos = list(_DOMAIN_PROTOTYPES.values())
                keys   = list(_DOMAIN_PROTOTYPES.keys())
                embs   = embedder._model.encode(protos, normalize_embeddings=True)
                _domain_prototype_embeddings = {k: e for k, e in zip(keys, embs)}

            text_emb = embedder._model.encode([text], normalize_embeddings=True)[0]
            sims = {
                domain: float(text_emb @ emb)
                for domain, emb in _domain_prototype_embeddings.items()
            }
            return max(sims, key=sims.get)
        except Exception:
            pass  # fall through to default

    return "pathophysiology"


# ── Stub components (for testing without Ollama or ChromaDB) ──────────────────

class StubEntropyEngine:
    """
    Deterministic fake entropy engine for testing.

    SWAP POINT — replace with the real engine adapter:
        from apiro.entropy.engine import EntropyEngine

        class RealEntropyAdapter:
            def __init__(self):
                self._engine = EntropyEngine()
            def compute(self, claim: str, context_chunks=None) -> float:
                result = self._engine.epistemic_certainty_entropy(claim, context_chunks)
                return result if result is not None else 0.5

    The stub returns a value that slowly declines with each call to mimic
    realistic saturation behaviour in test runs.
    """

    def __init__(self, start: float = 0.85, step: float = 0.06):
        self._value = start
        self._step = step

    def compute(self, claim: str, context_chunks: list[str] | None = None) -> float:
        """
        Returns a slowly declining entropy. Each call to compute() decreases
        the value by `step` to simulate convergence, bottoming out at 0.05.
        """
        val = max(0.05, self._value)
        self._value = max(0.05, self._value - self._step)
        return round(val, 4)

    def epistemic_certainty_entropy(self, claim: str, context_chunks: list[str] | None = None) -> float:
        """Alias to compute() for compatibility with real EntropyEngine interface."""
        return self.compute(claim, context_chunks)



class StubChromaClient:
    """
    Fake ChromaDB client. Returns fixed medical context chunks.

    SWAP POINT — replace with a real ChromaDB client:
        import chromadb
        chroma_client = chromadb.Client()
        # or: chromadb.HttpClient(host="localhost", port=8000)
    """

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 6,
    ) -> dict:
        """Returns stub context chunks that look like real ChromaDB output."""
        stub_docs = [
            "Troponin elevation above 99th percentile indicates myocardial injury.",
            "ST-elevation on ECG in V1-V4 leads suggests anterior STEMI.",
            "Aspirin 300mg loading dose is first-line in ACS management.",
            "Primary PCI within 90 minutes is the gold standard for STEMI.",
            "Beta-blockers reduce mortality in post-MI patients without contraindications.",
            "ACE inhibitors are indicated post-MI with reduced ejection fraction.",
        ][:n_results]
        return {"documents": [stub_docs]}


# ── LLM prompt template ───────────────────────────────────────────────────────
# Design rationale:
#   The prompt must be domain-anchored and evidence-constrained to prevent
#   topic drift (e.g. STEMI → calyceal arteritis). Three hard rules are
#   enforced explicitly in the prompt text:
#     1. Stay within the parent domain or one directly adjacent domain.
#     2. Every hypothesis must be grounded in the retrieved evidence — do not
#        introduce organ systems, conditions, or drugs not mentioned above.
#     3. Output exactly 3 short, specific, single-sentence clinical claims.
#   These rules are verbose by design: LLMs follow explicit constraints more
#   reliably than implicit style guidance.

HYPOTHESIS_PROMPT_TEMPLATE = """\
You are Apiro, a precise clinical differential-diagnosis engine.

Your task: given a parent clinical claim, the patient's clinical case presentation, and retrieved medical evidence, generate
exactly 3 child hypotheses that deepen the diagnostic reasoning.

=== PARENT CLAIM ===
{claim}

=== MEDICAL DOMAIN ===
{domain}

=== PATIENT CLINICAL PRESENTATION ===
{case_context}

=== RETRIEVED EVIDENCE (use ONLY what is stated here) ===
{rag_chunks}

=== STRICT RULES ===
1. DOMAIN LOCK: Every hypothesis MUST remain within the "{domain}" domain or one
   directly clinically adjacent domain (e.g. pathophysiology ↔ lab findings).
   Do NOT introduce unrelated organ systems, rare syndromes, or diseases not
   mentioned in the evidence or patient presentation above.
2. EVIDENCE GROUNDED: Every hypothesis must be directly derivable from the
   evidence or patient presentation above. Do not speculate beyond what is supported.
3. CLINICAL SPECIFICITY: Each hypothesis must be a specific, testable clinical
   claim — not a vague statement. Include mechanism, finding, or intervention.
4. FORMAT: Output exactly 3 hypotheses, one per line, no numbering, no preamble,
   no explanation. Each hypothesis is a single sentence under 25 words.

=== OUTPUT (3 lines only) ===\
"""

# Parametric fallback prompt — used when corpus retrieval returns too few chunks.
# The engine switches to pure LLM parametric knowledge so rare-disease nodes
# still expand meaningfully instead of recycling thin or irrelevant context.
PARAMETRIC_PROMPT_TEMPLATE = """\
You are Apiro, a precise clinical differential-diagnosis engine.

Your task: given a parent clinical claim and the patient's clinical case presentation, generate exactly 3 child hypotheses
that deepen the diagnostic reasoning using established medical knowledge.

=== PARENT CLAIM ===
{claim}

=== MEDICAL DOMAIN ===
{domain}

=== PATIENT CLINICAL PRESENTATION ===
{case_context}

=== STRICT RULES ===
1. DOMAIN: Stay within the "{domain}" domain or one clinically adjacent domain.
2. KNOWLEDGE-BASED: Use your medical training knowledge to generate plausible
   clinical hypotheses. Include known mechanisms, biomarkers, or findings.
3. RARE/UNCOMMON diseases are acceptable — this mode is specifically for cases
   where corpus coverage is sparse.
4. FORMAT: Output exactly 3 hypotheses, one per line, no numbering, no preamble,
   no explanation. Each hypothesis is a single sentence under 25 words.

=== OUTPUT (3 lines only) ===\
"""


# ── NodeExpander ──────────────────────────────────────────────────────────────

class NodeExpander:
    """
    Expands a single node into 3 child hypothesis nodes.

    Args:
        entropy_engine:        EntropyEngine (Phase 1) or StubEntropyEngine for testing.
                               Must have a .compute(claim, chunks) -> float method.
                               SWAP POINT: replace StubEntropyEngine with EntropyEngine.
        chroma_client:         ChromaDB client or StubChromaClient for testing.
        llm_client:            LLM client — must have a .chat(prompt: str) -> str method.
        collection_name:       ChromaDB collection to query (default: "medical_knowledge").
        contradiction_detector: Optional, used to flag contradictions inline.
    """

    def __init__(
        self,
        entropy_engine,
        chroma_client,
        llm_client,
        collection_name: str = "medical_knowledge",
        contradiction_detector=None,
        inline_contradiction_check: bool = False,
        n_children: int = N_CHILD_HYPOTHESES,
        n_diagnoses: int = N_DIFFERENTIAL,
        allow_abstention: bool = False,
    ):
        self.entropy_engine = entropy_engine
        self.chroma_client = chroma_client
        self.llm_client = llm_client
        self.collection_name = collection_name
        self.contradiction_detector = contradiction_detector
        # ApiroTraversal owns the contradiction pass and runs it batched over
        # the same node set. Set True only when driving the expander directly
        # without a traversal.
        self.inline_contradiction_check = inline_contradiction_check
        # Number of child hypotheses to keep per expansion. Config-driven:
        # N_CHILD_HYPOTHESES used to be imported and then ignored, with the
        # count hard-coded to 3 in _parse_hypotheses' default argument, so
        # changing it in config.py had no effect on the engine.
        self.n_children = n_children
        # Size of the final differential. A harness that lets its baselines
        # offer five candidates must let this arm offer five too.
        self.n_diagnoses = n_diagnoses
        # Whether synthesis may decline to answer. OFF by default.
        #
        # Offering it unconditionally was a regression: on the 2026-08-30
        # C-NIAH run the engine replied INSUFFICIENT EVIDENCE on 5 of 10
        # cases that all had a findable needle, while neither baseline
        # abstained once. An abstention option belongs on a benchmark that
        # contains unanswerable cases (build_niah_cases.py --counterfactual)
        # and nowhere else — on an answerable case it converts a possible hit
        # into a guaranteed miss.
        self.allow_abstention = allow_abstention

    @staticmethod
    def _generate_node_id(parent_id: str, index: int) -> str:
        """Deterministic child ID: {parent_id}_c{index}"""
        return f"{parent_id}_c{index}"

    # Sentence scaffolding added by the axiom extractor. It is there so the NLI
    # detector sees grammatical claims, but it is dead weight in a vector query:
    # every seed embeds partly as "the patient presents with the clinical
    # finding of", which pulls all seeds toward each other and away from the
    # textbook passage that actually describes the finding.
    _SEED_PREFIXES = (
        "the patient presents with the clinical finding of ",
        "the patient has a lab result or vital sign showing ",
        "the patient has a lab result showing ",
        "the patient has a documented diagnosis of ",
        "the patient denies the clinical finding of ",
        "the patient has a history of ",
        "the patient has undergone ",
        "the patient is on ",
    )

    @classmethod
    def _retrieval_query(cls, claim: str) -> str:
        """Strip forged-sentence scaffolding before querying the vector store."""
        lowered = claim.lower()
        for prefix in cls._SEED_PREFIXES:
            if lowered.startswith(prefix):
                return claim[len(prefix):].rstrip(". ").strip() or claim
        return claim

    def _retrieve_context(
        self,
        claim: str,
        domain: str = "",
        n_results: int = RAG_TOP_K,
    ) -> tuple[list[str], bool]:
        """
        Query ChromaDB for the top-N most relevant medical text chunks.

        Returns:
            (chunks, is_corpus_grounded)  where is_corpus_grounded is True when
            at least RAG_MIN_CHUNKS_FOR_GROUNDING chunks were returned, meaning
            the corpus has enough coverage to trust the evidence-based prompt.
            When False the caller should switch to the parametric prompt.
        """
        where: dict | None = None
        if RAG_DOMAIN_FILTER and domain:
            db_domain = domain.replace(" findings", "").lower()
            if db_domain in ("symptom", "vital"):
                db_domain = "pathophysiology"
            
            allowed_domains = {"genetics", "pharmacology", "imaging", "lab", "treatment", "comorbidity", "pathophysiology"}
            if db_domain in allowed_domains:
                where = {"medical_domain": db_domain}

        query_text = self._retrieval_query(claim)

        chunks: list[str] = []
        distances: list[float] = []
        try:
            try:
                result = self.chroma_client.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where,
                )
            except TypeError:
                result = self.chroma_client.query(
                    collection_name=self.collection_name,
                    query_texts=[query_text],
                    n_results=n_results,
                )
            docs = result.get("documents", [[]])
            chunks = docs[0] if docs else []
            dists = result.get("distances") or []
            distances = dists[0] if dists else []
        except Exception as e:
            logger.warning(f"[NodeExpander] ChromaDB query failed: {e}. Continuing without context.")

        # Drop chunks that are merely the nearest neighbours rather than actual
        # evidence. A vector store always returns top-k; without this the
        # expander hands the LLM six off-topic passages labelled
        # "use ONLY what is stated here" and steers it away from a rare
        # diagnosis it would otherwise have reached from parametric knowledge.
        if RAG_MAX_DISTANCE is not None and distances and len(distances) == len(chunks):
            kept = [c for c, d in zip(chunks, distances) if d is not None and d <= RAG_MAX_DISTANCE]
            if len(kept) != len(chunks):
                logger.info(
                    f"[NodeExpander] Dropped {len(chunks) - len(kept)}/{len(chunks)} retrieved "
                    f"chunks beyond distance {RAG_MAX_DISTANCE} for '{query_text[:50]}'."
                )
            chunks = kept

        is_grounded = len(chunks) >= RAG_MIN_CHUNKS_FOR_GROUNDING
        if not is_grounded:
            logger.info(
                f"[NodeExpander] Sparse corpus coverage ({len(chunks)} chunks) for "
                f"'{claim[:50]}' — switching to parametric mode."
            )
        return chunks, is_grounded


    def _sanitize_vignette(self, vignette: str) -> str:
        if not vignette:
            return ""
        # Remove common adversarial/jailbreak prompt injection markers
        bad_words = [
            r"\bignore\b.*\b(instructions|rules|prompt|above|previous)\b",
            r"\bforget\b.*\b(instructions|rules|prompt|above|previous)\b",
            r"\bdo\s+not\s+follow\b",
            r"\bsystem\s*:",
            r"\buser\s*:",
            r"\bassistant\s*:",
        ]
        sanitized = vignette
        for pattern in bad_words:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

    def _build_prompt(self, node: Node, chunks: list[str], graph, is_grounded: bool = True, vignette: str = None) -> str:
        """
        Build the hypothesis-generation prompt.

        When is_grounded=True (corpus returned enough chunks), uses the
        evidence-constrained HYPOTHESIS_PROMPT_TEMPLATE.
        When is_grounded=False (sparse corpus), uses PARAMETRIC_PROMPT_TEMPLATE
        so the LLM can reason from its training knowledge for rare diseases.
        """
        # Extract patient case context (all seed nodes with depth=0)
        seeds = [n.claim for n in graph.nodes.values() if n.depth == 0]
        seed_context = "\n".join(f"  - {s}" for s in seeds) if seeds else "  - [No seed context]"
        
        sanitized_vignette = self._sanitize_vignette(vignette)
        case_context = f"Original Clinical Vignette:\n{sanitized_vignette}\n\nSeed Findings:\n{seed_context}" if sanitized_vignette else seed_context

        if not is_grounded:
            return PARAMETRIC_PROMPT_TEMPLATE.format(
                claim=node.claim,
                domain=node.domain,
                case_context=case_context,
            )
        if chunks:
            rag_text = "\n".join(f"  [{i+1}] {chunk.strip()}" for i, chunk in enumerate(chunks))
        else:
            rag_text = "  [No context retrieved — be conservative, stay close to parent claim.]"
        return HYPOTHESIS_PROMPT_TEMPLATE.format(
            claim=node.claim,
            domain=node.domain,
            case_context=case_context,
            rag_chunks=rag_text,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return raw text response."""
        try:
            return self.llm_client.chat(prompt)
        except Exception as e:
            logger.error(f"[NodeExpander] LLM call failed: {e}")
            return ""

    # NOTE: an earlier design generated hypotheses and their entropies in one
    # LLM call, aligning each hypothesis to a slice of that call's token
    # logprobs (real Shannon entropy over the model's own token
    # distribution). That path (_call_llm_with_logprobs /
    # _align_logprobs_to_hypotheses) was never reconnected after the entropy
    # engine was rewritten — see apiro/entropy/engine.py's module docstring —
    # and was removed from here since nothing called it. The live entropy
    # signal is _batch_entropy() below, which scores each already-generated
    # hypothesis independently via EntropyEngine.temperature_corrected_entropy().

    def _parse_hypotheses(self, llm_output: str, limit: int = 3, pad: bool = False) -> list[str]:
        """
        Parse the LLM's output into up to `limit` hypothesis claims.

        Delegates to apiro.parsing.parse_claims. The previous implementation
        stripped a single leading bullet character, which left "**Hypothesis
        1:**" as "*Hypothesis 1:**" and admitted it to the graph as a clinical
        claim. See apiro/parsing.py for the measured cost of that.

        `pad` is False by default. Padding used to inject synthetic
        "[Expansion failed for: ...]" strings, which then entered the graph as
        real nodes: the entropy engine scores anything starting with "[" at
        ln(2), the maximum, so these placeholders sorted straight to the top of
        the depth>=1 frontier and got expanded ahead of genuine hypotheses,
        burning the node budget on nothing. An expansion that yields two usable
        hypotheses should return two.
        """
        hypotheses = parse_claims(llm_output, limit=limit)

        if pad:
            while len(hypotheses) < limit:
                hypotheses.append(
                    f"[Expansion failed for: {hypotheses[0][:40] if hypotheses else 'unknown'}]"
                )

        return hypotheses

    def _batch_entropy(
        self,
        hypotheses: list[str],
        chunks: list[str],
        case_context: str = "",
    ) -> list[tuple[float, float | None]]:
        """
        Score entropy for all hypotheses concurrently.

        Uses the entropy engine's temperature_corrected_entropy on the pre-built
        verification prompt for each hypothesis. Results are collected concurrently
        but share the already-retrieved RAG chunks, avoiding redundant context
        construction and letting the caller reuse the same chunk list.

        Under config.ENTROPY_SIGNAL == "posterior" this asks how confident the
        model is that each hypothesis is the primary diagnosis FOR THIS
        PATIENT, and returns both the entropy and that confidence. Under
        "breadth" it uses the original diagnostic-breadth question and returns
        no confidence — see apiro/entropy/engine.py for why the default moved.

        Falls back to ln(2) (max binary uncertainty) on any failure so the node
        stays high-priority in the frontier.

        Returns:
            One ``(entropy, confidence_or_None)`` per hypothesis, in order.
        """
        DEFAULT = 0.693  # ln(2) — max binary uncertainty fallback

        def _score_hyp(hyp: str) -> tuple[float, float | None]:
            try:
                if (ENTROPY_SIGNAL == "posterior"
                        and hasattr(self.entropy_engine, "score_hypothesis")):
                    scored = self.entropy_engine.score_hypothesis(hyp, case_context)
                    if scored is not None:
                        return scored
                    return DEFAULT, None
                prompt = self.entropy_engine._build_verification_prompt(hyp, chunks)
                val = self.entropy_engine.temperature_corrected_entropy(prompt)
                return (val if val is not None else DEFAULT), None
            except Exception:
                return DEFAULT, None

        if not hypotheses:
            return []

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(hypotheses), self.n_children)
        ) as executor:
            scores = list(executor.map(_score_hyp, hypotheses))

        return scores

    def expand(self, node: Node, graph, vignette: str = None) -> list[Node]:
        """
        Main expansion method — called by traversal.py for each frontier node.

        Steps:
          1. Retrieve RAG context
          2. Build prompt
          3. Call LLM
          4. Parse hypotheses
          5. For each hypothesis: compute entropy, classify domain,
             check contradictions, create Node + Edge, add to graph
          6. Return list of 3 new nodes
        """
        logger.info(f"[NodeExpander] Expanding: '{node.claim[:60]}'")

        # Step 1: RAG retrieval with grounding check
        chunks, is_grounded = self._retrieve_context(node.claim, domain=node.domain)

        # Step 2 & 3: Prompt + LLM
        # Use evidence-constrained prompt when corpus is reliable;
        # parametric prompt when corpus coverage is too sparse to trust.
        prompt = self._build_prompt(node, chunks, graph, is_grounded=is_grounded, vignette=vignette)
        raw_output = self._call_llm(prompt)

        # Step 4: Parse
        hypotheses = self._parse_hypotheses(raw_output, limit=self.n_children)

        # Step 5a: Batch entropy — score all 3 hypotheses concurrently via
        # EntropyEngine.temperature_corrected_entropy(), which (per
        # apiro/entropy/engine.py) asks the LLM to self-report how many
        # distinct diagnoses could explain the claim and maps that count
        # through a fixed table. This is a bounded uncertainty heuristic,
        # not Shannon entropy computed over a token probability distribution.
        scored = self._batch_entropy(
            hypotheses, chunks, case_context=self._sanitize_vignette(vignette) or node.claim
        )

        new_nodes = []

        for i, hypothesis in enumerate(hypotheses):
            entropy, confidence = scored[i]

            domain = classify_domain(hypothesis, embedder=getattr(self.chroma_client, '_emb', None))

            # Step 5b: Semantic DAG Merging.
            # The expanding node, its ancestors, and every depth-0 axiom are
            # excluded. Merging into an ancestor makes a cycle and throws the
            # expansion away; merging into an axiom replaces a new hypothesis
            # with a restatement of a fact the engine already had.
            exclude = {node.id} | graph.ancestors_of(node.id)
            exclude |= {n.id for n in graph.nodes.values() if n.depth == 0}
            match = graph.find_semantic_match(hypothesis, exclude_ids=exclude)
            if match:
                logger.info(
                    f"[NodeExpander] Semantic match: merging '{hypothesis[:40]}' "
                    f"into existing node {match.id}"
                )
                edge = Edge(parent_id=node.id, child_id=match.id)
                try:
                    graph.add_edge(edge)
                except ValueError as e:
                    logger.debug(f"[NodeExpander] Could not add merged edge: {e}")
                # We skip contradiction checks and adding the node, 
                # but we don't append to new_nodes because it's already in the graph.
                continue

            # Step 5c: Create the node (uses main's full Node dataclass)
            child_id = self._generate_node_id(node.id, i)
            child_node = Node(
                id=child_id,
                claim=hypothesis,
                entropy_score=entropy,
                domain=domain,
                depth=node.depth + 1,
                parent_id=node.id,
                # Belief, kept separately from uncertainty. The entropy map is
                # symmetric about p = 0.5, so H alone cannot tell "confidently
                # yes" from "confidently no" — and synthesis needs that.
                metadata=({"confidence": confidence} if confidence is not None else {}),
            )

            # Step 5d: Create the edge
            edge = Edge(parent_id=node.id, child_id=child_id)

            # Step 5e: Optional inline contradiction pass.
            # OFF by default: ApiroTraversal runs the same comparison over the
            # same node set immediately after expand() returns, batched. Doing
            # it here as well doubled the LLM-judge calls (the expensive stage
            # of the two-stage detector) for a flag the traversal recomputes.
            if self.contradiction_detector and self.inline_contradiction_check:
                for existing in list(graph.nodes.values()):
                    if not self.contradiction_detector.should_check(hypothesis, existing.claim):
                        continue
                    result = self.contradiction_detector.check(hypothesis, existing.claim)
                    if result.label == "contradiction" and result.score >= CONTRADICTION_THRESHOLD:
                        edge.contradiction_flag = True
                        logger.info(
                            f"[NodeExpander] Contradiction flagged: "
                            f"'{hypothesis[:40]}' vs '{existing.claim[:40]}'"
                        )
                        break


            # Step 5f: Add to graph
            # add_node() may silently drop the node if it exceeds max_depth or
            # max_nodes budget. Only add the edge if the node was actually accepted.
            graph.add_node(child_node)
            if child_id not in graph.nodes:
                # Rejected by the graph (past max_depth). Returning it anyway
                # made the traversal contradiction-check and log a node that
                # does not exist, and let phantom nodes reach the caller.
                logger.debug(
                    f"[NodeExpander] Node '{child_id}' rejected by graph "
                    f"(depth={child_node.depth} exceeds max_depth) — dropping."
                )
                continue
            graph.add_edge(edge)
            new_nodes.append(child_node)
            logger.debug(
                f"  → Child {i}: '{hypothesis[:60]}' "
                f"(entropy={entropy:.3f}, domain={domain})"
            )

        return new_nodes


    def synthesize_differential(
        self,
        graph,
        top_k: int = 15,
        vignette: str = None,
        n_diagnoses: int | None = None,
    ) -> list[str]:
        """
        Synthesize a final differential diagnosis from the belief graph.

        Four-step architecture:

          1. Partition nodes into confirmed findings (depth-0 anchors /
             ruled-out), viable hypotheses (depth >= 1, clean), and
             contradicted hypotheses (contradiction_penalty > 0).
          2. Build a CONTRADICTED-HYPOTHESES section that weighs these down
             *softly* — a non-zero contradiction penalty is a disagreement
             signal, NOT a proof of impossibility. Soft-pruning is documented
             as "penalise, do not delete, so alternative hypotheses stay alive
             in the synthesis layer." Only genuine depth-0 negations are hard
             constraints.
          3. Build the confirmed-findings grounding section, instructing the
             model to prefer a single unifying primary etiology over restating
             a symptom complex or a chronic risk factor.
          4. Assemble the prompt, call the LLM, parse up to 3 diagnoses.

        Args:
            graph:     The BeliefGraph containing gathered evidence.
            top_k:     Max number of exploration claims passed to the LLM.
            vignette:  Raw (or axiom-enriched) clinical presentation.
            n_diagnoses: How many ranked candidates to return. Defaults to the
                expander's `n_diagnoses`. Must match what the baselines are
                allowed to offer, or the comparison is not a comparison.

        Returns:
            A list of up to `n_diagnoses` clinical diagnoses, most likely first.
        """
        n_diagnoses = self.n_diagnoses if n_diagnoses is None else n_diagnoses
        logger.info(
            f"[NodeExpander] Synthesizing final differential "
            f"({n_diagnoses} candidates)..."
        )

        # --- Step 1: partition -------------------------------------------------
        anchors: list[Node]      = []   # depth 0, affirmed  -> ground truth
        ruled_out: list[Node]    = []   # negated            -> hard constraint
        explored: list[Node]     = []   # depth >= 1, clean  -> viable hypotheses
        contradicted: list[Node] = []   # depth >= 1, penalised -> soft down-weight

        for n in graph.nodes.values():
            claim_l = n.claim.lower()
            is_negated = (
                "denies" in claim_l
                or claim_l.startswith("the patient has no ")
                or getattr(n, "polarity", "") == "negated"
            )
            if n.depth == 0:
                (ruled_out if is_negated else anchors).append(n)
                continue
            if n.is_rabbit_hole:
                continue          # known dead-end
            if is_negated:
                ruled_out.append(n)
            elif getattr(n, "contradiction_penalty", 0.0) > 0.0:
                contradicted.append(n)
            else:
                explored.append(n)

        # Rank exploration claims by diagnostic specificity: a claim compatible
        # with one diagnosis narrows the differential, a claim compatible with
        # ten does not. Deeper claims break ties.
        def specificity(n: Node) -> tuple:
            """Sort key: most believable first, most specific as the tiebreak.

            When a node carries a posterior confidence, rank by that
            descending. Ranking by ascending entropy alone would put a
            confidently RULED OUT hypothesis at the top of the differential,
            because the entropy map is symmetric — H = 0.234 is both 5% and 95%
            confidence. Nodes without a confidence (the "breadth" signal, or a
            failed elicitation) fall back to the old ascending-entropy order.
            """
            h = n.entropy_score if n.entropy_score is not None else 0.693
            confidence = (n.metadata or {}).get("confidence")
            if confidence is None:
                return (1, round(h, 4), -n.depth)
            return (0, -round(float(confidence), 4), -n.depth)

        explored.sort(key=specificity)
        contradicted.sort(key=specificity)

        top_nodes = explored[:top_k]
        if not top_nodes and not anchors:
            logger.warning("[NodeExpander] Graph is empty — nothing to synthesize.")
            return []

        logger.info(
            f"[NodeExpander] Synthesis context: {len(anchors)} anchors, "
            f"{len(top_nodes)}/{len(explored)} exploration claims, "
            f"{len(ruled_out)} ruled-out, {len(contradicted)} contradicted."
        )

        def _bullets(nodes, dedupe: set) -> str:
            out = []
            for n in nodes:
                claim = n.claim.strip()
                key = claim.lower()
                if key in dedupe:
                    continue
                dedupe.add(key)
                out.append(f"  - {claim}")
            return "\n".join(out)

        seen: set[str] = set()
        anchors_text      = _bullets(anchors, seen)
        evidence_text     = _bullets(top_nodes, seen)
        contradicted_text = _bullets(contradicted[:5], seen)
        ruled_out_text    = _bullets(ruled_out, seen)

        # --- Steps 2 & 3: assemble prompt -------------------------------------
        sections = ["You are Apiro, a precise clinical differential-diagnosis engine.\n"]
        sections.append(
            "Your task: identify the single underlying disease that best explains this"
            " patient, and give the top 3 candidate diagnoses ranked by likelihood.\n"
        )

        clean_vignette = self._sanitize_vignette(vignette)
        if clean_vignette:
            sections.append(f"=== PATIENT PRESENTATION ===\n{clean_vignette}\n")

        # Step 3: confirmed objective findings grounding, with etiology-ranking rule.
        if anchors_text:
            sections.append(
                "=== CONFIRMED OBJECTIVE FINDINGS (deterministically extracted — treat as"
                " ground truth) ===\n"
                f"{anchors_text}\n"
                "When ranking candidate diagnoses, prefer a single unifying PRIMARY"
                " underlying etiology that explains the objective findings over a"
                " secondary symptom complex or a non-specific chronic history (e.g. name"
                " the primary disease driving an acute deterioration rather than merely"
                " restating a chronic risk factor or the presenting symptom).\n"
            )

        # Hard constraint: depth-0 negations genuinely exclude diagnoses.
        if ruled_out_text:
            sections.append(
                "=== RULED-OUT / NEGATED FINDINGS (a diagnosis requiring these is wrong) ===\n"
                f"{ruled_out_text}\n"
            )

        if evidence_text:
            sections.append(
                "=== REASONING TRACE (claims the engine derived, most specific first) ===\n"
                f"{evidence_text}\n"
            )

        # Step 2: contradicted hypotheses — SOFT down-weight, not a hard ban.
        if contradicted_text:
            sections.append(
                "=== CONTRADICTED HYPOTHESES (evidence disagreed — weigh down heavily) ===\n"
                f"{contradicted_text}\n"
                "These claims conflicted with other evidence in the graph. This is a"
                " down-weighting signal, NOT a proof of impossibility: only propose one"
                " of these as the primary diagnosis if it explains the CONFIRMED objective"
                " findings better than any non-contradicted alternative.\n"
            )

        # Output contract. An 8B model ignores "no preamble, no explanation"
        # roughly half the time — on the committed C-NIAH run it spent 57% of
        # Apiro's answer slots on markdown scaffolding. A required per-line
        # sentinel is followed far more reliably, and lets the parser discard
        # everything the model says around the answer (see apiro/parsing.py).
        sections.append(
            "=== STRICT RULES ===\n"
            f"1. Output exactly {n_diagnoses} lines, most likely first.\n"
            f"2. Every line MUST start with '{DIFFERENTIAL_SENTINEL} ' followed by the"
            " disease name and nothing else.\n"
            "3. Give only the specific disease name (e.g."
            f" '{DIFFERENTIAL_SENTINEL} Type 1 autoimmune pancreatitis')."
            " No preamble, numbering, mechanism, or explanation.\n"
            "4. Name the specific underlying primary disease, not a syndrome or a"
            " symptom (e.g. prefer 'Pheochromocytoma' over 'Hypertensive crisis').\n"
            "5. Every diagnosis must be compatible with ALL confirmed and ruled-out"
            " findings above.\n"
            "6. Weigh the CONFIRMED OBJECTIVE FINDINGS above the typical"
            " presentation. When a finding rules out the diagnosis the presentation"
            " suggests, follow the finding.\n"
            + (
                f"7. If the evidence above does not support any specific diagnosis,"
                f" reply with exactly '{DIFFERENTIAL_SENTINEL} {ABSTENTION_SENTINEL}'"
                f" and nothing else. Declining is correct when the note cannot"
                f" support an answer; guessing is not.\n"
                if self.allow_abstention else ""
            )
            + f"\n=== OUTPUT ({n_diagnoses} lines, each beginning"
            f" '{DIFFERENTIAL_SENTINEL} ') ==="
        )

        prompt = "\n".join(sections)

        # --- Step 4: call + parse ---------------------------------------------
        raw_output = self._call_llm(prompt)
        diagnoses = parse_differential(raw_output, limit=n_diagnoses)

        # A formatting miss used to cost the engine the whole comparison: an
        # unparseable answer became an empty differential, scored as a loss
        # against baselines whose raw text was searched line by line. One
        # retry with a harder instruction costs a single call on the case where
        # it matters and nothing on the cases where it does not.
        # An explicit refusal is a complete answer, not a short one. Retrying
        # it would badger the model out of the behaviour rule 7 asks for, and
        # would turn a correct abstention into a fabricated diagnosis. Only
        # honoured when abstention was offered: otherwise an unprompted hedge
        # ("I cannot determine...") would silently discard the differential.
        if self.allow_abstention and detect_abstention(raw_output):
            logger.info("[NodeExpander] Synthesis declined: evidence insufficient.")
            return [ABSTENTION_SENTINEL]

        if len(diagnoses) < n_diagnoses:
            logger.info(
                f"[NodeExpander] Synthesis parsed {len(diagnoses)}/{n_diagnoses} "
                f"diagnoses — retrying with a stricter output contract."
            )
            retry_prompt = (
                f"{prompt}\n\n"
                f"REMINDER: your previous answer could not be read. Reply with"
                f" EXACTLY {n_diagnoses} lines. Each line must be"
                f" '{DIFFERENTIAL_SENTINEL} <disease name>'. Write nothing else —"
                f" no introduction, no bold, no numbering, no explanation."
            )
            retry_output = self._call_llm(retry_prompt)
            if self.allow_abstention and detect_abstention(retry_output):
                logger.info("[NodeExpander] Synthesis declined on retry.")
                return [ABSTENTION_SENTINEL]
            retry_diagnoses = parse_differential(retry_output, limit=n_diagnoses)
            # Keep whichever attempt yielded more usable candidates, merging the
            # first attempt's answers in behind rather than discarding them.
            if len(retry_diagnoses) > len(diagnoses):
                merged = list(retry_diagnoses)
                seen = {d.lower() for d in merged}
                for d in diagnoses:
                    if d.lower() not in seen and len(merged) < n_diagnoses:
                        merged.append(d)
                        seen.add(d.lower())
                diagnoses = merged

        if not diagnoses:
            logger.warning(
                "[NodeExpander] Synthesis produced no parseable diagnosis after "
                "a retry. Returning an empty differential."
            )

        logger.info(f"[NodeExpander] Synthesis complete: {diagnoses}")
        return diagnoses
