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
import math
from typing import Optional

from apiro.graph.node import Node
from apiro.graph.edge import Edge
from apiro.config import (
    RAG_DOMAIN_FILTER,
    N_CHILD_HYPOTHESES,
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
        self._node_counter = 0

    def _generate_node_id(self, parent_id: str, index: int) -> str:
        """Deterministic child ID: {parent_id}_c{index}"""
        self._node_counter += 1
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

    def _call_llm_with_logprobs(self, prompt: str) -> tuple[str, list]:
        """Call the LLM client, requesting logprobs if supported."""
        if hasattr(self.llm_client, "generate_with_logprobs"):
            try:
                return self.llm_client.generate_with_logprobs(prompt)
            except Exception as e:
                logger.error(f"[NodeExpander] LLM call with logprobs failed: {e}")
        try:
            return self.llm_client.chat(prompt), []
        except Exception as e:
            logger.error(f"[NodeExpander] LLM call failed: {e}")
            return "", []

    def _align_logprobs_to_hypotheses(self, logprobs: list, hypotheses: list[str]) -> list[float]:
        """
        Align token logprobs to the parsed hypotheses.
        Returns a list of uncertainty/entropy scores (float) for each hypothesis.
        """
        DEFAULT = 0.693  # ln(2)
        if not logprobs or not hypotheses:
            return [DEFAULT] * len(hypotheses)
            
        lines_logprobs = []
        current_line = []
        for item in logprobs:
            token_text = item.get("token", "")
            logprob = item.get("logprob", 0.0)
            current_line.append(logprob)
            if "\n" in token_text:
                if current_line:
                    lines_logprobs.append(current_line)
                    current_line = []
        if current_line:
            lines_logprobs.append(current_line)
            
        scores = []
        for idx, hyp in enumerate(hypotheses):
            if idx < len(lines_logprobs) and lines_logprobs[idx]:
                line_lps = lines_logprobs[idx]
                avg_lp = sum(line_lps) / len(line_lps)
                p = math.exp(avg_lp)
                # Map token average probability to binary Shannon entropy
                p = max(0.5001, min(0.9999, p))
                entropy = - p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
                entropy = max(0.05, min(0.693, entropy))
                scores.append(round(entropy, 4))
            else:
                scores.append(DEFAULT)
                
        while len(scores) < len(hypotheses):
            scores.append(DEFAULT)
            
        return scores[:len(hypotheses)]

    # Regex to detect preamble/header lines that are NOT actual hypotheses.
    # LLMs often prefix their output with labels like "Hypotheses:" or "Output:"
    # which must be stripped before parsing.
    _PREAMBLE_RE = re.compile(
        r"^(hypothes[ie]s?|output|answer|diagnos[ie]s?|differential|results?|response|here are|the top|my top|list)\s*:?\s*$",
        re.IGNORECASE,
    )

    def _parse_hypotheses(self, llm_output: str, limit: int = 3, pad: bool = False) -> list[str]:
        """
        Parse the LLM's output into up to `limit` hypothesis strings.

        Defensive: strips preamble headers and numbering, drops empty lines.

        `pad` is False by default. Padding used to inject synthetic
        "[Expansion failed for: ...]" strings, which then entered the graph as
        real nodes: the entropy engine scores anything starting with "[" at
        ln(2), the maximum, so these placeholders sorted straight to the top of
        the depth>=1 frontier and got expanded ahead of genuine hypotheses,
        burning the node budget on nothing. An expansion that yields two usable
        hypotheses should return two.
        """
        lines = llm_output.strip().split("\n")
        hypotheses = []
        for line in lines:
            clean = re.sub(r"^\s*\d+\s*[\.)]\s*|^\s*[-*•]\s*", "", line.strip())
            clean = clean.strip()
            if not clean:
                continue
            # Skip preamble/header lines that are not actual clinical claims
            if self._PREAMBLE_RE.match(clean):
                continue
            # Skip degenerate fragments (bare punctuation, stray markdown fences)
            if len(clean) < 4 or clean.startswith("```"):
                continue
            hypotheses.append(clean)

        if len(hypotheses) > limit:
            hypotheses = hypotheses[:limit]

        if pad:
            while len(hypotheses) < limit:
                hypotheses.append(
                    f"[Expansion failed for: {hypotheses[0][:40] if hypotheses else 'unknown'}]"
                )

        return hypotheses

    def _batch_entropy(self, hypotheses: list[str], chunks: list[str]) -> list[float]:
        """
        Score entropy for all hypotheses concurrently.

        Uses the entropy engine's temperature_corrected_entropy on the pre-built
        verification prompt for each hypothesis. Results are collected concurrently
        but share the already-retrieved RAG chunks, avoiding redundant context
        construction and letting the caller reuse the same chunk list.

        Falls back to ln(2) (max binary uncertainty) on any failure so the node
        stays high-priority in the frontier.
        """
        DEFAULT = 0.693  # ln(2) — max binary uncertainty fallback
        
        def _score_hyp(hyp: str) -> float:
            try:
                prompt = self.entropy_engine._build_verification_prompt(hyp, chunks)
                val = self.entropy_engine.temperature_corrected_entropy(prompt)
                return val if val is not None else DEFAULT
            except Exception:
                return DEFAULT

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
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
        hypotheses = self._parse_hypotheses(raw_output)

        # Step 5a: Batch entropy — score all 3 hypotheses in one pass using
        # the verification signal (first-token Yes/No entropy). This is the
        # correct epistemic uncertainty measure — token logprobs from generation
        # measure generation fluency, not clinical decision-boundary certainty.
        entropies = self._batch_entropy(hypotheses, chunks)

        new_nodes = []

        for i, hypothesis in enumerate(hypotheses):
            entropy = entropies[i]

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


    def synthesize_differential(self, graph, top_k: int = 15, vignette: str = None) -> list[str]:
        """
        Synthesize a final differential diagnosis from the belief graph.

        The synthesizer is given three tiers of context, in this order:

          1. The patient's presentation (the vignette).
          2. Every confirmed depth-0 anchor — the deterministic axioms. These
             ARE the patient's facts; a differential written without them is
             guesswork.
          3. Exploration claims (depth >= 1), ranked by *diagnostic specificity*.

        Two bugs are fixed here, both of which handed the win to a plain
        zero-shot LLM:

          - The vignette was never passed in, so the bare-LLM baseline saw the
            whole case and Apiro saw only disembodied graph claims.
          - Nodes were ranked by entropy DESCENDING and truncated at top_k.
            Entropy in this engine is differential breadth (how many diagnoses
            a finding is compatible with), so descending order fed the LLM the
            15 *vaguest* claims in the graph and cut every specific one — and,
            because depth-0 axioms carry entropy ~0.01, it cut the patient's
            own findings first.

        Contradiction-penalised nodes are no longer silently deleted either.
        Soft-pruning is documented as "penalise, do not delete, so alternative
        hypotheses stay alive in the synthesis layer" — they are now listed
        last and explicitly labelled as disputed.

        Args:
            graph:     The BeliefGraph containing gathered evidence.
            top_k:     Max number of exploration claims passed to the LLM.
            vignette:  Raw (or axiom-enriched) clinical presentation.

        Returns:
            A list of up to 3 specific clinical diagnoses, most likely first.
        """
        logger.info("[NodeExpander] Synthesizing final differential diagnosis...")

        anchors: list[Node]   = []   # depth 0, affirmed
        ruled_out: list[Node] = []   # depth 0, negated
        explored: list[Node]  = []   # depth >= 1, clean
        disputed: list[Node]  = []   # depth >= 1, contradiction-penalised

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
                disputed.append(n)
            else:
                explored.append(n)

        # Rank exploration claims by diagnostic specificity: a claim compatible
        # with one diagnosis narrows the differential, a claim compatible with
        # ten does not. Deeper claims break ties — they are the payoff of the
        # traversal rather than a first-hop restatement of the seed.
        def specificity(n: Node) -> tuple:
            h = n.entropy_score if n.entropy_score is not None else 0.693
            return (round(h, 4), -n.depth)

        explored.sort(key=specificity)
        disputed.sort(key=specificity)

        top_nodes = explored[:top_k]
        if not top_nodes and not anchors:
            logger.warning("[NodeExpander] Graph is empty — nothing to synthesize.")
            return []

        logger.info(
            f"[NodeExpander] Synthesis context: {len(anchors)} anchors, "
            f"{len(top_nodes)}/{len(explored)} exploration claims, "
            f"{len(ruled_out)} ruled-out, {len(disputed)} disputed."
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
        anchors_text   = _bullets(anchors, seen)
        evidence_text  = _bullets(top_nodes, seen)
        disputed_text  = _bullets(disputed[:5], seen)
        ruled_out_text = _bullets(ruled_out, seen)

        sections = ["You are Apiro, a precise clinical differential-diagnosis engine.\n"]
        sections.append(
            "Your task: identify the single underlying disease that best explains this"
            " patient, and give the top 3 candidate diagnoses ranked by likelihood.\n"
        )

        clean_vignette = self._sanitize_vignette(vignette)
        if clean_vignette:
            sections.append(f"=== PATIENT PRESENTATION ===\n{clean_vignette}\n")
        if anchors_text:
            sections.append(
                "=== CONFIRMED FINDINGS (deterministically extracted — treat as ground truth) ===\n"
                f"{anchors_text}\n"
            )
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
        if disputed_text:
            sections.append(
                "=== DISPUTED CLAIMS (contradicted other evidence — weigh down, do not ignore) ===\n"
                f"{disputed_text}\n"
            )

        sections.append(
            "=== STRICT RULES ===\n"
            "1. Output exactly 3 diagnoses, one per line, most likely first.\n"
            "2. Give only the specific disease name (e.g. 'Type 1 autoimmune pancreatitis')."
            " No preamble, numbering, explanation, or mechanism.\n"
            "3. Name the specific underlying primary disease, not a syndrome or a symptom"
            " (e.g. prefer 'Pheochromocytoma' over 'Hypertensive crisis').\n"
            "4. Every diagnosis must be compatible with ALL confirmed findings above.\n\n"
            "=== OUTPUT (3 lines only) ==="
        )

        prompt = "\n".join(sections)

        raw_output = self._call_llm(prompt)
        diagnoses = self._parse_hypotheses(raw_output, limit=3)

        logger.info(f"[NodeExpander] Synthesis complete: {diagnoses}")
        return diagnoses
