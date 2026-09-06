"""
graph/critic.py
------------------
Global Critic to dynamically halt graph traversal once sufficient diagnostic
evidence has been accumulated.
"""

import logging
from apiro.graph.belief_graph import BeliefGraph

logger = logging.getLogger(__name__)

CRITIC_PROMPT_TEMPLATE = """\
You are an expert clinical diagnostician overseeing an automated diagnostic reasoning engine.
The engine has been generating hypotheses based on the patient's case presentation.

=== PATIENT CLINICAL PRESENTATION ===
{case_context}

=== CURRENT TOP HYPOTHESES ===
{top_nodes}

Your task: Evaluate if there is sufficient evidence to confidently halt the search and declare a specific primary etiology (underlying disease), or if critical diagnostic information is still missing.
Are the current top hypotheses specific and comprehensive enough to identify the exact underlying primary cause of the patient's presentation without needing further exploration?

IMPORTANT: Do NOT halt if the hypotheses only describe secondary clinical syndromes (e.g., congestive heart failure, pulmonary edema, respiratory failure, acute kidney injury) or generic symptoms without pinpointing the specific primary disease causing them.

Answer YES or NO on the first line.
On the second line, provide a brief 1-sentence reason.
"""

class CriticEngine:
    def __init__(self, llm_client):
        self.llm = llm_client

    def evaluate_halting(self, graph: BeliefGraph, vignette: str = None, top_k: int = 5) -> bool:
        """
        Evaluate if the graph has converged on a diagnosis.
        Returns True if the search should halt, False otherwise.
        """
        if not vignette:
            return False
            
        # Filter to generated hypotheses (depth > 0) that are not rabbit holes or heavily penalized
        candidates = [
            n for n in graph.nodes.values()
            if n.depth > 0 and not n.is_rabbit_hole and getattr(n, "contradiction_penalty", 0.0) < 0.5
        ]
        # Sort by entropy ascending (lowest entropy/uncertainty first = most certain)
        candidates.sort(key=lambda n: n.entropy_score if n.entropy_score is not None else 1.0)
        top_nodes = candidates[:top_k]
        
        if not top_nodes:
            return False
            
        nodes_text = "\n".join(f"- {n.claim}" for n in top_nodes)
        
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            case_context=vignette,
            top_nodes=nodes_text
        )
        
        try:
            # No LLM client in this codebase accepts max_tokens/temperature
            # kwargs on generate() (they're fixed at construction time), and
            # StubLLMClient — used throughout the test suite — only exposes
            # chat(), not generate(). Use whichever the client actually has.
            if hasattr(self.llm, "generate"):
                response = self.llm.generate(prompt)
            else:
                response = self.llm.chat(prompt)

            first_line_clean = response.strip().split('\n')[0].strip().upper()
            first_word = first_line_clean.split()[0].strip(".,!?\"'") if first_line_clean.split() else ""
            
            if first_word == "YES":
                logger.info(f"[GlobalCritic] Halting approved. Reason: {response.strip().split(chr(10))[1:]}")
                return True
            else:
                logger.debug(f"[GlobalCritic] Halting rejected. Reason: {response.strip().split(chr(10))[1:]}")
                return False
        except Exception as e:
            logger.error(f"[GlobalCritic] Failed to evaluate halting: {e}")
            return False
