"""
axioms/seeding.py — axioms → belief-graph seed nodes
=====================================================

The CLI, the web app and the evaluation harness each built their own seed
nodes from the extracted axioms, and they disagreed:

  * investigate.py / app.py assigned every axiom entropy 0.01, so a negated
    finding ("the patient denies chest pain") became just as certain and just
    as high-priority an anchor as a hard lab value;
  * run_pmc_eval.py derived entropy from the axiom weight and polarity.

Same engine, three behaviours, and only the eval path was ever measured. This
module is the single implementation.
"""

from __future__ import annotations

from apiro.graph.node import Node
from apiro.axioms.models import ClinicalAxiom

# Base certainty of a deterministically extracted, affirmed finding.
SEED_BASE_ENTROPY = 0.01
# Low-weight (non-specific) axioms are not as informative as a cardinal
# finding, so they start marginally less certain.
SEED_WEIGHT_SPREAD = 0.15
# A negated or historical finding constrains the differential far less than an
# affirmed one, and must not sit at the very front of the frontier.
SEED_NEGATED_FLOOR = 0.40
SEED_HISTORICAL_FLOOR = 0.30


def seed_entropy(axiom: ClinicalAxiom) -> float:
    """Entropy for an axiom's seed node, by diagnostic weight and polarity."""
    weight = float(getattr(axiom, "weight", 0.5) or 0.0)
    base = SEED_BASE_ENTROPY + (1.0 - weight) * SEED_WEIGHT_SPREAD
    polarity = getattr(axiom, "polarity", "affirmed")
    if polarity == "negated":
        return round(max(SEED_NEGATED_FLOOR, base + 0.3), 4)
    if polarity == "historical":
        return round(max(SEED_HISTORICAL_FLOOR, base + 0.2), 4)
    return round(base, 4)


def axioms_to_seed_nodes(axioms: list[ClinicalAxiom]) -> list[Node]:
    """Build depth-0 seed nodes, carrying the axiom weight into the frontier."""
    seeds: list[Node] = []
    for ax in axioms:
        seeds.append(Node(
            id=ax.id,
            claim=ax.text,
            entropy_score=seed_entropy(ax),
            domain=ax.domain,
            depth=0,
            metadata={
                "axiom_weight": float(getattr(ax, "weight", 0.0) or 0.0),
                "polarity":     getattr(ax, "polarity", "affirmed"),
                "confidence":   float(getattr(ax, "confidence", 1.0) or 1.0),
            },
        ))
    return seeds


def enrich_vignette(vignette: str, axioms: list[ClinicalAxiom]) -> str:
    """Append the deterministic findings so the LLM reads the cleaned facts."""
    if not axioms:
        return vignette
    lines = "\n".join(f"- {ax.text}" for ax in axioms)
    return f"{vignette}\n\n[Deterministic Clinical Findings]\n{lines}\n"


def build_seeds(vignette: str, extractor) -> tuple[list[Node], list[ClinicalAxiom], str]:
    """
    Extract axioms and turn them into seed nodes plus an enriched vignette.

    Guarantees at least one seed: if extraction yields nothing (NER model
    unavailable, terse free-text input), the raw presentation itself is seeded.
    Previously an empty axiom list produced an empty frontier, so the traversal
    stopped instantly with stop_reason="no_frontier" and synthesised a
    differential from an empty graph — an automatic loss against any baseline.
    """
    axioms = extractor.extract(vignette)
    seeds = axioms_to_seed_nodes(axioms)
    if not seeds:
        seeds = [Node(
            id="ax_fallback_0",
            claim=f"The patient presents with the following clinical picture: {vignette.strip()[:400]}",
            entropy_score=SEED_BASE_ENTROPY,
            domain="pathophysiology",
            depth=0,
            metadata={"axiom_weight": 1.0, "polarity": "affirmed", "fallback": True},
        )]
    return seeds, axioms, enrich_vignette(vignette, axioms)
