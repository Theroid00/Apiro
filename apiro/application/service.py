"""Canonical application service for all Apiro investigation entry points."""

from __future__ import annotations

from apiro.config import N_DIFFERENTIAL, REASONING_MODE
from apiro.reasoning.models import InvestigationResult


class InvestigationService:
    """Select an engine while keeping setup, results, and telemetry uniform."""

    def __init__(self, resources, *, default_mode: str = REASONING_MODE):
        self.resources = resources
        self.default_mode = self._validate_mode(default_mode)

    def investigate(
        self,
        narrative: str,
        *,
        mode: str | None = None,
        n_diagnoses: int = N_DIFFERENTIAL,
        max_depth: int = 6,
        case_name: str = "investigation",
        log_dir=None,
        allow_abstention: bool = False,
        on_event=None,
    ) -> InvestigationResult:
        if not narrative or not narrative.strip():
            raise ValueError("Clinical findings must not be empty")

        selected_mode = self._validate_mode(mode or self.default_mode)
        scheduler = getattr(self.resources, "model_scheduler", None)
        telemetry_before = scheduler.snapshot() if scheduler is not None else None

        if selected_mode == "simple":
            result = self._run_simple(
                narrative,
                n_diagnoses=n_diagnoses,
                allow_abstention=allow_abstention,
                on_event=on_event,
            )
        else:
            result = self._run_legacy(
                narrative,
                n_diagnoses=n_diagnoses,
                max_depth=max_depth,
                case_name=case_name,
                log_dir=log_dir,
                allow_abstention=allow_abstention,
                on_event=on_event,
            )

        if scheduler is not None:
            result.model_telemetry = scheduler.delta(telemetry_before)
        return result

    def _run_simple(
        self, narrative: str, *, n_diagnoses: int, allow_abstention: bool, on_event
    ):
        from apiro.graph.contradiction import ContradictionDetector
        from apiro.reasoning.simple import SimpleReasoner

        contradiction = ContradictionDetector(
            model=self.resources.model,
            ollama_url=self.resources.ollama_url,
            scheduler=getattr(self.resources, "model_scheduler", None),
        )
        return SimpleReasoner(
            embedder=self.resources.embedder,
            llm_client=self.resources.llm_client,
            axiom_extractor=self.resources.axiom_extractor,
            contradiction_detector=contradiction,
            n_diagnoses=n_diagnoses,
            allow_abstention=allow_abstention,
        ).run(narrative, on_event=on_event)

    def _run_legacy(
        self,
        narrative: str,
        *,
        n_diagnoses: int,
        max_depth: int,
        case_name: str,
        log_dir,
        allow_abstention: bool,
        on_event,
    ) -> InvestigationResult:
        from apiro.axioms.seeding import build_seeds
        from apiro.graph.belief_graph import BeliefGraph

        traversal = self.resources.create_traversal(
            n_diagnoses=n_diagnoses,
            allow_abstention=allow_abstention,
            log_dir=log_dir,
        )
        graph = BeliefGraph()
        seeds, axioms, enriched = build_seeds(
            narrative, self.resources.axiom_extractor
        )
        legacy_result = traversal.run(
            seed_nodes=seeds,
            graph=graph,
            max_depth=max_depth,
            case_name=case_name,
            vignette=enriched,
            on_event=on_event,
        )
        return InvestigationResult.from_legacy(legacy_result, axioms=axioms)

    @staticmethod
    def _validate_mode(mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in {"simple", "legacy"}:
            raise ValueError("reasoning mode must be 'simple' or 'legacy'")
        return normalized
