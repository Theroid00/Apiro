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
        elif selected_mode == "investigator":
            result = self._run_investigator(
                narrative,
                n_diagnoses=n_diagnoses,
                allow_abstention=allow_abstention,
                on_event=on_event,
            )
        else:
            raise ValueError(f"unsupported new-engine mode: {selected_mode}")

        if scheduler is not None:
            result.model_telemetry = scheduler.delta(telemetry_before)
        return result

    def _run_simple(
        self, narrative: str, *, n_diagnoses: int, allow_abstention: bool, on_event
    ):
        return self._build_reasoner(
            "simple", n_diagnoses=n_diagnoses, allow_abstention=allow_abstention
        ).run(narrative, on_event=on_event)

    def _run_investigator(
        self, narrative: str, *, n_diagnoses: int, allow_abstention: bool, on_event
    ):
        return self._build_reasoner(
            "investigator", n_diagnoses=n_diagnoses,
            allow_abstention=allow_abstention,
        ).run(narrative, on_event=on_event)

    def _build_reasoner(self, mode: str, *, n_diagnoses: int, allow_abstention: bool):
        """Build one bounded new-engine pipeline with shared dependencies."""
        from apiro.graph.contradiction import ContradictionDetector
        from apiro.reasoning.investigator import InvestigatorReasoner
        from apiro.reasoning.simple import SimpleReasoner

        contradiction = ContradictionDetector(
            model=self.resources.model,
            ollama_url=self.resources.ollama_url,
            scheduler=getattr(self.resources, "model_scheduler", None),
        )
        reasoner_type = {
            "simple": SimpleReasoner,
            "investigator": InvestigatorReasoner,
        }.get(mode)
        if reasoner_type is None:
            raise ValueError(f"unsupported new-engine mode: {mode}")
        return reasoner_type(
            embedder=self.resources.embedder,
            llm_client=self.resources.llm_client,
            axiom_extractor=self.resources.axiom_extractor,
            contradiction_detector=contradiction,
            n_diagnoses=n_diagnoses,
            allow_abstention=allow_abstention,
        )

    @staticmethod
    def _validate_mode(mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in {"simple", "investigator"}:
            raise ValueError(
                "reasoning mode must be 'simple' or 'investigator'"
            )
        return normalized
