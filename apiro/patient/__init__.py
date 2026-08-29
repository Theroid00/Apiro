"""apiro/patient — structured patient representation from clinical vignettes.

.. deprecated::
    DEPRECATED AND UNUSED. No production code path imports this package; it
    survives from the hypothesis-testing architecture purged in commit
    8a001c7. Kept as the starting point for structured-field seed selection
    (see docs/IMPROVEMENTS.md), not as part of the live pipeline. Do not add
    new callers.
"""
from apiro.patient.context import PatientContext, extract_patient_context

__all__ = ["PatientContext", "extract_patient_context"]
