# Improvements

A living reference for known issues in Apiro — both the ones that have been
fixed (with the reasoning, so they don't get silently reintroduced) and the
ones still open. `Log.md` is the chronological development diary; this file
is the topic-indexed version of the same material, plus a running list of
what's left.

## Fixed, with rationale (so it stays fixed)

### Premature saturation
`SaturationDetector` fires when a window of recent entropy scores has a low
mean, low variance, and a flat trend — meant to detect that the engine has
converged on an answer. Axioms are seeded at a fixed near-zero entropy
(~0.01), and the depth-aware frontier expands all seed nodes before any
generated hypothesis. With five or more axioms, the saturation window
filled entirely with 0.01 values — mean below theta, zero variance, flat
trend, all three conditions true — and the traversal halted after the seed
expansions, before exploring a single hypothesis.

Fix: `SaturationDetector(exploration_only=True)` only counts depth ≥ 1
expansions toward the window. See `apiro/graph/saturation.py`.

### Synthesis never saw the patient
Final synthesis ranked graph nodes by entropy *descending* and truncated to
15. Entropy here is differential breadth, so descending-entropy sort
surfaces the *vaguest* claims first — and since seeded axioms carry
entropy ≈ 0.01, the patient's own confirmed findings sorted last and were
always the ones cut. The raw vignette also wasn't passed to synthesis at
all, so a plain zero-shot baseline (which reads the whole case) had more
information than Apiro did.

Fix: synthesis now includes the raw vignette and prioritizes confirmed
findings and low-entropy (specific) nodes over high-entropy (vague) ones.
See `NodeExpander.synthesize_differential()` in `apiro/graph/expander.py`.

### The contradiction guardrail was dead code
`CONTRADICTION_THRESHOLD_EF` was 0.92 and the fast keyword pre-filter
returned exactly 0.92 for a match — every consumer checked `score > 0.92`,
which a value of exactly 0.92 never satisfies. No keyword-based
contradiction ever crossed the threshold, so the deterministic guardrail
the whole soft-pruning design depends on never fired.

Fix: the fast filter now returns `FAST_FILTER_CONTRADICTION_SCORE = 0.95`,
strictly above the threshold. See `apiro/graph/contradiction.py`.

### Soft-pruning ate live differentials, not just resolved ones
The contradiction penalty applied to *any* contradicting pair. Competing
differential diagnoses contradict each other by construction (that's what
makes them alternatives), so every detected pair removed one candidate —
picked by entropy, not by evidence — which is the exact failure mode
soft-pruning exists to prevent.

Fix: the penalty is scoped to pairs where one side is a confirmed/high-
confidence finding, not to any two competing hypotheses. See
`ContradictionDetector` and its callers in `apiro/graph/traversal.py`.

### Smaller fixes from the same audit
Rabbit-hole detection was graph-global, so it flagged the single
highest-entropy (most informative) frontier node whenever exploration got
interesting, aborting the traversal at `max_depth` on one deep branch;
`[Expansion failed...]` placeholders were scored at maximum entropy and so
were expanded ahead of real hypotheses; RAG-retrieved chunks were injected
under a "use ONLY what is stated here" instruction that starved synthesis
of the vignette itself; NER produced duplicate and non-clinical anchors;
the lab parser discarded values like "Hemoglobin of 9.5 g/dL" because "of"
got captured into the parsed name; negation leaked across sentence
boundaries; and the CLI read fields (`patient_context`, `ranked_hypotheses`)
that belonged to the hypothesis-testing engine purged in commit `8a001c7`,
so it raised `AttributeError` at the end of every run.

Regression coverage for all of the above: `tests/test_traversal_regressions.py`
(45 stub-only tests, no Ollama/ChromaDB/model download required).

### The C-NIAH harness did not parse on the project's minimum Python
`scripts/run_niah_eval.py` contained `f"{'length \\ depth':<16}"`. A literal
backslash inside an f-string expression is a `SyntaxError` before Python 3.12,
and `pyproject.toml` declares `requires-python = ">=3.10"`. The flagship
benchmark could not be imported, let alone run, on the interpreter the project
says it supports.

Fix: the label is built outside the f-string. See `_length_depth_matrix()`.

### A truncated, shadowed AURC function
`apiro/eval/calibration.py` defined `_aurc_from_ranking` twice at module level.
The first definition computed a sort order and then ended mid-body — no return
statement, so it evaluated to `None` — and was silently shadowed by a complete
second definition further down. Nothing failed, because Python takes the last
binding, but a reordering or a partial merge would have made
`compute_risk_coverage` return `float(None)`.

Fix: the dead stub is removed, and `tests/test_calibration.py` now pins AURC
against hand-computed values.

### Benchmark reporting bugs
Three defects in `run_niah_eval.py` that corrupted published output rather than
crashing:
- `f"{tokens}k"` labelled 8,000-token contexts as **"8000k"** — eight million
  tokens — in every report and in `data/niah_eval_results.json`.
- `case.get("depth_fraction") or case.get("depth")` treats `0.0` as absent, so
  a needle placed at the very top of the haystack — the control condition for
  the long-context claim — was bucketed as `"unknown"` and dropped from the
  matrix.
- Both matrix axes were sorted lexicographically, so `100%` printed before
  `25%` (and `16k` would print before `2k`).

### Missing import in the calibration CLI
`scripts/run_safety_calibration_eval.py` annotated with `Any` and `Callable`
without importing either. `from __future__ import annotations` meant the
annotations were never evaluated, so it ran — but every annotation in the file
was unresolvable to `typing.get_type_hints` and to any static checker.

### The child-hypothesis count was not configurable
`apiro/graph/expander.py` imported `config.N_CHILD_HYPOTHESES` and never used
it; the count was hard-coded as `_parse_hypotheses`' default `limit=3` and a
`ThreadPoolExecutor(max_workers=3)`. Changing the config value had no effect on
the engine. It is now a `NodeExpander(n_children=...)` argument defaulting to
the config value.

### Triplicated component wiring
`_build_real_components` and a private `_ChromaAdapter` were copy-pasted
verbatim into `run_pmc_eval.py` and `run_niah_eval.py`, and the copies had
already drifted (one passed a 120 s LLM timeout, the other the 90 s default) —
so two benchmarks were reporting comparable-looking numbers from stacks that
were not identical. Now shared via `apiro/eval/harness.py`.

### The C-NIAH stub arm ran against an empty graph
`build_niah_cases.py` emits no `seed_nodes` field, and stub mode skips axiom
extraction, so the Apiro arm received an empty frontier: the traversal returned
`stop_reason="no_frontier"` on iteration 1 and synthesised from an empty graph.
A guaranteed loss reported as a result. It now seeds the presentation itself
and logs a warning.

### Ungradeable ground truth in the PMC case set
`scripts/generate_pmc_cases.py` stored the model's raw stage-2 reply as
`target_diagnosis` with no validation. Four of the ten committed cases carry
multi-paragraph prose instead of a diagnosis label (Case 9 begins *"Here is the
acute, primary presenting diagnosis:\n\nAppendicitis\n\nHowever, it's worth
noting that..."*). `_check_synthesis_hit` normalises the entire blob before
matching, so those cases cannot be scored correctly for any arm and depress
every reported PMC accuracy. The generator now post-processes the label and
drops cases where nothing usable survives; **`data/pmc_cases.json` still needs
regenerating.**

### Documented results that the artifacts contradict
The README reported the PMC bare-LLM arm at 20% (that is the five-case figure;
the ten-case run in `data/latest_pmc_benchmark_output.txt` shows 1/10 = 10%),
and claimed Apiro's "sole win on Case 4 (Colon Adenocarcinoma)" — in the
captured run Apiro *failed* Case 4. Both corrected, with the discrepancy noted
in place rather than quietly overwritten.

### Apiro was answering with 1.3 candidates against baselines offering 7
The largest single defect found, and a measurement one rather than a reasoning
one. On the committed C-NIAH run, the Apiro arm's differential looked like:

    ['*Diagnosis 1:**', 'Acute Myeloid Leukemia (AML)', '*Diagnosis 2:**']

43 of 75 answer slots (57%) held markdown scaffolding or preamble instead of a
diagnosis; five cases held none at all, twelve held exactly one. The cause was
`_parse_hypotheses` stripping a single leading bullet character (so
`**Diagnosis 1:**` became `*Diagnosis 1:**` and was admitted as an answer) with
a `$`-anchored preamble regex that only caught bare headers.

Compounding it, the harnesses graded the baselines over *every non-empty line*
of raw output — uncapped, averaging 7.2 candidates per case and reaching 17 —
while capping Apiro at 3 parsed slots. Roughly a five-fold difference in how
many guesses each arm was allowed.

Fix: `apiro/parsing.py`, one parser shared by the engine and every arm, at a
common `config.N_DIFFERENTIAL`; a `DX:` output sentinel in the synthesis prompt
with one retry on an under-filled parse; and an `n_candidates` field per arm in
every per-case record so the asymmetry cannot return unnoticed. Replayed over
the committed outputs this lifts usable diagnoses from 18/75 to 52/75 and cuts
zero-diagnosis cases from 15 to 1.

**Every published number predates this fix and must be regenerated.**

### Smaller cleanups
Unused imports across 18 modules; three config constants nothing read
(`MAX_TRAVERSAL_DEPTH`, `MAX_NODES_PER_RUN`, `EVAL_EXCLUDE_SEED_HITS`); a dead
`_node_counter` on `NodeExpander`; a dead `_print_evaluator_summary` in the NIAH
harness that also mis-called `_print_summary` with the wrong argument type; an
unread `real_entropy` field on the web API's request model; a `global`
statement in `BeliefGraph._get_embedder` that declared a name it never assigned
while reading and writing `globals()` by hand; `investigate.py`'s `--output`
flag documented but never implemented (now wired to `BeliefGraph.export_json`)
and its `--real-entropy` flag removed; `run_eval.sh` sourcing a `venv/` path
that does not exist and describing the run as "HADCE", an engine purged in July
2026; and a `.gitignore` whose `data/*.json` rule silently excluded every
benchmark case set and results file the README cites.

## Open items

These are real, currently unresolved, roughly in order of how much they'd
matter to someone reading the code closely:

- **The "Shannon entropy" and "NLI" framing overstate what runs.** The live
  entropy signal (`apiro/entropy/engine.py`) asks the LLM to self-report a
  count of plausible diagnoses and maps it through a fixed table — a
  reasonable heuristic, not entropy computed over a token probability
  distribution. Contradiction detection is a keyword/antonym pre-filter
  plus an LLM yes/no judge, not a cross-encoder NLI model. Both are
  documented accurately in code comments; the README should read the same
  way.
- **`data/axiom_weights.yaml` is a small, hand-curated list** (~20 entities)
  keyed to findings that appear in the benchmark cases used during
  development. Anything not on the list gets a flat default weight, so the
  weighting mechanism does more work on the eval set than it will on novel
  vignettes. Growing this list (or replacing it with something that
  generalizes, e.g. a lightweight learned specificity score) is the next
  step if seed-selection quality needs to scale past the current corpus.
- **`apiro/eval/evaluator.py`'s clinical synonym groups** are similarly
  built around the specific diagnoses in the benchmark sets. Legitimate
  for grading, but worth disclosing alongside any reported accuracy number.
- **`apiro/axioms/lab_parser.py` is regex-based** and has already needed
  several rounds of fixes for phrasing it didn't anticipate (see the
  "Hemoglobin of 9.5 g/dL" fix above). Expect more of the same as the
  corpus grows; there's no principled parser here, just an accumulating
  set of patterns.
- **`apiro/patient/context.py` is unused, and now explicitly marked so.** A
  complete module for structured patient-context extraction, built for the
  hypothesis-testing architecture purged in `8a001c7`. It carries a
  deprecation banner and is retained on purpose: the open question is whether
  seed selection would be better anchored on structured fields (age, sex, a
  parsed lab dict) than on free text, and this is the working implementation
  of that idea. Decide the question or delete the module; do not add callers
  in the meantime.
- **`apiro/corpus/mimic_adapter.py` has no benchmark wired to it.** It is
  complete and unit-tested (`tests/test_mimic_adapter.py`) but no harness
  calls it — the same state `clinical_case_adapter.py` was in before the
  CUPCase benchmark. A MIMIC discharge-diagnosis benchmark is the obvious
  next external evaluation.

### Benchmark validity — the largest open item

- **All published numbers predate the parsing/grading fix** and are not a valid
  baseline. Re-run per `docs/BENCHMARKING.md` before comparing anything.
- **Every published comparison is underpowered.** Recomputed from
  `data/niah_eval_results.json`: Apiro vs Standard RAG on C-NIAH is +28.0 pp
  with an exact McNemar p of **0.119**, and a paired bootstrap CI on the delta
  of [+0.0, +56.0] pp. Apiro vs the bare LLM is p = 0.549. The PMC set is
  N = 10 with intervals over thirty points wide. Nothing in the README's
  tables is statistically established; the direction is consistent and the
  sample sizes are too small to confirm it. Generating 60–100+ C-NIAH cases
  (`--num-cases`) is the cheapest fix: power is 35% at N = 25, 87% at N = 75 and
  95% at N = 100. `NEEDLE_BANK` was widened from 6 to 20 diagnoses so that
  scaling N produces genuinely independent cases rather than pseudo-replicates —
  without that, a larger N would have manufactured significance rather than
  measured it.
- **C-NIAH is self-authored end to end.** The cases, the needles, the
  distractors and the stub responses that recover them all come from
  `scripts/build_niah_cases.py` in this repository. It is a good instrumented
  probe that the mechanism fires; it is not independent evidence. The CUPCase
  benchmark (`scripts/run_cupcase_eval.py`) exists to supply the external
  counterpart — **it has been implemented but not yet run**, so there are no
  results for it.
- **`data/pmc_cases.json` needs regenerating.** Four of ten ground-truth
  labels are unusable prose (see the fixed-items section). The generator is
  fixed; the data is not.

### Calibration

- **The confidence signal is a placeholder, and the published ECE says so.**
  `scripts/run_safety_calibration_eval.py` derives Apiro's confidence from
  traversal signals with hand-set coefficients, and the baselines' from output
  length and hedging words; the script marks both
  `# >>> ASSUMPTION (REPLACE WITH REAL MODEL) <<<`. The numbers in
  `data/calibration_eval_results.json` are now published in the README, and
  they are poor: Apiro's ECE is 0.452 — the worst of the three arms — driven
  by severe *under*-confidence (mean confidence 0.228 against 68% accuracy).
  At the documented operating point of τ = 0.65 Apiro abstains on **all 25
  cases**. The AURC result (0.119 vs 0.536 for RAG) is the one figure that
  survives, and it is a statement about confidence *ranking*, not about
  calibrated probability. A fitted calibrator, and a τ re-derived from it, is
  the prerequisite for any safety claim.
