# Apiro Accuracy Audit — `feature/apiro-accuracy-fixes`

A full static read of the engine (`apiro/graph`, `apiro/axioms`, `apiro/entropy`,
`apiro/eval`, `scripts/`) looking for one thing: **why does Apiro score 30–40%
on `pmc_cases.json` when a bare zero-shot LLM on the same model scores similar
or better?**

Nothing here was run against Ollama or ChromaDB — this machine has neither.
Every finding is traced from code, and every fix is covered by a stub-only
regression test in `tests/test_traversal_regressions.py` (45 tests, no model
downloads, no network).

---

## TL;DR — the four that decide the benchmark

| # | Finding | Effect on accuracy |
|---|---------|--------------------|
| 1 | Saturation fired on the engine's own seed nodes | **Traversal halted after ~5 seed expansions on every case.** Apiro was one round of RAG wearing a graph. |
| 2 | Synthesis ranked nodes by entropy *descending* and truncated at 15 | The final prompt got the 15 **vaguest** claims in the graph and **none of the patient's confirmed findings**. The baseline reads the whole case; Apiro read fragments. |
| 3 | Fast-filter contradictions scored exactly at the threshold, compared with `>` | The **deterministic guardrail never fired once**. The entire Hybrid Apiro justification was dead code. |
| 4 | Soft-pruning penalised any contradicting pair | Every detected hypothesis-vs-hypothesis contradiction **deleted one live differential**, chosen by entropy rather than evidence. |

Fix 1 alone changes what the engine does; fixes 2–4 change whether that work
reaches the answer.

---

## 1. Premature saturation on deterministic seeds

`apiro/graph/saturation.py`, `apiro/graph/traversal.py`

Hybrid Apiro injects every extracted axiom as a depth-0 seed with a fixed
entropy of ~0.01, and `get_frontier(depth_aware=True)` scores depth-0 nodes at
`2.0 - H`, so **all seeds expand before any hypothesis**. `SaturationDetector`
then read the last 5 expansions:

```
recent      = [0.01, 0.01, 0.01, 0.01, 0.01]
avg 0.01    < theta 0.55         ✓ low_entropy
variance 0.0 < max_variance 0.04 ✓ low_variance
trend 0.0   <= 0                 ✓ non_rising
                                 → SATURATED
```

Any case producing ≥ 5 axioms — i.e. every real case — halted with
`stop_reason="saturation"` after five seed expansions, before a single
generated hypothesis was ever explored. This is consistent with the reported
"~28 s per case": the engine was not doing the work the architecture describes.

**Fix.** Saturation is measured over exploration expansions only
(`depth >= 1`), with a warm-up floor (`SATURATION_MIN_EXPLORATION = 8`)
enforced in the traversal regardless of which detector is injected.
`BeliefGraph.get_recent_entropies/get_entropy_trend` take `min_depth`, and
`count_expansions(min_depth=...)` was added. The detector's original
depth-blind math is preserved behind `exploration_only=False` so the published
unit tests still pin it.

> **Expect longer runs.** This is the fix that makes Apiro actually traverse.
> `MAX_EXPLORATION_EXPANSIONS` (default 24) is the wall-clock knob.

---

## 2. The synthesizer never saw the patient

`apiro/graph/expander.py::synthesize_differential`

Two compounding problems:

* The vignette was never passed in. The bare-LLM arm is prompted with the full
  case; Apiro's final answer was argued from graph claims alone.
* Nodes were sorted **by entropy descending** and cut at `top_k=15`. Entropy
  here is *differential breadth* — how many diagnoses a finding is compatible
  with — so descending order selects the **least** discriminating claims. And
  because depth-0 axioms carry entropy ~0.01, the patient's own confirmed
  findings sorted last and were always truncated away.

Contradiction-penalised nodes were also dropped entirely, though the project
log describes soft-pruning as "a penalty rather than outright deletion... to
keep alternative hypotheses alive in the synthesis layer".

**Fix.** The prompt is now built in tiers — presentation → confirmed anchors →
ruled-out findings → reasoning trace ranked by *specificity* (entropy
ascending, depth as tie-break) → disputed claims, labelled as disputed rather
than deleted. Anchors and negated findings are never truncated.

---

## 3. The deterministic guardrail could not fire

`apiro/graph/contradiction.py`, `apiro/graph/traversal.py`

`_fast_filter` returned `score=0.92`. `CONTRADICTION_THRESHOLD_EF` is `0.92`.
Every consumer tested `score > CONTRADICTION_THRESHOLD_EF`. `0.92 > 0.92` is
`False`, so **no keyword or negation contradiction ever crossed the bar** —
only the LLM judge (0.95) could. The hyperkalemia-vs-hypokalemia example that
motivates the whole architecture would have been detected and then discarded.

**Fix.** Named constant `FAST_FILTER_CONTRADICTION_SCORE = 0.95`, comparison
changed to `>=`, and antonym matching switched to word boundaries — plain
substring matching fired `low` inside "blood f**low**", `high` inside
"**high**ly", `left` inside "c**left**", `mg` inside "mg/dL". Harmless while
the guardrail was dead; a false-prune source once it fires.

`check_batch` also stopped being a sequential list comprehension: cache and
fast-filter hits resolve inline and only genuinely ambiguous pairs go to the
LLM judge, concurrently.

---

## 4. Soft-pruning removed live differentials

`apiro/graph/traversal.py`

The penalty applied to *any* contradicting pair, penalising the deeper or
higher-entropy node. Competing differentials contradict each other by
construction, so each detected pair eliminated one alternative from the
frontier and the synthesis — selected by entropy, not by evidence.

**Fix.** The penalty applies only when one side is a deterministic depth-0
anchor, which is what the design claims: a hypothesis contradicting a real lab
value gets pruned. Hypothesis-vs-hypothesis contradictions are flagged and
logged but not penalised. Two anchors contradicting each other is reported as a
data problem instead of penalising one at random. Contradiction candidates are
restricted to anchors and already-expanded claims, so three hypotheses
generated from one prompt are no longer checked against each other.

---

## 5. Rabbit-hole detection killed the best node available

`apiro/graph/rabbit_hole.py`

`check()` read the last N expansions **anywhere in the graph** and, if that
global curve was rising, flagged whatever node the traversal had just picked.
The traversal picks the highest-entropy depth ≥ 1 node, and a rising global
entropy curve is exactly what a fresh batch of open questions looks like — so
the detector systematically flagged the most informative node on the frontier
and removed it permanently from the traversal *and* the synthesis
(`is_rabbit_hole` excludes it from both), with no reference to whether that
node's own reasoning path had degenerated.

**Fix.** The reversal is measured on the node's ancestor chain. Nodes with no
lineage keep the global behaviour, which is what the existing unit tests build.

---

## 6. Traversal aborted on a single deep node

`apiro/graph/traversal.py`

```python
if frontier[0].depth >= max_depth:
    break          # ends the ENTIRE run
```

The top-scoring node being at the depth limit ended the whole traversal and
discarded an otherwise healthy frontier. **Fix:** expand the best node still
eligible; stop only when none are.

---

## 7. Placeholder hypotheses were expanded as real nodes

`apiro/graph/expander.py::_parse_hypotheses`

Parsing padded short LLM output to three entries with
`"[Expansion failed for: ...]"`. Those strings became graph nodes, and
`EntropyEngine.differential_breadth_entropy` returns `_DEFAULT_HIGH` (ln 2, its
maximum) for any claim starting with `[` — so placeholders sorted to the **top**
of the exploration frontier and were expanded ahead of real hypotheses.

**Fix.** Padding is opt-in and off; an expansion yielding two usable hypotheses
returns two.

---

## 8. RAG injected nearest-neighbour noise as evidence

`apiro/graph/expander.py`

ChromaDB always returns top-k however distant, and the prompt says
*"RETRIEVED EVIDENCE (use ONLY what is stated here)"*. A rare-disease node was
handed six confidently formatted passages about something else and steered
away from the answer parametric knowledge would have reached.
`RAG_MIN_CHUNKS_FOR_GROUNDING` could never trigger the parametric fallback,
because a populated corpus always returns k chunks.

**Fix.** Chunks beyond `RAG_MAX_DISTANCE` (0.65) are dropped and the parametric
fallback engages as designed; adapters pass distances through. Vector queries
also strip the axiom extractor's forged-sentence scaffolding — every seed was
embedding partly as *"the patient presents with the clinical finding of"*,
pulling seeds toward each other and away from the passage describing the actual
finding.

---

## 9. The deterministic anchors were noisy

`apiro/axioms/*`

Everything here becomes an unprunable "absolute certainty" node, so noise is
expensive.

* **NER** had no confidence floor, no entity-type filter and no deduplication.
  A vignette mentioning "chest pain" four times produced four identical
  anchors, and non-clinical groups produced anchors like *"The patient presents
  with the clinical finding of 45-year-old."* → confidence threshold,
  diagnostic-group whitelist mapped to real Apiro domains, dedup, and per-type
  sentence templates.
* **Lab parsing** rejected the whole match if any captured word was a stopword.
  The name group greedily takes the words before the number, so
  *"Hemoglobin of 9.5 g/dL"* and *"Troponin was 5.2 ng/mL"* — the exact
  measurements this engine is built on — were **thrown away**. → filler is
  trimmed instead of rejecting the match, blood pressure is parsed before the
  general pattern with span claiming (no more junk `BP 88 /`), punctuation-only
  units and social-history quantities are rejected, repeats collapse.
* **Negation** used a flat 45-character lookback that crossed sentence
  boundaries, so *"No fever. Severe epigastric pain"* recorded the **pain** as
  denied — and a negated axiom tells the synthesizer any diagnosis requiring it
  is wrong. It also only inspected the first occurrence. → scope clipped at the
  nearest clause boundary, all occurrences considered, affirmed-anywhere wins.
* `ClinicalAxiom` now carries `raw_text`/`confidence`, so three modules stop
  recovering the entity by slicing a hard-coded sentence prefix.
* `AxiomExtractor.extract` caps the seed set at `MAX_SEED_NODES` (20) by
  diagnostic weight, always keeping measurements.

---

## 10. Three different seeding behaviours

`scripts/investigate.py`, `scripts/app.py`, `scripts/run_pmc_eval.py`

The CLI and web app hard-coded `entropy_score=0.01` for **every** axiom, so a
denied finding anchored the frontier exactly as hard as a positive lab value.
The eval harness derived entropy from weight and polarity. The engine behaved
differently in the UI than in the benchmark it is judged by.

**Fix.** `apiro/axioms/seeding.py` is the single implementation, and it
guarantees a seed: an empty axiom list previously left an empty frontier, so
the traversal stopped instantly and synthesised from an empty graph — an
automatic loss.

---

## 11. Crashes and dead code

* `scripts/investigate.py::print_report` read `result.patient_context` and
  `result.ranked_hypotheses` off a `TraversalResult` that has never had either
  (they belonged to the hypothesis-testing engine purged in `8a001c7`) —
  **AttributeError at the end of every CLI run**.
* `apiro/run.py::generate_with_logprobs` used `requests` while the import sat
  inside `generate()` — `NameError` on call.
* `NodeExpander.expand` appended children the graph had rejected, so the
  traversal contradiction-checked and logged nodes that do not exist.
* Semantic DAG merging could merge a child into its own parent (self-loop, the
  expansion silently discarded) or into a depth-0 axiom (hypothesis replaced by
  a fact already held).
* The expander ran a full O(N) contradiction pass that the traversal
  immediately recomputed batched — double LLM-judge calls for the same flag.
* `EntropyEngine` treated an unparseable count as a dead server, spending the
  retry budget and defaulting to ln 2 — promoting the claim to the top of the
  frontier.

---

## Algorithmic upgrade: relevance-weighted exploration

Beyond the bugs, one design issue limits how well this can ever score.

`differential_breadth_entropy(claim)` measures how many diagnoses a claim is
compatible with, **from the claim alone**. It ignores `context_chunks`, ignores
the patient, and is cached by claim text. So the entropy-first frontier ranks
*"chest pain has many causes"* (0.693) above *"aquaporin-4 antibodies indicate
NMOSD"* (0.10): the engine spends its budget widening the differential rather
than resolving it — the opposite of a detective.

`BeliefGraph.set_case_anchor()` embeds the presentation once and exploration
priority becomes:

```
priority = H(claim) * (RELEVANCE_FLOOR + (1 - RELEVANCE_FLOOR) * cos(claim, case))
```

Uncertainty is still what drives exploration, but uncertainty *about this
patient*. With no anchor set, the frontier is bit-identical to before, so this
is opt-in. Seed ties (all axioms share a near-fixed entropy) break on the
axiom's diagnostic weight, so the sharpest anchor expands first.

`RELEVANCE_FLOOR = 1.0` disables it if you want to A/B the pure entropy-first
frontier.

---

## What to check on the machine with Ollama + ChromaDB

```bash
python scripts/run_pmc_eval.py --real --out data/eval_after.json
```

1. **`stop_reason` distribution.** Before this branch every case should read
   `saturation`. After, expect `exploration_budget`, `saturation`,
   `critic_halt` or `max_depth`. If you still see `saturation` on every case
   with `explored_nodes` in single digits, something is re-introducing seed
   entropies into the window.
2. **`explored_nodes`.** Should be > 8 per case. This is the single number that
   says whether the engine ran at all.
3. **Latency.** Will rise — the engine now does the traversal it always claimed
   to. Tune `MAX_EXPLORATION_EXPANSIONS` (24) and `MAX_SEED_NODES` (20).
4. **`RAG_MAX_DISTANCE`.** 0.65 is a starting point for `all-mpnet-base-v2`
   cosine distance. Log a run's distances and set it where relevant chunks
   separate from noise; too tight sends everything to parametric mode.
5. **Head-to-head counts.** "Apiro wins where bare LLM fails" is the number
   that matters. A high "bare LLM wins where Apiro fails" with healthy
   traversal stats points at the synthesis prompt, not the traversal.

Tuning order if accuracy is still short: `RELEVANCE_FLOOR` →
`MAX_EXPLORATION_EXPANSIONS` → `RAG_MAX_DISTANCE` → `THETA_BY_DOMAIN`.

---

## Not fixed (deliberate)

* **`_check_synthesis_hit` grades with the same model being evaluated.** An
  8B judge deciding whether its own output matches the ground truth is a real
  measurement risk, but changing it changes every historical number in
  `Log.md`. Worth a separate, deliberate decision.
* **`apiro/patient/context.py` is orphaned** — `PatientContext` is imported
  nowhere except its own package and a stale docstring. It is a good source of
  structured constraints (age/gender/chief complaint) if you want to re-anchor
  the synthesis prompt; left untouched for now.
* **`WBC 18.2 x10^9/L` still does not parse** — the general lab regex needs a
  purely alphabetic unit. Adding scientific-notation units is a contained
  follow-up.
* **Stale tests**: `tests/test_entropy_engine.py` still asserts the pre-rewrite
  logprob API (6 failures, all pre-existing on `main`), and two `test_html_spec`
  tests need `chromadb` / a live Ollama. Left as found so the baseline stays
  comparable.
