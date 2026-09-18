# Apiro Evaluation Strategy

A first-principles plan for testing whether Apiro does what it claims. It is written independently of the existing evaluation code and is intended to be frozen before any confirmatory run.

---

## 0. The goal being evaluated

Apiro's claim is that **an explicit search process can make an LLM's diagnosis more robust to misleading context** than simply prompting the model or giving it retrieved evidence. It proposes three mechanisms:

1. **Anchoring** to findings extracted from the patient narrative.
2. **Uncertainty-directed exploration** of hypotheses.
3. **Contradiction-aware down-weighting** of hypotheses that conflict with patient evidence.

An evaluation is iron-clad when every plausible alternative explanation for a positive result has been ruled out in advance. It must also ensure that a negative result is informative rather than merely underpowered.

The alternative explanations this plan is built to rule out:

| Alternative explanation | Where it is handled |
|---|---|
| Apiro wins only because it uses far more compute | §3 compute-matched arms, §9 cost curves |
| Apiro is "robust" because it ignores all input changes | §4.3 should-change controls |
| Test cases were memorized during pretraining | §4.4 contamination controls |
| Prompts or thresholds were tuned on the test set | §5 sealed splits |
| The grader favours Apiro's phrasing | §6 blinded, validated scoring |
| The effect is one lucky model or seed | §7.5 repeated runs, §10 multiple backbones |
| The effect comes from a component other than the three mechanisms | §8 factorial ablation |
| Baselines were weakly prompted | §3.3 equal tuning budget |
| Many metrics were tried and the best one reported | §2 pre-registered primary endpoint, §7.3 gatekeeping |
| Distractors were accidentally informative | §4.2 clinician validation |

---

## 1. Research questions and hypotheses

| ID | Question | Type | Hypothesis |
|---|---|---|---|
| **H1** | Does Apiro reduce the accuracy lost when distractors are added? | **Primary, superiority** | The accuracy drop from clean to distracted cases is smaller for Apiro than for the strongest compute-matched baseline. |
| H2 | Does Apiro keep clean-case accuracy? | Secondary, non-inferiority | Clean top-1 accuracy is no more than 5 points below the strongest baseline. |
| H3 | Does Apiro still update when evidence *should* change the diagnosis? | Secondary, non-inferiority | Correct updating on should-change pairs is not worse than baseline. |
| H4 | Which of the three mechanisms cause the effect? | Mechanistic | At least one mechanism has a non-zero main effect on H1's endpoint. |
| H5 | Are Apiro's confidence scores more useful for abstention? | Exploratory | Lower selective risk at matched coverage. |
| H6 | Is any gain worth its cost? | Descriptive | Apiro lies on or above the baseline accuracy-vs-compute frontier. |

Only H1 can support the headline claim. H2 and H3 are guardrails: a win on H1 that fails either one is reported as a trade-off, not a success.

---

## 2. Estimands (what exactly is measured)

### 2.1 Unit of analysis

A **case family** is one clinical case with a single validated gold diagnosis, plus perturbed variants of it (§4). All statistics cluster at the family level.

### 2.2 Primary endpoint — Distractor-Induced Accuracy Drop (DIAD)

For arm *a*, over all should-not-change pairs:

```
DIAD(a) = Acc@1(a, clean) − Acc@1(a, distracted)
```

The primary contrast is:

```
Δ = DIAD(best compute-matched baseline) − DIAD(Apiro)
```

A positive Δ means Apiro loses less accuracy.

**Why this rather than a conditional trap rate:** a rate conditioned on "cases the arm solved when clean" compares arms on *different subsets* of cases. A stronger arm then gets penalized for attempting harder cases. DIAD uses every pair for every arm, a design analogous to intention-to-treat. Conditional flip rates are still reported, but as secondary endpoints.

### 2.3 Secondary endpoints

| Endpoint | Definition |
|---|---|
| Clean Acc@1 | Gold diagnosis is rank 1 on the clean variant |
| Distracted Acc@1 | Gold diagnosis is rank 1 on the distracted variant |
| Flip-to-distractor rate | Rank 1 moves from gold to the distractor-suggested diagnosis |
| Pair robustness | Correct on both clean and distracted |
| Should-change update rate | Rank 1 moves to the new gold diagnosis when discriminating evidence changes |
| Acc@3 | Secondary capability measure, never substituted for Acc@1 |
| Must-not-miss error rate | Proportion of cases where a pre-labelled dangerous diagnosis is absent from top 3 |

### 2.4 Estimand definitions that must be fixed in advance

- **Abstention:** in primary analyses, an abstention counts as incorrect. The alternative is reported only as a sensitivity analysis.
- **Parse failure:** counts as incorrect for every arm, with parse-failure rates reported per arm.
- **Output format:** every arm returns exactly three ranked diagnoses under the same instruction.

---

## 3. Comparison arms

### 3.1 Required arms

| Arm | Purpose |
|---|---|
| **A0** Bare LLM, direct answer | Floor |
| **A1** Bare LLM with chain-of-thought | Rules out "any reasoning helps" |
| **A2** Standard RAG (same corpus, retriever, backbone) | Rules out "evidence alone helps" |
| **A3** Self-consistency RAG, compute-matched | Rules out "more calls help" |
| **A4** Iterative / agentic RAG (retrieve–reason–revise loop), compute-matched | Rules out "any multi-step search helps" |
| **A5** Debiasing prompt ("consider findings that don't fit; ignore irrelevant details") | Rules out "a better prompt is enough" |
| **A6** Apiro, full system | Treatment |
| **A7** Oracle evidence (gold-relevant passages supplied) | Ceiling for retrieval quality |

The primary contrast in §2.2 compares Apiro against **the best of A2–A5**, chosen on the development split and locked before the test run. Comparing against the weakest baseline is not allowed.

### 3.2 Compute matching

- Record prompt tokens, completion tokens, model calls, and wall-clock time for every run.
- A3 and A4 are run at **three budget points**: about 1×, 0.5×, and 1× of Apiro's median tokens per case. Self-consistency is scaled by the number of samples; iterative RAG by the number of loops.
- The primary contrast uses the baseline at the **token budget closest to Apiro's median**. Token budget is fixed as the matching unit in advance.

### 3.3 Fair tuning

- Every arm gets the **same development-split tuning budget**: the same number of prompt iterations, the same dev cases, and the same person-hours logged.
- Baseline prompts are written or reviewed by someone other than the Apiro author.
- All arms use the same backbone, decoding settings for final answers, retriever, corpus, answer format, and context-length limit (with the limit verified as not silently truncating inputs).

---

## 4. Test data design

### 4.1 Source cases

Assemble clean cases from at least three independent sources, so no single dataset's quirks drive the result:

- Published clinical vignette benchmarks (differential-diagnosis style, not multiple-choice).
- Structured diagnostic datasets converted to narrative form.
- **Recent case reports published after the backbone models' training cutoff** (the contamination-resistant set; see §4.4).

Inclusion criteria: one clinician-agreed primary diagnosis; enough information for a clinician to reach it; English; no images required.

Stratify the sample by specialty, by rare vs common disease, and by narrative length (short, medium, long), and report results per stratum.

### 4.2 Controlled perturbations

Each clean case generates variants by **minimal edits**, so any change in output can be attributed to that edit.

**Should-not-change perturbations** (gold diagnosis unchanged):

| Family | Example |
|---|---|
| Bystander finding | Irrelevant comorbidity or incidental lab |
| Anchoring suggestion | "Referring doctor suspects X" |
| Misleading history | Past diagnosis pointing to a common look-alike |
| Salient but non-discriminative symptom | Vivid detail typical of a distractor disease |
| Recency / ordering | Distractor placed last |
| Long-context burial | Key finding buried among filler text |
| Social / authority pressure | Patient or senior clinician insists on X |
| Lexical noise | Colloquial wording, abbreviations, typos |

**Should-change perturbations** (gold diagnosis changes):

| Family | Example |
|---|---|
| Discriminating cue swap | One finding changed so a different diagnosis becomes correct |
| Key finding removed | The finding that confirms the diagnosis is removed, so the gold becomes a different or less specific answer |

Should-change pairs are essential. Without them, a system that never changes its answer would look perfectly robust.

**Validation of every perturbation:**

1. Two independent clinicians label each variant's correct diagnosis, blind to which variant is the clean one.
2. Target agreement is Cohen's κ ≥ 0.7. Disagreements go to a third clinician; unresolved variants are excluded, and exclusions are logged.
3. For should-not-change variants, clinicians also confirm the distractor is *plausible*. Trivially irrelevant noise does not test anchoring.
4. Distractor diagnoses are pre-labelled, so flip-to-distractor can be scored exactly.

### 4.3 Held-out perturbation families

Two should-not-change families are **never seen during development**. They appear only in the test set. This measures whether robustness generalizes, rather than reflecting tuning to known distractor types. Results are reported separately for seen and unseen families.

### 4.4 Contamination controls

- **Post-cutoff subset:** cases published after every backbone's training cutoff. The primary analysis is repeated on this subset alone.
- **Memorization probe:** give the model the first half of a case and ask it to continue. High verbatim overlap flags a case as likely memorized. Report results with and without flagged cases.
- **Surface rewrite:** a clinician-validated paraphrase of each clean case. A large accuracy gap between original and paraphrase indicates memorization.
- **Corpus leakage check:** confirm that the retrieval corpus does not contain the source case text itself. Run a sensitivity analysis with a corpus from which each test case's source documents are removed.

---

## 5. Splits and anti-leakage protocol

| Split | Share | Use |
|---|---|---|
| Development | ~20% of families | Prompt work, thresholds, bug fixing, baseline selection |
| Calibration | ~15% | Fitting confidence and abstention thresholds only |
| **Locked test** | ~65% | One confirmatory run |

- Splitting is done **by case family**, so no variant of a test case is ever seen in development.
- Stratification by source, specialty, and length is kept across splits.
- The locked test set is stored in encrypted or access-controlled form, and its hash is published in the pre-registration **before** any Apiro development against these families.
- A **freeze record** is published before the test run. It lists model versions and weight digests, prompts, thresholds, the corpus snapshot hash, the retriever, the parser, the grader, the analysis plan, and the random seeds.
- The test set is run **once**. If a pipeline bug is found afterward, the fix is logged, the whole run is repeated for **all arms**, and both runs are reported.

---

## 6. Scoring and grading

### 6.1 Diagnosis matching

Free-text diagnoses must be matched to the gold label without favouring any arm's phrasing:

1. **Normalize:** lowercase, expand abbreviations, and map to a standard ontology concept (e.g., UMLS/SNOMED CT).
2. **Exact concept match** counts as correct.
3. **Hierarchy rules, fixed in advance:** a more specific child of the gold concept is correct. A parent is correct only if the gold label itself is at that level. A sibling is incorrect.
4. **Unmapped outputs** go to a grader (§6.2).

### 6.2 Grader requirements

- If an LLM grader is used, it comes from a **different model family** than the backbone under test.
- The grader sees only the gold diagnosis and the candidate string. It does not see the arm name, the case text, the reasoning trace, or the case variant type.
- **Validation:** a stratified sample of at least 300 grading decisions is labelled by clinicians. The grader must reach κ ≥ 0.8 against clinicians, with no significant difference in agreement between arms. If it fails, clinicians grade every unmapped output.
- Arm outputs are shuffled and anonymized before any human grading.

### 6.3 Scoring audit

A script-independent re-scoring is done on a random 10% of cases by someone who did not write the scorer. Any discrepancy triggers a full audit before results are unsealed.

---

## 7. Statistical analysis plan

### 7.1 Pre-registration

The following are publicly time-stamped before the locked run: hypotheses, the primary endpoint, the primary contrast, the comparator-selection rule, the sample size, analysis models, multiplicity handling, exclusion rules, and decision thresholds (§11).

### 7.2 Primary model

Fit a mixed-effects logistic regression on correctness:

```
correct ~ arm × condition + (1 | case_family) + (1 | source)
```

with `condition ∈ {clean, distracted}`. The primary test is the **arm × condition interaction** for Apiro vs the locked comparator, converted to a difference in accuracy drop (Δ) with a 95% confidence interval.

Confirmatory checks:

- Case-family **cluster bootstrap** (10,000 resamples) for the CI on Δ.
- Exact paired tests on per-pair outcomes (e.g., McNemar on pair robustness).

### 7.3 Multiplicity

Use **hierarchical gatekeeping**. Each test is interpreted only if the previous one succeeded:

1. H1 superiority (α = 0.05, two-sided)
2. H2 non-inferiority (margin 5 points)
3. H3 non-inferiority (margin 5 points)
4. H4 mechanism main effects (Holm correction across the three mechanisms)

H5 and H6 are exploratory and are labelled as such.

### 7.4 Sample size

- Set the **smallest effect worth detecting** in advance, for example a 7-point reduction in accuracy drop.
- Estimate the pair outcome distributions and the within-family correlation from the development split.
- Choose the number of test families by **simulation** that achieves 90% power for H1 after multiplicity and clustering.
- As a rough guide, a paired design detecting a 10-point difference with about 20% discordant pairs needs about 160 pairs at 80% power. After clustering, multiple perturbation families, and a 90% power target, **plan for roughly 400–600 case families**. The simulation number replaces this guide.
- If the budget cannot reach the required size, the study is declared exploratory in the pre-registration, not after the results are seen.

### 7.5 Stochasticity

- Run each arm **k = 3 to 5 times** per case variant with different seeds (non-zero temperature where the arm uses it).
- Model run as a nested random effect, and report the **between-run variance** alongside the between-case variance.
- A result that is significant only in some seeds is reported as not robust.

### 7.6 Making a null result informative

If H1 is not significant, run an **equivalence test (TOST)** with ±3 points as the equivalence margin:

- Equivalence shown → "Apiro provides no meaningful robustness benefit at this scale."
- Neither difference nor equivalence shown → "inconclusive," with the CI reported.

A null result is never described as evidence of no effect without the equivalence test.

### 7.7 Sensitivity analyses (fixed in advance)

- Post-cutoff subset only
- Excluding memorization-flagged cases
- Each perturbation family separately; seen vs unseen families
- Strict vs lenient diagnosis matching
- Abstention scored as missing rather than incorrect
- Leave-one-source-out
- Conditional flip rate (the earlier trap-rate style) for comparability with prior work

---

## 8. Mechanism attribution (H4)

### 8.1 Factorial ablation

Run a **2 × 2 × 2 factorial** over the three mechanisms, on a pre-specified random subset of the locked test set:

| Factor | On | Off |
|---|---|---|
| Anchoring | Extracted findings seed the search | Search seeded from narrative only |
| Uncertainty-directed frontier | Priority by uncertainty | Random frontier at the same budget |
| Contradiction handling | Penalty applied | No contradiction checks |

Estimate main effects and interactions on the primary endpoint. This isolates which mechanism matters and whether they only work together.

### 8.2 Component validity

Each mechanism is only as good as its inputs, so measure these on labelled samples:

| Component | Metric | Labelled data |
|---|---|---|
| Finding extraction | Precision / recall / F1, including negation | 100 clinician-annotated cases |
| Contradiction detection | Precision / recall | ~300 claim pairs labelled by clinicians |
| Retrieval | Recall@k, MRR for gold-relevant evidence | 200 query–passage relevance judgements |
| Uncertainty signal | AUROC of confidence for correct vs incorrect hypotheses | Derived from runs |

### 8.3 Causal interventions

Directly edit Apiro's internal state and measure the change in its output:

- **Inject a correct contradiction flag** against the distractor hypothesis. Does the flip rate fall?
- **Inject a false contradiction flag** against the gold hypothesis. Does accuracy fall?
- **Remove the discriminating finding** from the extracted findings. Does robustness collapse?

If editing a mechanism's state does not change the output, that mechanism is not doing the work, whatever the ablation suggests.

### 8.4 Process evaluation

On a blinded sample of about 100 traces, clinicians rate:

- whether the trace recognized the distractor as non-discriminative;
- whether the final answer follows from the trace, or the trace is post-hoc;
- whether unsupported claims were introduced.

This checks that improvements come from the claimed reasoning and not from coincidence.

---

## 9. Cost and efficiency (H6)

- Report tokens, calls, wall-clock time, and peak memory per case, as median and 90th percentile, per arm.
- Plot **accuracy (clean and distracted) against tokens per case** for all arms and budget points, and mark the Pareto frontier.
- Report **robustness gained per 1,000 tokens** relative to the best baseline.
- All timing runs use identical hardware and serving configuration, recorded in the freeze record.

A result where Apiro is more robust but sits below the self-consistency frontier is reported as: "more robust, but not an efficient way to buy robustness."

---

## 10. Generalization

The primary result uses one pre-declared backbone. The same frozen protocol is then repeated, as secondary analyses, on:

- **a second model family** of similar size;
- **a larger model** (about 70B class), to test whether the benefit shrinks as base models improve;
- **a second retrieval corpus**, to test dependence on the knowledge source.

The effect is described as general only if its direction is consistent across these conditions. Otherwise, the claim is limited to the conditions in which it held.

---

## 11. Decision rules (fixed in advance)

| Outcome | Conclusion to report |
|---|---|
| H1 significant, H2 and H3 non-inferior, holds on post-cutoff subset and ≥ 2 backbones | "Apiro improves robustness to misleading context without losing accuracy or responsiveness, under the tested conditions." |
| H1 significant, but H2 or H3 fails | "Apiro trades robustness for [accuracy / responsiveness]." |
| H1 significant only against weaker, non-matched baselines | "No evidence of benefit beyond additional compute." |
| H1 significant on seen families but not unseen families | "Robustness does not generalize to new distractor types." |
| H1 not significant; equivalence shown | "No meaningful benefit at this scale." |
| H1 not significant; equivalence not shown | "Inconclusive," with CI and the power achieved. |
| Effect disappears on post-cutoff or memorization-filtered cases | "Result likely driven by contamination." |

The conclusion in the final report must match one row of this table.

---

## 12. Threats to validity

| Threat | Type | Mitigation |
|---|---|---|
| Gold labels are wrong or ambiguous | Construct | Dual clinician labelling, κ threshold, exclusion log |
| Perturbations are unrealistic | External | Clinician plausibility rating; include real ambiguous case reports |
| Distractors leak the answer | Internal | Blind clinician check that gold is unchanged |
| Grader bias toward verbose answers | Measurement | Blinded grading, cross-family grader, arm-level agreement check |
| Silent context truncation | Internal | Verify effective context length, log truncation events per arm |
| Retrieval corpus contains the test case | Internal | Source-removed corpus sensitivity analysis |
| Researcher degrees of freedom | Statistical | Pre-registration, locked split, single run |
| Baseline under-tuning | Internal | Equal tuning budget, independent baseline author |
| Benchmark ≠ clinical practice | External | Stated scope limit; no clinical claims |
| Model/API drift | Reproducibility | Local weights with digests; no hosted model updates mid-study |

---

## 13. Reproducibility and transparency

The following are released:

- The pre-registration and freeze record, with timestamps.
- All case variants that licensing permits, with clinician labels and agreement statistics.
- Raw outputs for every arm, seed, and case, plus parsed answers and grades.
- The analysis notebook, which regenerates every table from raw outputs.
- The run manifest: hardware, software versions, weight digests, corpus hash.
- A failure-case appendix: at least 20 randomly selected Apiro errors and 20 baseline errors, with traces.
- Any deviations from the pre-registration, each with a reason.

---

## 14. Execution phases and gates

| Phase | Work | Exit gate |
|---|---|---|
| **1. Data** | Collect sources, generate perturbations, clinician labelling, splitting | κ ≥ 0.7; test hash published |
| **2. Measurement** | Ontology mapping, grader validation, component labelled sets | Grader κ ≥ 0.8 with no arm bias |
| **3. Development** | Tune all arms under equal budget; select comparator; pilot for variance estimates | Freeze record complete |
| **4. Power** | Simulation-based sample size; confirm budget | Pre-registration published |
| **5. Confirmatory run** | All arms, k seeds, locked test | Scoring audit passes |
| **6. Mechanisms** | Factorial ablation, interventions, process review | Pre-specified subset complete |
| **7. Generalization** | Second backbone, larger backbone, second corpus | Same frozen protocol |
| **8. Reporting** | Tables, CIs, decision-rule mapping, release | Conclusion matches §11 |

No phase may start using the locked test set before Phase 5.

---

## 15. Report checklist

- [ ] Primary endpoint and contrast exactly as pre-registered
- [ ] Every arm, including the compute-matched and debiasing baselines
- [ ] Effect sizes with 95% CIs, not only p-values
- [ ] Clean accuracy, distracted accuracy, and should-change updating together
- [ ] Seen vs unseen perturbation families
- [ ] Post-cutoff and memorization-filtered results
- [ ] Between-seed variance
- [ ] Factorial mechanism effects and intervention results
- [ ] Accuracy-vs-compute frontier
- [ ] Results on the additional backbones and corpus
- [ ] Threats to validity and all deviations
- [ ] A conclusion drawn from one row of §11, with no clinical-use claims
