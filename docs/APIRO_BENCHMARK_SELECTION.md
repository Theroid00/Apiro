# Apiro Benchmark Selection

Which benchmarks to run Apiro on, and why.

Apiro's claim is **robustness to misleading context in diagnosis**. The benchmark set must therefore answer three questions:

1. **Does it resist distractors?** (robustness)
2. **Is that real, not memorized?** (contamination)
3. **Does it cost accuracy on normal cases?** (guardrail)

---

## Tier 1 — Primary benchmark (test the actual claim)

### Paired distractor-report benchmark — main result

- **What it tests:** whether Apiro preserves the correct diagnosis when the same
  report is augmented with irrelevant, plausible, or misleading clinical
  content.
- **Why it is primary:** distractor rejection is Apiro's defining purpose. A
  clean-case benchmark can show diagnostic competence, but it cannot establish
  that Apiro is better at separating signal from noise.
- **Design:** every case must have an exactly paired `clean` and `distracted`
  report. The clinical signal, ground truth, report length budget, model, corpus,
  and decoding settings remain fixed; only the distractor changes.
- **Required distractor families:** irrelevant history/comorbidity, salient but
  non-causal symptoms, old or temporally mismatched events, normal/noisy test
  results, patient self-diagnosis or anchoring language, and plausible external
  context. Score these both in aggregate and by family.
- **Primary endpoint:** top-1 paired retention, defined as `P(correct on
  distracted | correct on clean)`. Also report the paired top-1 degradation,
  wrong-top-diagnosis flip rate, confidence change, and unsupported-evidence
  rate. Top-3 retention is secondary and must not replace top-1.
- **Recommended construction:** use MedDistractQA-style clean/distracted pairs
  for an initial reproducible run, then build a contamination-resistant set of
  post-cutoff PMC case reports with controlled distractor injection. Keep the
  answer choices out of the model prompt and score free-text diagnoses.
- **Acceptance rule:** Apiro should retain materially more correct top-1
  diagnoses than bare LLM and ordinary RAG, without a meaningful clean-case
  accuracy penalty.

### MedEinst — focused secondary benchmark

- **What it tests:** the Einstellung effect in differential diagnosis. Models favour common patterns over patient-specific evidence when misleading features are present.
- **Why it fits Apiro:** it is a useful focused test of anchoring and the
  Einstellung effect, but it covers one cognitive-bias pattern rather than the
  full distractor-report problem. It should support, not define, the headline
  result.
- **Paper:** https://arxiv.org/abs/2601.06636
- **Caveat:** an 8B backbone solves few control cases at rank 1, so a large number of pairs is needed to get enough usable ones.

### Trap injection on DDXPlus-style cases

- **What it tests:** four trap types added to diagnostic cases:
  - patient self-diagnosis (confirmation bias)
  - irrelevant medical history
  - non-critical external noise (environment, lifestyle)
  - blurred primary vs secondary symptoms
- **Why it fits Apiro:** these map directly onto Apiro's distractor-resistance mechanisms, and cover distractor families MedEinst lacks.
- **Paper:** https://arxiv.org/abs/2510.09275
- **Use:** apply the same trap types to your own case sets as well (see Tier 3).

### BiasMedQA-style cognitive traps — secondary

- **What it tests:** cognitive-bias and distraction prompts (anchoring, false
  consensus, bystander details).
- **Caveat:** both are built on multiple-choice MedQA. Answer options leak information and don't match Apiro's free-text output.
- **Supporting evidence:** a February 2026 medRxiv factorial study found that boards-style multiple-choice benchmarks overestimate distractor bias compared with realistic free-response settings.
  https://www.medrxiv.org/content/10.64898/2026.02.12.26346164v1
- **Rule:** remove the answer options and score free text, or drop these benchmarks.

---

## Tier 2 — Clean-accuracy guardrail

Robustness is meaningless if Apiro gets worse on normal cases.

| Benchmark | What it is | Role |
|---|---|---|
| **MedCaseReasoning** | Real published case reports with free-text diagnoses and clinician reasoning | Hard, realistic clean-accuracy set |
| **CUPCase** | Rare-disease cases | Tests the cases where retrieval should matter most |
| **DDXPlus** | Large synthetic structured-symptom dataset | Sanity check only; synthetic and relatively easy |

---

## Tier 3 — Contamination-proof set (build it yourself)

- **Source:** PubMed Central Open Access case reports published **after the backbone model's training cutoff** (for Llama 3.1, roughly 2024 onward).
- **Size:** 200–300 cases.
- **Why:** public benchmarks may be in the model's pretraining data; these cases cannot be.
- **Use:** run the clean versions, then apply the Tier 1 trap types.
- **Payoff:** if Apiro's advantage holds here, a memorization objection has no ground.

---

## Optional — Premature commitment

### MIMIC-IV-Ext Clinical Decision Making

- **What it tests:** diagnosis with information revealed step by step, from real patient records.
- **Why:** shows whether Apiro commits too early as evidence arrives.
- **Access:** requires PhysioNet credentialed access. Apply early if you plan to use it.

---

## Skip

| Benchmark | Reason |
|---|---|
| MedQA / MedMCQA as primary | Saturated, multiple-choice, heavily contaminated |
| HealthBench | Measures conversational health advice, not diagnosis |
| Synthetic C-NIAH as headline | Fine as a mechanism test, not as primary evidence |

---

## Minimum recommended set (given compute limits)

| # | Benchmark | Question it answers |
|---|---|---|
| 1 | **MedEinst** | Does Apiro resist distractors? (primary result) |
| 2 | **Post-cutoff PMC cases with injected traps** | Is it real, not memorized? |
| 3 | **MedCaseReasoning (clean)** | Does it cost clean accuracy? |

Together these answer the three questions a reviewer or interviewer will ask.
