# Benchmarking

Main evaluates distractor resistance with matched clinical reports. Every
benchmark compares the same three arms where supported:

- Bare LLM
- Standard RAG
- Apiro (`simple` or `investigator`)

## Active Benchmarks

### MedDistractQA

Primary benchmark. Each diagnosis question is evaluated once with its
distracting sentence removed and once with it present.

```bash
APIRO_REASONING_MODE=simple \
  python scripts/run_meddistractqa_eval.py --n 100
APIRO_REASONING_MODE=investigator \
  python scripts/run_meddistractqa_eval.py --n 100
```

Report top-1 accuracy for clean and distracted cases, paired retention, and
runtime/model-call counts. Use the same seed, model, corpus, and case IDs for
every arm.

### MedEinst

Secondary paired benchmark for anchoring bias. It supplies control/trap pairs
and reports control accuracy, trap accuracy, Bias Trap Rate, and pair
resilience.

```bash
python scripts/run_medeinst_eval.py --n-pairs 60
```

### MINT

Local incremental-evidence benchmark. Its JSON input must contain
`case_id`, `ground_truth`, and non-empty `turns`; each turn has `text` and may
set `category` and `is_lure`.

```bash
python scripts/run_mint_eval.py --dataset-json path/to/mint.json --n 100
```

## Reproducibility

Use `--describe-only` before a live run. Every completed run writes a manifest,
case-level results, model settings, and timing data under `data/runs/`.
Keep prompts, model digest, corpus manifest, seed, thresholds, and selected
case IDs fixed before comparing branches.

Run the offline suite first:

```bash
pytest -q
```

Tiny smoke runs are useful for wiring only. They cannot support an accuracy
claim; use a predeclared, held-out sample for conclusions.

## Archived Experiments

C-NIAH, CUPCase, DDXPlus, PMC, MIMIC, and the old safety-calibration tooling
belong to the legacy implementation and are not active main-branch
benchmarks. Reproduce them from `archive/legacy-main` rather than restoring
their adapters or generators here.
