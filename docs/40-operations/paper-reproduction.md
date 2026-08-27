# Reproducing the ISETrace paper experiments

This document is the paper-facing reproduction entry point. It records the
scientific inputs, Hydra overrides, run matrices, post-hoc transformations, and
artifact locations. Detailed package and implementation notes under this
repository are developer references and are not required to review the paper.

## 1. Environment and inputs

Run commands from the repository root on Linux with a CUDA-capable GPU:

```bash
uv sync
```

`uv run python experiment/run.py ...` invokes input provisioning before the
Prefect flow. Missing encoder directories are downloaded from the official Hugging Face Hub into the paths used by the configs:

| Method family | Model source | Local path |
|---|---|---|
| Dense, Dense-FT, GraphRAG, Provenance Path, R-GCN | `intfloat/e5-base-v2` revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | `models/intfloat-e5-base-v2` |
| Cross-Encoder-FT | `BAAI/bge-reranker-base` revision `2cfc18c9415c912f9d8155881c133215df768a70` | `models/BAAI-bge-reranker-base` |

The advisor package contains the complete ISETrace raw trajectories, intents,
and frozen query corpus. It intentionally excludes downloaded models,
checkpoints, embeddings, and `data/processed/`; selected workflows regenerate
those assets as needed.

Canonical inputs:

```text
data/isetrace/raw/intents/intents.jsonl
data/isetrace/raw/trajectories/trajectories-00000.jsonl ... trajectories-00007.jsonl
data/isetrace/query-authoring/isetrace-v7-raw.jsonl                 # 6,597 records
data/isetrace/query-authoring/isetrace-v7-raw.jsonl.metadata.jsonl
```

The active split is trajectory-disjoint and uses `split_seed=13`:

| Split | Trajectories | Natural queries in the full profile |
|---|---:|---:|
| train | 2,410 | 3,894 |
| dev | 352 | 580 |
| test | 1,207 | 2,000 |

The legacy `data/isetrace/splits/v1` allocation is not used. One Hydra job runs
one method, optional variant, and model seed. It writes output-only files under
`runs/<name>/` and reusable assets under `data/processed/`.

A small lifecycle check can be run with any method, for example:

```bash
uv run python experiment/run.py \
  name=isetrace_bm25_smoke dataset=isetrace profile=smoke \
  method=bm25 split_seed=13 device=cuda:0
```

## 2. Main retrieval experiment

The deterministic methods run once with `profile=full`, `split_seed=13`:

```bash
uv run python experiment/run.py name=isetrace_v7_bm25 \
  dataset=isetrace profile=full method=bm25 split_seed=13 device=cuda:0
uv run python experiment/run.py name=isetrace_v7_dense_flat \
  dataset=isetrace profile=full method=dense method.variant=flat \
  split_seed=13 device=cuda:0
uv run python experiment/run.py name=isetrace_v7_dense_pu \
  dataset=isetrace profile=full method=dense method.variant=provenance_unit \
  split_seed=13 device=cuda:0
uv run python experiment/run.py name=isetrace_v7_graphrag \
  dataset=isetrace profile=full method=graphrag split_seed=13 device=cuda:0
uv run python experiment/run.py name=isetrace_v7_provenance_path \
  dataset=isetrace profile=full method=provenance_path split_seed=13 device=cuda:0
```

Trainable methods use model seeds `13 17 29 37 41` while keeping
`split_seed=13`. For each seed, run:

| Paper condition | Hydra method and variant |
|---|---|
| Flat Dense-FT | `method=dense_ft method.variant=flat` |
| Provenance-Unit Dense-FT | `method=dense_ft method.variant=provenance_unit` |
| Flat Cross-Encoder-FT | `method=cross_encoder method.variant=flat` |
| Provenance-Unit Cross-Encoder-FT | `method=cross_encoder method.variant=provenance_unit` |
| Exact seed passthrough | `method=provenance_unit_dense_ft_rgcn method.variant=wo_graph` |
| Residual R-GCN | `method=provenance_unit_dense_ft_rgcn method.variant=full_rgcn` |

Example:

```bash
uv run python experiment/run.py \
  name=isetrace_v7_dense_ft_pu_s13 \
  dataset=isetrace profile=full method=dense_ft \
  method.variant=provenance_unit seed=13 split_seed=13 device=cuda:0
```

The relation/topology controls keep the same data, candidate view, seed
provider, and training recipe. Repeat each over all five model seeds:

```text
method=provenance_unit_dense_ft_rgcn
method.variant=homogeneous_gcn
method.variant=wo_feeds
method.variant=wo_execution_ownership
method.variant=wo_artifact_io
method.variant=wo_chunk_adjacency
method.variant=random_edges
```

The formal delivered runs are under `results/isetrace/evalv8/runs/`. Seeds 37
and 41 retain later historical run names, but use the same scientific
conditions. Every complete run contains its resolved config, manifest, final
metrics, 2,000 per-task rows, and compact ranked prefix.

For trainable methods, report the mean and sample standard deviation over five
seeds. Paired method differences first average matched-seed per-query
differences and then use 10,000 trajectory-cluster bootstrap replicates with
bootstrap seed 13. `scripts/aggregate_main_results.py` performs these checks and
computations from explicit `LABEL=RUN_PATH` inputs.

## 3. Flat chunk-size sweep

The sweep changes only the flat candidate cap:

```text
dataset.chunking.max_tokens = 128, 256, 384, 512
dataset.chunking.reserved_tokens = 8
dataset.chunking.overlap_tokens = 64
```

For every cap, run BM25 and frozen Dense once, and run Flat Dense-FT and Flat
Cross-Encoder-FT over seeds `13 17 29 37 41`. Example:

```bash
uv run python experiment/run.py \
  name=isetrace_v7_chunksweep_dft_c128_s13 \
  dataset=isetrace profile=full method=dense_ft method.variant=flat \
  dataset.chunking.max_tokens=128 seed=13 split_seed=13 device=cuda:0
```

The 512-token condition is the main experiment configuration and need not be
trained again. Delivered 128/256/384 runs are under
`results/isetrace/evalv8/runs/chunksweep/`; 512-token runs remain in their main
method directories. The frozen summaries are under
`results/isetrace/chunksweep/`.

For each trainable method/cap cell, aggregate five-seed mean and sample SD. The
paper's flat oracle is not a deployable run: independently for each reported
metric, select the best mean among caps 128/256/384/512. The complete grid and
the development selections are retained in `combined.csv` and
`dev-selection.csv`.

## 4. E4 audit and human correction

E4 is a frozen audit, not a deterministic model-training experiment. Its
reviewer and human decisions are primary observed data and are retained rather
than regenerated.

`results/isetrace/e4-audit/` contains the stratified 500-query packet, two blind
model-reviewer judgments, disagreement material, and final model adjudication.
`results/isetrace/e4-human-validation/` contains the human field decisions,
corrected adjudication, task subsets, and five-seed sensitivity results.

The human-corrected analysis follows these transformations:

1. Compare the five `llm_final` fields with the human-corrected `final` fields in
   `human_corrected_adjudication.jsonl`. This yields record/field concordance and
   the 24 corrected fields.
2. A query enters `human_quality_pass_task_ids.json` when the corrected
   `valid_query`, `answerable`, `gold_accuracy`, and `gold_completeness` fields
   are all `yes` (469 queries).
3. A query enters `human_verified_task_ids.json` when it passes those four
   quality fields and the corrected query type equals the authored type (367
   queries).
4. Filter each formal method/seed `metrics/per_task.jsonl` by the corresponding
   task-ID set. Recompute five-seed means, method order, and residual-minus-exact
   seed trajectory-cluster intervals with 10,000 replicates and seed 13.
5. For type sensitivity, join `human_corrected_query_metadata.jsonl` and group
   the same per-task rows by corrected `direct_recall`, `linked_recall`, and
   `multi_fact_recall`.

The original model-verified and human-verified tables are retained side by side,
so the paper conclusions can be checked using ordinary JSONL filtering and the
same aggregation rule without reproducing the reviews themselves. The optional
local review UI is documented under `.ui/codex/README.md`.

## 5. Mechanism and evidence-dispersion analysis

`scripts/analyze_isetrace_mechanisms.py` consumes one prepared formal test
artifact plus four matched five-seed run groups:

```text
--flat-run                 Flat Dense-FT
--provenance-unit-run      Provenance-Unit Dense-FT
--seed-run                 exact seed passthrough
--rgcn-run                 residual R-GCN
```

Repeat each run option once per seed, pass the prepared test dataset directory
referenced by a newly completed formal run's `assets/manifest.yaml`, and write:

```bash
uv run python scripts/analyze_isetrace_mechanisms.py \
  --prepared-dir <processed-test-dataset-directory> \
  --flat-run <flat-s13> --flat-run <flat-s17> --flat-run <flat-s29> \
  --flat-run <flat-s37> --flat-run <flat-s41> \
  --provenance-unit-run <pu-s13> --provenance-unit-run <pu-s17> \
  --provenance-unit-run <pu-s29> --provenance-unit-run <pu-s37> \
  --provenance-unit-run <pu-s41> \
  --seed-run <seed-s13> --seed-run <seed-s17> --seed-run <seed-s29> \
  --seed-run <seed-s37> --seed-run <seed-s41> \
  --rgcn-run <rgcn-s13> --rgcn-run <rgcn-s17> --rgcn-run <rgcn-s29> \
  --rgcn-run <rgcn-s37> --rgcn-run <rgcn-s41> \
  --bootstrap-samples 10000 --bootstrap-seed 13 \
  --output results/isetrace/evalv8/analysis/mechanisms.json \
  --output-csv results/isetrace/evalv8/analysis/mechanisms.csv \
  --figure docs/raw/figs/mechanism-dispersion.pdf \
  --dispersion-figure docs/raw/figs/gold-event-dispersion.pdf
```

The script derives minimum flat set cover, distinct gold-bearing event count,
and observable provenance distance from frozen inputs without using retrieval
outputs to define the strata.

## 6. Training-size robustness

Use nested train-trajectory prefixes while holding development and test fixed:

| Percentage | `dataset.trajectories.splits.train` |
|---:|---:|
| 10 | 241 |
| 25 | 603 |
| 50 | 1,205 |
| 100 | 2,410 |

For every size and seed `13 17 29 37 41`, run:

```text
method=dense_ft method.variant=provenance_unit
method=provenance_unit_dense_ft_rgcn method.variant=wo_graph
method=provenance_unit_dense_ft_rgcn method.variant=full_rgcn
```

Include `dataset=isetrace profile=full split_seed=13` and the selected
`dataset.trajectories.splits.train` value in every command. This is 60 runs.
Aggregate them with:

```bash
uv run python scripts/analyze_isetrace_training_size.py \
  --run-root runs \
  --output results/isetrace/training-size/analysis.json \
  --output-csv results/isetrace/training-size/analysis.csv \
  --report results/isetrace/training-size/report.md \
  --figure results/isetrace/training-size/training-size.pdf
```

The delivered copies under `results/isetrace/training-size/runs/` retain all
resolved configs and per-task rows.

## 7. E10 end-to-end answer transfer

E10 uses 469 human-corrected quality-pass queries, the canonical seed-13
rankings, a 2,048-token evidence budget, and five conditions: Flat Dense-FT,
Provenance-Unit Dense-FT, residual R-GCN, gold oracle, and no evidence.

The frozen `prepared.jsonl`, answers, blind judgments, manifest, and report are
under `results/isetrace/e10-end-to-end-qa/`. The conclusions can be recomputed
offline without API calls:

```bash
uv run python scripts/e10/run_isetrace_end_to_end_qa.py --action report
```

To repeat answer generation or judging, create `.env` from `.env.example` with
`MODEL_ID`, `API_KEY`, and `BASE_URL`, then run the desired resumable stages:

```bash
uv run python scripts/e10/run_isetrace_end_to_end_qa.py --action answer --workers 8
uv run python scripts/e10/run_isetrace_end_to_end_qa.py --action judge --workers 8
```

These stages incur external API cost. `scripts/e10/README.md` documents pacing,
checkpointing, and limited debug runs. Rebuilding the frozen prepared evidence
also requires the seed-13 formal retrieval inputs and prepared test artifact;
the advisor package retains the resulting `prepared.jsonl` instead of the large
processed cache.

## 8. E11 controlled efficiency

E11 is hardware-conditioned. It reuses the completed seed-13 Flat Dense-FT,
Provenance-Unit Dense-FT, Provenance-Unit Cross-Encoder-FT, exact seed, and
residual R-GCN checkpoints; it does not retrain them. To repeat it from scratch,
first regenerate those five seed-13 formal runs and their processed assets.

Run one method at a time on the same otherwise idle GPU:

```bash
uv run python scripts/benchmark_isetrace_efficiency.py \
  --run <seed-13-run-directory> --label <method-label> \
  --device cuda:0 --warmup-queries 20 --repeats 3 \
  --expected-task-count 2000 \
  --output results/isetrace/e11/measurements/<method-label>.json
```

Aggregate the five measurement files with:

```bash
uv run python scripts/report_isetrace_efficiency.py \
  --input flat_dense_ft=results/isetrace/e11/measurements/flat_dense_ft.json \
  --input pu_dense_ft=results/isetrace/e11/measurements/pu_dense_ft.json \
  --input pu_cross_encoder_ft=results/isetrace/e11/measurements/pu_cross_encoder_ft.json \
  --input exact_seed=results/isetrace/e11/measurements/exact_seed.json \
  --input residual_rgcn=results/isetrace/e11/measurements/residual_rgcn.json \
  --output-json results/isetrace/e11/efficiency.json \
  --output-csv results/isetrace/e11/efficiency.csv \
  --output-tex results/isetrace/e11/efficiency-table.tex
```

Different hardware should reproduce the protocol and relative comparison, not
the exact latency, throughput, or peak-memory values measured on the MetaX C500.
The delivered measurement JSON files preserve the paper's original hardware and
artifact identities.

## 9. Delivered result map

```text
results/isetrace/evalv8/                main runs, controls, mechanisms
results/isetrace/chunksweep/             chunk-sweep summaries
results/isetrace/training-size/          60-run robustness study
results/isetrace/e4-audit/               frozen dual-review audit
results/isetrace/e4-human-validation/    human corrections and sensitivity
results/isetrace/e10-end-to-end-qa/      prepared evidence, answers, judgments
results/isetrace/e11/                    controlled efficiency measurements
```

The package is designed to support two levels of review: directly recomputing
paper tables from frozen per-task/results data, and rerunning retrieval training
from the included raw corpus. It does not include external API credentials,
downloaded foundation models, trained checkpoints, or reusable processed caches.
