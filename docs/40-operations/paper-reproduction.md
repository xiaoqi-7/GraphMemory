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
Prefect flow. Missing dataset files and encoder directories are downloaded from
the public Hugging Face Hub at fixed revisions:

| Input | Hugging Face source | Local path |
|---|---|---|
| ISETrace trajectories | `valiere/ISETrace` revision `e40e04d41c04e4eb4bae181ebdd41b61c688081b` | `data/isetrace/raw/trajectories/` |
| Frozen memory queries and metadata | `HazeLocus/ISETrace-Memory-Queries` revision `3130aeeaeb2b8d292c29f4c8e23303be7463732c` | `data/isetrace/query-authoring/` |
| Dense, Dense-FT, GraphRAG, Provenance Path, R-GCN encoder | `intfloat/e5-base-v2` revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | `models/intfloat-e5-base-v2` |
| Cross-Encoder-FT backbone | `BAAI/bge-reranker-base` revision `2cfc18c9415c912f9d8155881c133215df768a70` | `models/BAAI-bge-reranker-base` |

Downloaded models, checkpoints, embeddings, and `data/processed/` are not part
of the supplementary package; selected workflows materialize them as needed.

Canonical experiment inputs:

```text
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

### ISETrace counts at each stage

Verified on 2026-11-14 against the pinned revision, the public dataset card, and
the local pipeline. Raw shards were downloaded on 2026-07-30.

| Stage | Trajectories | Queries |
|---|---:|---:|
| Raw revision `e40e04d` (8 shards: 7 × 3,000 + 2,132; current `main` head) | 23,132 | — |
| Pass canonicalization (164 fail event–field invariants) | 22,968 | — |
| Authoring input: first shuffled trajectories (seed 13) | 6,010 | — |
| Authoring accepted (4,836 tasks rejected: 2,681 validation exhausted, 2,128 model rejection, 27 trajectory rejection) | 4,013 | 6,597 |
| Resolved to a canonical trajectory (123 queries uncompilable; 44 trajectories lose all queries) | 3,969 | 6,474 |
| Assigned to train/dev/test (no trajectory left unused) | 3,969 | 6,474 |

The dataset card states 23,132 trajectories. The Hugging Face dataset viewer
reports 22,847 rows for the `trajectories` config because its Parquet
conversion is partial (`"partial": true` from
`datasets-server.huggingface.co/size`). This explains the lower number on the
public page. The stage counts above can be reproduced by replaying
`_match_query_contexts` and `allocate_trajectory_splits` in
`graph_memory/datasets/isetrace/benchmark_adapter.py`. The prepared artifact's
`counts.json` records the same `trajectories_seen/accepted/rejected` and
`queries_resolved/uncompilable` values.

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

The camera-ready tables come from the rerun in Section 3, which repeats this
matrix and adds the tool-field view, the node-local MLP, and the trained-only
controls.

## 3. Camera-ready rerun

The camera-ready paper takes every row of the main, attribution, and relation
tables, and every residual, query-type, human-subset, mechanism, and efficiency
analysis, from one rerun on the same prepared test artifact. The original
submission runs under `results/isetrace/evalv8/` are still the source of the
chunk-size sweep, the training-size study, and the answer study, which the
paper labels "original batch".

The rerun adds three conditions to Section 2:

- the tool-field flat view (B), which keeps only tool arguments and outputs;
- the parameter-matched node-local MLP residual;
- a trained-only checkpoint rule for the controls whose development selection
  keeps epoch 0.

### 3.1 Run matrix

Every job uses the same base command. Trainable methods run with seeds
`13 17 29 37 41`; deterministic methods run once with `seed=13`:

```bash
uv run python experiment/run.py dataset=isetrace profile=full split_seed=13 \
  seed=<seed> name=<run-name> device=cuda:0 <overrides>
```

| Run name (`_s<seed>` for trainable) | Overrides | Paper condition |
|---|---|---|
| `isetrace_v7_dense_ft_flat` | `method=dense_ft method.variant=flat` | Dense-FT (A) |
| `isetrace_v7_dense_ft_flat_tool_io` | `method=dense_ft method.variant=flat_tool_io` | Dense-FT (B) |
| `isetrace_v7_dense_ft_pu` | `method=dense_ft method.variant=provenance_unit` | Dense-FT (D) |
| `isetrace_v7_cross_encoder_flat` | `method=cross_encoder method.variant=flat` | Cross-Encoder-FT (A) |
| `isetrace_v7_cross_encoder_flat_tool_io` | `method=cross_encoder method.variant=flat_tool_io` | Cross-Encoder-FT (B) |
| `isetrace_v7_cross_encoder_pu` | `method=cross_encoder method.variant=provenance_unit` | Cross-Encoder-FT (D) |
| `isetrace_v7_rgcn_full` | `method=provenance_unit_dense_ft_rgcn method.variant=full_rgcn` | R-GCN residual |
| `isetrace_v7_rgcn_wo_graph` | same method, `method.variant=wo_graph` | Seed passthrough |
| `isetrace_v7_rgcn_node_mlp` | same method, `method.variant=node_mlp` | Node-local MLP residual |
| `isetrace_v7_rgcn_<control>` | same method, `method.variant=<control>` | Relation and topology controls |
| `isetrace_v7_rgcn_<control>_trained` | same method, `method.variant=<control> method.train.selection.include_initial_state=false` | Trained-only rows of the relation table |
| `isetrace_v7_bm25` | `method=bm25` | BM25 (A) |
| `isetrace_v7_dense_flat` | `method=dense method.variant=flat` | Dense (A) |
| `isetrace_v7_dense_flat_tool_io` | `method=dense method.variant=flat_tool_io` | Dense (B) |
| `isetrace_v7_dense_pu` | `method=dense method.variant=provenance_unit` | Dense (D) |
| `isetrace_v7_graphrag` | `method=graphrag` | Semantic-graph expansion (A) |
| `isetrace_v7_provenance_path` | `method=provenance_path` | Provenance Path (D) |

The `<control>` values are:

- for `isetrace_v7_rgcn_<control>`: `homogeneous_gcn`, `wo_feeds`,
  `wo_execution_ownership`, `wo_artifact_io`, `wo_chunk_adjacency`, and
  `random_edges`;
- for `isetrace_v7_rgcn_<control>_trained`: `node_mlp`, `wo_feeds`,
  `wo_execution_ownership`, and `random_edges`. These are the controls whose
  default selection keeps epoch 0 for every seed.

The paper's "semantic-graph expansion" is the `graphrag` method, an adaptation
of the LLM-free FastGraphRAG indexing mode. The `node_mlp` variant sets its
hidden width automatically to match the R-GCN's trainable parameter count
(width 684; 3.342M versus 3.346M parameters).

For each seed, run `isetrace_v7_rgcn_full` first. It materializes that seed's
Provenance-Unit Dense-FT model, training pairs, and frozen embeddings in the
Prefect cache. The Provenance-Unit Dense-FT run and every other
`isetrace_v7_rgcn_*` run of the same seed then reuse those assets instead of
rebuilding them. The flat Dense-FT and cross-encoder runs do not depend on this
order.

The completed runs are kept locally under `results/isetrace/wsdm-final/runs/`.
Each holds its resolved config, Hydra overrides, run log, training metrics,
final metrics, 2,000 per-task rows, and ranked predictions. They do not include
checkpoints.

### 3.2 Aggregation

All paired analyses use `scripts/aggregate_main_results.py` with 10,000
trajectory-cluster bootstrap replicates and bootstrap seed 13. The helper below
expands `LABEL=RUN_PREFIX` pairs into one `--run` argument per seed:

```bash
R=results/isetrace/wsdm-final/runs
A=results/isetrace/wsdm-final/analysis
H=results/isetrace/e4-human-validation
runs() { for s in 13 17 29 37 41; do for p in "$@"; do
  printf -- '--run %s=%s/%s_s%s ' "${p%%=*}" "$R" "${p#*=}" "$s"; done; done; }
agg() { uv run python scripts/aggregate_main_results.py \
  --expected-task-count 2000 --bootstrap-samples 10000 --bootstrap-seed 13 "$@"; }
```

Candidate views under Dense-FT give the Dense-FT rows of the attribution table.
`views.json` reports (D) and (A) against (B); `views-vs-flat.json` reports (B)
and (D) against (A):

```bash
V="flat=isetrace_v7_dense_ft_flat flat_tool_io=isetrace_v7_dense_ft_flat_tool_io pu=isetrace_v7_dense_ft_pu"
agg $(runs $V) --trainable flat --trainable flat_tool_io --trainable pu \
  --baseline flat_tool_io --output $A/views.json --output-csv $A/views.csv
agg $(runs $V) --trainable flat --trainable flat_tool_io --trainable pu \
  --baseline flat --output $A/views-vs-flat.json --output-csv $A/views-vs-flat.csv
```

Cross-Encoder-FT views and the R-GCN-versus-cross-encoder comparison:

```bash
C="ce_flat=isetrace_v7_cross_encoder_flat ce_flat_tool_io=isetrace_v7_cross_encoder_flat_tool_io ce_pu=isetrace_v7_cross_encoder_pu full_rgcn=isetrace_v7_rgcn_full"
T="--trainable ce_flat --trainable ce_flat_tool_io --trainable ce_pu --trainable full_rgcn"
agg $(runs $C) $T --baseline ce_flat_tool_io --output $A/ce-views.json
agg $(runs $C) $T --baseline ce_flat --output $A/ce-views-vs-flat.json
agg $(runs $C) $T --baseline ce_pu --output $A/rgcn-vs-ce-pu.json
```

The training-free baselines are deterministic, so they need no `--trainable`
flag. The four outputs differ only in the baseline:

```bash
F="--run bm25=$R/isetrace_v7_bm25 --run dense_flat=$R/isetrace_v7_dense_flat \
  --run dense_flat_tool_io=$R/isetrace_v7_dense_flat_tool_io \
  --run dense_pu=$R/isetrace_v7_dense_pu --run graphrag=$R/isetrace_v7_graphrag \
  --run provenance_path=$R/isetrace_v7_provenance_path"
agg $F --baseline dense_flat_tool_io --output $A/training-free.json
agg $F --baseline dense_flat --output $A/training-free-vs-flat.json
agg $F --baseline bm25 --output $A/training-free-vs-bm25.json
agg $F --baseline dense_pu --output $A/training-free-vs-dense-pu.json
```

Residual comparison. `residual.json` reports R-GCN minus MLP;
`residual-vs-seed.json` reports both residuals minus the seed:

```bash
M="full_rgcn=isetrace_v7_rgcn_full node_mlp=isetrace_v7_rgcn_node_mlp seed=isetrace_v7_rgcn_wo_graph"
T="--trainable full_rgcn --trainable node_mlp --trainable seed"
agg $(runs $M) $T --baseline node_mlp --output $A/residual.json --output-csv $A/residual.csv
agg $(runs $M) $T --baseline seed --output $A/residual-vs-seed.json --output-csv $A/residual-vs-seed.csv
```

Relation controls, reported as full model minus control:

```bash
X="full_rgcn=isetrace_v7_rgcn_full"
for c in homogeneous_gcn wo_feeds wo_execution_ownership wo_artifact_io wo_chunk_adjacency random_edges; do
  X="$X $c=isetrace_v7_rgcn_$c"; done
agg $(runs $X) $(for p in $X; do echo --trainable ${p%%=*}; done) \
  --baseline full_rgcn --delta-direction baseline-minus-method \
  --output $A/relations.json --output-csv $A/relations.csv
```

Trained-only controls. `trained-only-vs-full.json` reports full minus each
trained control; `trained-only-vs-seed.json` reports each trained control minus
the seed:

```bash
X="full_rgcn=isetrace_v7_rgcn_full seed=isetrace_v7_rgcn_wo_graph"
for c in node_mlp wo_feeds wo_execution_ownership random_edges; do
  X="$X ${c}_trained=isetrace_v7_rgcn_${c}_trained"; done
T=$(for p in $X; do echo --trainable ${p%%=*}; done)
agg $(runs $X) $T --baseline full_rgcn --delta-direction baseline-minus-method \
  --output $A/trained-only-vs-full.json
agg $(runs $X) $T --baseline seed --output $A/trained-only-vs-seed.json
```

Human-audit subsets (Section 5 defines them). The 367-query human-verified
subset filters the view and residual runs:

```bash
agg $(runs $V) --trainable flat --trainable flat_tool_io --trainable pu \
  --baseline flat_tool_io --task-ids $H/human_verified_task_ids.json \
  --output $A/human-verified-views.json
agg $(runs $M) --trainable full_rgcn --trainable node_mlp --trainable seed \
  --baseline seed --task-ids $H/human_verified_task_ids.json \
  --output $A/human-verified-residual.json
```

The human-type analysis groups the 469 quality-pass queries by their
human-corrected type, using these steps:

1. Copy the `isetrace_v7_rgcn_full_s*` and `isetrace_v7_rgcn_wo_graph_s*` runs.
2. In each copy's `metrics/per_task.jsonl`, replace `memory_mode` with the
   corrected type from `$H/human_corrected_query_metadata.jsonl`, joining
   `task_id` to `query_id`.
3. Aggregate the copies with `--trainable full_rgcn --trainable seed
   --baseline seed --task-ids $H/human_quality_pass_task_ids.json`. Write the
   output to `$A/human-type-residual.json`.

The original per-task files are not modified.

The mechanism and dispersion strata use the Section 6 command. Pass the
`isetrace_v7_dense_ft_flat`, `isetrace_v7_dense_ft_pu`,
`isetrace_v7_rgcn_wo_graph`, and `isetrace_v7_rgcn_full` runs of this rerun,
and write the results to `$A/mechanisms.json` and `$A/mechanisms.csv`.

The candidate-view statistics in the paper's Table 1 are computed from the
prepared test artifact only. Use the test dataset directory referenced in any
rerun's `assets/manifest.yaml`:

```bash
uv run python scripts/report_isetrace_candidate_views.py \
  --prepared-dir <processed-test-dataset-directory> \
  --cap-tokens 504 --short-tokens 64 --output $A/candidate-views.json
```

Efficiency measurements and per-epoch training time for this rerun are
described in Section 9.

## 4. Flat chunk-size sweep

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

## 5. E4 audit and human correction

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

## 6. Mechanism and evidence-dispersion analysis

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

## 7. Training-size robustness

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

## 8. E10 end-to-end answer transfer

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

## 9. E11 controlled efficiency

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

For the camera-ready revision, the seed-13 exact seed, residual R-GCN, and
node-local MLP runs (`isetrace_v7_rgcn_{wo_graph,full,node_mlp}_s13`) were
re-measured with the same command on one otherwise idle NVIDIA A800. The
measurements are in `results/isetrace/wsdm-final/efficiency/`. Training time
per epoch comes from two timing-only reruns,
`isetrace_v7_timing_{full_rgcn,node_mlp}_s13`, which use
`method.train.selection.best_metric=dev_mrr` to obtain a fresh training cache
key. Read `train_elapsed_seconds` in each
`runs/<name>/training/train_metrics.jsonl`.

Different hardware should reproduce the protocol and relative comparison, not
the exact latency, throughput, or peak-memory values measured on the MetaX C500.
The delivered measurement JSON files preserve the paper's original hardware and
artifact identities.

## 10. Delivered result map

```text
results/isetrace/wsdm-final/            camera-ready rerun: runs, analysis, efficiency
results/isetrace/evalv8/                submission runs, controls, mechanisms
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
