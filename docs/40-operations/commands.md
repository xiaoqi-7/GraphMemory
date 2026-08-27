# Experiment command patterns

Paper-specific inputs, five-seed matrices, Chunk Sweep, E4, mechanisms,
Training-Size, E10, and E11 are consolidated in
[`paper-reproduction.md`](paper-reproduction.md). This file keeps only generic
development commands.

## One job

One Hydra job selects one final method and optional variant, creates one Prefect
flow, and writes one output tree under `runs/<name>/`:

```bash
uv run python experiment/run.py \
  name=isetrace_bm25_smoke dataset=isetrace profile=smoke \
  method=bm25 split_seed=13 device=cuda:0

uv run python experiment/run.py \
  name=isetrace_rgcn_smoke dataset=isetrace profile=smoke \
  method=provenance_unit_dense_ft_rgcn method.variant=full_rgcn \
  seed=13 split_seed=13 device=cuda:0
```

`cache.refresh=true` forces reusable Prefect tasks to re-execute. Device and
worker placement do not enter scientific cache identity.

## Hydra sweeps

A multirun creates independent jobs; it does not combine methods or variants
inside one scientific run:

```bash
uv run python experiment/run.py -m \
  name=isetrace_candidate_views dataset=isetrace profile=full \
  method=dense_ft method.variant=flat,provenance_unit \
  seed=13,17,29,37,41 split_seed=13 device=cuda:0
```

For concurrent multi-GPU work, launch separate jobs with explicit `device=`
values. `CUDA_VISIBLE_DEVICES` may restrict the devices visible to PyTorch.
R-GCN frozen encoding can use all visible CUDA devices when
`encoding.enable_gpupool=true`.

## Inspect configuration

```bash
uv run python experiment/inspect.py kind=configs
uv run python experiment/inspect.py kind=methods
uv run python experiment/inspect.py kind=profiles
uv run python experiment/inspect.py kind=variants
```

A complete run retains:

```text
config/resolved.yaml
config/overrides.yaml
assets/manifest.yaml
workflow/summary.yaml
metrics/final.metrics.csv
metrics/per_task.jsonl
predictions/ranked_prefix.jsonl.gz
training/*                    # trainable methods only
debug/failure_cases.jsonl
```

Large datasets, embeddings, and checkpoints remain content-addressed under
`data/processed/` and are referenced by the run manifest.

## Collect output-only artifacts

```bash
uv run python scripts/deliver/collect_run_artifacts.py \
  --name <run-name> --expected-per-task-count 2000
```

The collector validates and copies small output artifacts without copying
processed datasets or checkpoints.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run basedpyright --level error
uv run python -m compileall -q graph_memory experiment scripts tests
git diff --check
```
