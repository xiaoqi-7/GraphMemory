# ISETrace E10 End-to-End QA

This experiment tests whether retrieval improvements transfer to answer quality.
Its frozen contract is:

- 469 human-audited quality-pass test queries
- canonical checkpoint seed 13
- Flat Dense-FT, provenance-unit Dense-FT, and residual R-GCN
- gold-oracle and no-evidence controls
- 2,048 retrieved-evidence tokens using the frozen E5 tokenizer
- one answer call and one blind judge call per condition
- 10,000 paired trajectory-cluster bootstrap samples (seed 13)

The reusable Responses client lives in `graph_memory/infrastructure/responses.py`.
ISETrace E10 contracts, preparation, prompts, reporting, and orchestration live in
`graph_memory/datasets/isetrace/end_to_end_qa/`; the script is only a thin CLI entry.

The frozen E4 inputs are read from `results/isetrace/e4-human-validation/`
(task IDs and corrected metadata) and
`results/isetrace/e4-audit/review_packet.jsonl` (the formal review packet).
`prepare` additionally requires the matching processed formal test artifact and
seed-13 ranked prefixes. The advisor package retains the completed
`prepared.jsonl`, so `report` and resumable `answer`/`judge` do not require
rebuilding that large processed artifact.

The runner reads `MODEL_ID`, `API_KEY`, and `BASE_URL` from the repository `.env`
at the start of `answer` and `judge`. Credentials are never written to artifacts.

```bash
uv run python scripts/e10/run_isetrace_end_to_end_qa.py --action prepare
uv run python scripts/e10/run_isetrace_end_to_end_qa.py --action answer --workers 8
uv run python scripts/e10/run_isetrace_end_to_end_qa.py --action judge --workers 8
uv run python scripts/e10/run_isetrace_end_to_end_qa.py --action report
```

`answer` and `judge` print the starting completed/pending counts, periodic progress,
and a heartbeat after 30 seconds without a completed request. They atomically checkpoint
after each worker-sized batch, so rerunning resumes from persisted digest-matching records.
`report` prints its bootstrap phase, 30-second heartbeat, and elapsed time.

The Responses client proactively spaces requests at `RESPONSES_RPM` (default `9`)
across all threads for the same endpoint/model. HTTP 429 waits for the next paced slot
and retries forever without consuming the ordinary transport retry budget; a transient
lower limit reported by one response does not permanently ratchet down the process.
Set `RESPONSES_RPM` no higher than the gateway's sustained RPM quota.

`--limit N` is a resumability/debugging control for `prepare`, `answer`, and
`judge`, and always counts queries rather than condition records. Therefore
`answer --limit 5` and `judge --limit 5` each target 25 records (five conditions
for each of five queries). A limited `prepare` writes an explicitly limited
manifest; rerun full `prepare` before the formal experiment. Outputs live in
`results/isetrace/e10-end-to-end-qa/`.
