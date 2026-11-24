"""Describe ISETrace candidate views from one prepared split artifact.

Reports, per candidate view (flat, flat_tool_io, provenance_unit):
candidates per trajectory, candidate token-length distribution, short and
capped candidate shares, source-character duplication caused by window overlap,
and how often a gold span is split across more than one candidate.

Inputs are frozen prepare outputs only; no retrieval output is read.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

VIEWS = {
    "flat": "flat_candidates",
    "flat_tool_io": "flat_tool_io_candidates",
    "provenance_unit": "provenance_candidates",
}


def _percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _distribution(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _chars(spans: list[dict]) -> list[tuple[str, str, int, int]]:
    return [
        (
            span["event_id"],
            span.get("json_pointer") or "",
            span["char_start"],
            span["char_end"],
        )
        for span in spans
        if span.get("char_start") is not None and span.get("char_end") is not None
    ]


def describe(
    tasks: list[dict], labels: list[dict], *, cap_tokens: int, short_tokens: int
) -> dict:
    by_graph: dict[str, dict] = {}
    for task in tasks:
        by_graph.setdefault(task["graph_id"], task)
    gold_by_task = {label["task_id"]: label["gold_evidence_spans"] for label in labels}

    report: dict[str, object] = {
        "trajectories": len(by_graph),
        "queries": len(tasks),
        "cap_tokens": cap_tokens,
        "short_tokens": short_tokens,
        "views": {},
    }
    for view, field in VIEWS.items():
        lengths: list[int] = []
        per_trajectory: list[int] = []
        tokens_per_trajectory: list[int] = []
        covered_chars = 0
        emitted_chars = 0
        for record in by_graph.values():
            candidates = record.get(field) or []
            if not candidates:
                raise ValueError(
                    f"{record['graph_id']} has no {field}; re-prepare the split"
                )
            per_trajectory.append(len(candidates))
            counts = [
                int(candidate["metadata"]["token_count"]) for candidate in candidates
            ]
            lengths.extend(counts)
            tokens_per_trajectory.append(sum(counts))
            unique: dict[tuple[str, str], set[int]] = defaultdict(set)
            for candidate in candidates:
                for event_id, pointer, start, end in _chars(candidate["source_spans"]):
                    emitted_chars += end - start
                    unique[(event_id, pointer)].update(range(start, end))
            covered_chars += sum(len(offsets) for offsets in unique.values())

        split_spans = 0
        gold_spans = 0
        for task in tasks:
            candidates = task.get(field) or []
            for event_id, pointer, start, end in _chars(gold_by_task[task["task_id"]]):
                gold_spans += 1
                touching = 0
                fully_inside = False
                for candidate in candidates:
                    for c_event, c_pointer, c_start, c_end in _chars(
                        candidate["source_spans"]
                    ):
                        if (c_event, c_pointer) != (event_id, pointer):
                            continue
                        if c_start < end and start < c_end:
                            touching += 1
                            fully_inside = fully_inside or (
                                c_start <= start and end <= c_end
                            )
                if not fully_inside and touching > 1:
                    split_spans += 1

        report["views"][view] = {
            "candidates_per_trajectory": _distribution(per_trajectory),
            "candidate_tokens": _distribution(lengths),
            "tokens_per_trajectory": _distribution(tokens_per_trajectory),
            "share_at_or_below_short": sum(n <= short_tokens for n in lengths)
            / len(lengths),
            "share_at_cap": sum(n >= cap_tokens for n in lengths) / len(lengths),
            "duplicated_source_char_share": 1.0 - covered_chars / emitted_chars,
            "gold_spans": gold_spans,
            "gold_spans_split_across_candidates": split_spans,
            "gold_span_split_share": split_spans / gold_spans if gold_spans else 0.0,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--cap-tokens", type=int, default=504)
    parser.add_argument("--short-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = json.loads((args.prepared_dir / "tasks.json").read_text())
    labels = json.loads((args.prepared_dir / "labels.json").read_text())
    report = describe(
        tasks, labels, cap_tokens=args.cap_tokens, short_tokens=args.short_tokens
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
