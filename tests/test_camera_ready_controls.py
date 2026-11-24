"""Controls added for the WSDM camera-ready attribution experiments.

1. ``flat_tool_io``: fixed flat windows restricted to tool arguments/outputs, so the
   flat view has the same field eligibility as provenance units.
2. ``node_mlp``: a parameter-matched residual that never reads graph edges.
"""

from __future__ import annotations

from dataclasses import replace
from typing import get_args

import pytest
import torch

from graph_memory.datasets.isetrace import (
    ISETraceLabelRecord,
    ISETraceRankingRecord,
    adapt_dense_training_split,
    adapt_isetrace_record,
    parse_isetrace_record,
)
from graph_memory.datasets.isetrace.retrieval_views import (
    flat_trajectory_candidates,
    provenance_unit_candidates,
)
from graph_memory.datasets.selection import text_ranking_requests_for_dataset
from graph_memory.graphs.provenance import build_provenance_graph
from graph_memory.models.graph_retriever.batching import collate_evidence_tasks
from graph_memory.models.graph_retriever.factory import build_model_from_config
from graph_memory.models.graph_retriever.internals.contracts import TrainingBatch
from graph_memory.models.graph_retriever.provenance import (
    PROVENANCE_RELATION_VOCAB,
    matched_node_mlp_hidden_dim,
    provenance_rgcn_model_config,
    tensorize_provenance_ranking_task,
)
from graph_memory.retrieval.methods.ids import DenseCandidateView
from graph_memory.trajectories import SourceSpan
from tests.isetrace_fixtures import isetrace_record
from tests.test_isetrace_nontrain_retrieval import CharacterOffsetTokenizer
from tests.test_provenance_rgcn_tensorization import (
    DeterministicEmbeddingProvider,
    _graph_and_request,
)

_TOOL_IO_POINTERS = {"/raw_arguments", "/content"}


def _trajectory():
    return adapt_isetrace_record(
        parse_isetrace_record(isetrace_record()), source_revision="fixture-revision"
    )


def _flat(trajectory, **kwargs):
    return flat_trajectory_candidates(
        trajectory,
        tokenizer=CharacterOffsetTokenizer(),
        max_tokens=64,
        overlap_tokens=16,
        **kwargs,
    )


# --- flat_tool_io candidate view ---------------------------------------------


def test_tool_io_flat_view_excludes_messages_and_reasoning() -> None:
    trajectory = _trajectory()
    tool_event_ids = {call.event_id for call in trajectory.tool_calls} | {
        output.event_id for output in trajectory.tool_outputs
    }

    candidates = _flat(trajectory, field_scope="tool_io")

    spans = [span for candidate in candidates for span in candidate.source_spans]
    assert spans
    assert {span.event_id for span in spans} <= tool_event_ids
    assert {span.json_pointer for span in spans} <= _TOOL_IO_POINTERS
    assert all("[message role=" not in candidate.text for candidate in candidates)
    assert all(
        candidate.metadata["representation"] == "flat_tool_io_chunk"
        for candidate in candidates
    )


def test_tool_io_flat_view_covers_the_same_source_characters_as_provenance_units() -> None:
    trajectory = _trajectory()

    def covered(candidates) -> set[tuple[str, str | None, int]]:
        return {
            (span.event_id, span.json_pointer, offset)
            for candidate in candidates
            for span in candidate.source_spans
            if span.char_start is not None and span.char_end is not None
            for offset in range(span.char_start, span.char_end)
        }

    tool_io = covered(_flat(trajectory, field_scope="tool_io"))
    provenance = covered(provenance_unit_candidates(build_provenance_graph(trajectory)))

    assert tool_io == provenance


def test_default_flat_view_is_unchanged_by_the_field_scope_option() -> None:
    trajectory = _trajectory()

    default = _flat(trajectory)
    explicit = _flat(trajectory, field_scope="all")
    tool_io = _flat(trajectory, field_scope="tool_io")

    assert default == explicit
    assert all(candidate.item_id.startswith("flat-chunk:") for candidate in default)
    assert all(
        candidate.item_id.startswith("flat-tool-io-chunk:") for candidate in tool_io
    )
    assert any(
        span.json_pointer == "/reasoning_content"
        for candidate in default
        for span in candidate.source_spans
    )


def _ranking_and_label(trajectory, *, with_tool_io: bool = True):
    output = trajectory.tool_outputs[0]
    ranking = ISETraceRankingRecord(
        task_id="tool-io-task",
        graph_id=trajectory.trajectory_id,
        query_text="What did the first tool return?",
        flat_candidates=_flat(trajectory),
        provenance_candidates=provenance_unit_candidates(
            build_provenance_graph(trajectory)
        ),
        flat_tool_io_candidates=(
            _flat(trajectory, field_scope="tool_io") if with_tool_io else ()
        ),
    )
    label = ISETraceLabelRecord(
        task_id=ranking.task_id,
        graph_id=ranking.graph_id,
        gold_evidence_spans=(
            SourceSpan(
                event_id=output.event_id,
                json_pointer="/content",
                char_start=0,
                char_end=len(output.content),
            ),
        ),
    )
    return ranking, label


def test_each_view_routes_to_its_own_candidates_in_training_and_retrieval() -> None:
    ranking, label = _ranking_and_label(_trajectory())
    expected: dict[DenseCandidateView, object] = {
        "flat": ranking.flat_candidates,
        "flat_tool_io": ranking.flat_tool_io_candidates,
        "provenance_unit": ranking.provenance_candidates,
    }
    assert set(expected) == set(get_args(DenseCandidateView))

    for view, candidates in expected.items():
        requests, labels = adapt_dense_training_split([ranking], [label], view=view)
        retrieval = text_ranking_requests_for_dataset(
            "isetrace",
            [ranking.model_dump(mode="json")],
            isetrace_representation=view,
        )
        assert requests[0].candidates == candidates
        assert retrieval[0].candidates == candidates
        assert labels[0].gold_evidence_item_ids
        assert set(labels[0].gold_evidence_item_ids) <= {
            candidate.item_id for candidate in requests[0].candidates
        }


def test_legacy_artifact_without_tool_io_candidates_fails_loudly() -> None:
    ranking, label = _ranking_and_label(_trajectory(), with_tool_io=False)

    _ = adapt_dense_training_split([ranking], [label], view="flat")
    with pytest.raises(ValueError, match="re-prepare"):
        adapt_dense_training_split([ranking], [label], view="flat_tool_io")


# --- node_mlp residual -------------------------------------------------------


def _config(ablation_name: str, *, hidden_dim: int = 8, encoder_dim: int = 4):
    return provenance_rgcn_model_config(
        encoder_model="fake-encoder",
        encoder_dim=encoder_dim,
        query_prefix="query: ",
        passage_prefix="passage: ",
        encoder_batch_size=8,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.0,
        ablation_name=ablation_name,
    )


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_node_mlp_matches_paper_rgcn_parameter_count_within_one_percent() -> None:
    rgcn = build_model_from_config(_config("full_rgcn", hidden_dim=256, encoder_dim=768))
    mlp_config = _config("node_mlp", hidden_dim=256, encoder_dim=768)
    mlp = build_model_from_config(mlp_config)

    rgcn_params = _parameter_count(rgcn)
    mlp_params = _parameter_count(mlp)

    assert mlp_config.graph_encoder_type == "node_mlp"
    assert mlp_config.hidden_dim == matched_node_mlp_hidden_dim(
        reference_hidden_dim=256,
        num_layers=3,
        num_relations=len(PROVENANCE_RELATION_VOCAB),
        encoder_dim=768,
    )
    assert abs(mlp_params - rgcn_params) / rgcn_params < 0.01


def _batch_and_task(ablation_name: str):
    config = _config(ablation_name)
    _graph, request = _graph_and_request(
        task_id=f"{ablation_name}-task", query_text="Find the relevant output."
    )
    provider = DeterministicEmbeddingProvider()
    task = tensorize_provenance_ranking_task(
        request,
        model_config=config,
        text_embedding_provider=provider,
        seed_signal_provider=provider,
    )
    return config, collate_evidence_tasks([task]), task


def test_node_mlp_starts_as_exact_seed_passthrough() -> None:
    config, batch, task = _batch_and_task("node_mlp")
    model = build_model_from_config(config)

    assert config.scoring_mode == "seed_residual"
    assert torch.equal(model(batch), task.sample_node_features[:, 0])


def _without_edges(batch: TrainingBatch) -> TrainingBatch:
    graph = batch.graph_batch
    return replace(
        batch,
        graph_batch=replace(
            graph,
            edge_index=graph.edge_index[:, :0],
            relation_ids=graph.relation_ids[:0],
            edge_weights=graph.edge_weights[:0],
        ),
    )


def _randomized_outputs(ablation_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    config, batch, _task = _batch_and_task(ablation_name)
    assert batch.graph_batch.edge_index.numel() > 0
    torch.manual_seed(13)
    model = build_model_from_config(config)
    with torch.no_grad():
        for parameter in model.parameters():
            _ = parameter.normal_()
    _ = model.eval()
    with torch.no_grad():
        return model(batch), model(_without_edges(batch))


def test_node_mlp_output_ignores_graph_edges() -> None:
    with_edges, without_edges = _randomized_outputs("node_mlp")

    assert torch.equal(with_edges, without_edges)


def test_rgcn_output_does_depend_on_graph_edges() -> None:
    """Sanity check that the edge-removal probe can detect message passing."""

    with_edges, without_edges = _randomized_outputs("full_rgcn")

    assert not torch.equal(with_edges, without_edges)
