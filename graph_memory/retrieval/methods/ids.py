from __future__ import annotations

from typing import Literal, TypeAlias

from graph_memory.compat import StrEnum

DenseCandidateView: TypeAlias = Literal["flat", "flat_tool_io", "provenance_unit"]
"""Candidate view for text scorers.

``flat`` chunks the whole rendered trajectory; ``flat_tool_io`` chunks only tool
arguments and outputs with the same windows; ``provenance_unit`` uses the
field-aligned provenance units.
"""


class RetrievalMethodId(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    DENSE_FT = "dense_ft"
    CROSS_ENCODER = "cross_encoder"
    GRAPHRAG = "graphrag"
    PROVENANCE_PATH = "provenance_path"
    PROVENANCE_RGCN = "provenance_rgcn"
    DENSE_RGCN_GRAPH_RETRIEVER = "dense_rgcn_graph_retriever"
    DENSE_FT_RGCN_GRAPH_RETRIEVER = "dense_ft_rgcn_graph_retriever"


__all__ = ["DenseCandidateView", "RetrievalMethodId"]
