# Project overview

Graph structure should help recover complete evidence sets or execution paths beyond flat semantic retrieval.

## Domains

| Domain | Datasets | Methods |
|---|---|---|
| Evidence retrieval | HotpotQA, 2Wiki, MuSiQue | BM25, Dense, Dense-FT, GraphRAG, Dense R-GCN, Dense-FT R-GCN |
| Execution provenance | ISETrace frozen natural-query benchmark | BM25, Dense/Dense-FT/Cross-Encoder (`flat|provenance_unit`), GraphRAG, Provenance Path, Residual R-GCN |

The evidence workflow remains runnable. The revision-pinned ISETrace corpus has canonical trajectory adaptation, query-independent provenance graphs, and frozen trajectory-grouped natural-query train/dev/test ownership. The formal test contains 2,000 queries from 1,207 trajectories. A stratified 500-query audit has completed dual-model review, adjudication, and full human validation; paper analyses use the retained human-corrected subsets and five model-training seeds.

| Graph | Owner | Consumers |
|---|---|---|
| `EvidenceGraph` | dataset stage | evidence R-GCN only |
| GraphRAG entity graph | GraphRAG method (private) | GraphRAG only |
| `ProvenanceGraph` | canonical trajectory builder | motif synthesis + `provenance_path` + `provenance_rgcn` |

## Research boundary

- Flat methods are lexical/semantic baselines; the trainable Cross-Encoder is the strongest flat scorer and its provenance-unit variant is a candidate-matched graph-free control.
- GraphRAG is a deterministic retrieval-only FastGraphRAG adaptation: it builds a private noun-phrase co-occurrence graph over small text units, runs query-personalized PageRank, and projects graph scores back to the shared retrieval candidates. It never receives native provenance edges.
- Evidence R-GCN learns independent node scores on `EvidenceGraph`.
- `provenance_rgcn` is a new thin adapter over the maintained evidence R-GCN tensor/batching/optimizer/checkpoint stack; it does not restore the deleted label-conditioned provenance implementation or IDs.
- The new provenance graph is derived only from canonical execution trajectories and does not contain query nodes, answer nodes, labels, or semantic edge weights.
- Namespaced graph kinds plus canonical source spans reserve a clean extension point for later NLP-derived claims, decisions, and semantic relations.
- Motif shape and authoring intent are internal task-planning signals only. The durable benchmark output is the frozen four-field v7 record plus its metadata sidecar; review and human-correction artifacts remain separate so authored labels are not exposed to retrieval.
- The active ISETrace benchmark split is trajectory-disjoint, not intent-component grouped. It takes the natural test trajectories from the deterministic authored-query prefix, then assigns disjoint train/dev trajectories from the remaining authored pool with `split_seed`. It does not claim zero overlap in source-intent IDs or normalized intent text. The legacy `data/isetrace/splits/v1` component manifest is not consumed by the active workflow.
