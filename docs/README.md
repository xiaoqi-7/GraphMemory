# Graph Memory documentation

## Paper reproduction

Start here when reviewing or reproducing the current ISETrace paper:

- [`40-operations/paper-reproduction.md`](40-operations/paper-reproduction.md) — environment, automatic model provisioning, frozen data, Hydra run matrices, E4 transformations, E10/E11, and delivered result locations.
- [`../results/isetrace/README.md`](../results/isetrace/README.md) — physical layout of the locked run artifacts.
- [`raw/paper-v2.tex`](raw/paper-v2.tex) / `raw/paper-v2.pdf` — current manuscript source and compiled paper.

The advisor delivery contains this reproduction entry point, the current paper,
and the small set of architecture/contracts below. Historical plans, OpenSpec
changes, imported drafts, server notes, and implementation handoffs are
repository development material and are not part of the advisor package.

## Maintained technical reference

| Path | Purpose |
|---|---|
| [`00-overview/project-overview.md`](00-overview/project-overview.md) | Current domains, methods, and scientific boundary |
| [`20-contracts/data-contracts.md`](20-contracts/data-contracts.md) | Dataset, graph, and artifact contracts |
| [`20-contracts/retrieval-contracts.md`](20-contracts/retrieval-contracts.md) | Method/request/ranking matrix |
| [`30-design/architecture.md`](30-design/architecture.md) | Package ownership and execution flow |
| [`40-operations/commands.md`](40-operations/commands.md) | Generic development command patterns |

## Detailed developer operations

The remaining files under `40-operations/` document individual implementation
lifecycles, naming conventions, data authoring, cache behavior, and historical
run procedures. They are useful while changing the system, but
`paper-reproduction.md` is the canonical paper-facing protocol when wording or
seed counts differ.

Code and Pydantic contracts remain the authority for schemas and runtime
behavior; documentation owns the scientific run matrix and interpretation.
