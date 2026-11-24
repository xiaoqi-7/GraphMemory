"""Generate learning-curve figure for supplementary.tex fig:curves.

Reads rerun-batch train_metrics.jsonl for six conditions × five seeds,
plots dev SpanCov@5 (dev_recall_at_5) as mean ± sample std over seeds,
and saves to docs/raw/overleaf/.../wsdm_cr/figs/learning-curves.pdf.
"""

import json
import pathlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
RUNS = ROOT / "results" / "isetrace" / "wsdm-final" / "runs"
OUT = (
    ROOT
    / "docs"
    / "raw"
    / "overleaf"
    / "6a0c358cc348651e9a5ce8fd"
    / "wsdm_cr"
    / "figs"
    / "learning-curves.pdf"
)

SEEDS = [13, 17, 29, 37, 41]
N_EPOCHS = 15

# (label, run_prefix, color, linestyle, marker)
# Colors from Okabe-Ito palette; marker kept small.
CONDITIONS = [
    ("R-GCN", "isetrace_v7_rgcn_full", "#0072B2", "-", "o"),
    ("Node-local MLP", "isetrace_v7_rgcn_node_mlp", "#E69F00", "-", "s"),
    (r"$-$feeds", "isetrace_v7_rgcn_wo_feeds", "#D55E00", "--", "^"),
    (r"$-$ownership", "isetrace_v7_rgcn_wo_execution_ownership", "#CC79A7", "--", "v"),
    ("Random edges", "isetrace_v7_rgcn_random_edges", "#009E73", ":", "D"),
    ("Homogeneous", "isetrace_v7_rgcn_homogeneous_gcn", "#56B4E9", "-.", "P"),
]


def load_curve(run_prefix: str, seed: int) -> np.ndarray:
    """Return array of length N_EPOCHS+1 (epochs 0..15) of dev_recall_at_5."""
    path = RUNS / f"{run_prefix}_s{seed}" / "training" / "train_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # epoch 0 value is stored in every row's initial_dev_recall_at_5
    epoch0 = rows[0]["initial_dev_recall_at_5"]
    vals = [epoch0] + [r["dev_recall_at_5"] for r in rows]
    assert len(vals) == N_EPOCHS + 1, f"{path}: got {len(vals)} rows"
    return np.array(vals) * 100.0  # fraction -> percent


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
curves = {}
seed_epoch0 = []
for label, prefix, color, ls, marker in CONDITIONS:
    per_seed = np.stack([load_curve(prefix, s) for s in SEEDS])  # (5, 16)
    curves[label] = (per_seed.mean(0), per_seed.std(0, ddof=1), color, ls, marker)
    seed_epoch0.extend(per_seed[:, 0].tolist())

seed_level = float(
    np.mean(seed_epoch0)
)  # epoch-0 is identical across conditions / seeds

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
epochs = np.arange(N_EPOCHS + 1)

fig, ax = plt.subplots(figsize=(3.4, 2.6))

ax.axhline(
    seed_level,
    color="black",
    linewidth=0.8,
    linestyle="--",
    zorder=0,
    label=f"Seed ({seed_level:.1f})",
)

for label, (mean, std, color, ls, marker) in curves.items():
    ax.plot(
        epochs,
        mean,
        color=color,
        linestyle=ls,
        linewidth=1.2,
        marker=marker,
        markersize=3.5,
        markevery=3,
        label=label,
        zorder=2,
    )
    ax.fill_between(
        epochs, mean - std, mean + std, color=color, alpha=0.12, linewidth=0, zorder=1
    )

ax.set_xlim(-0.3, N_EPOCHS + 0.3)
ax.set_xlabel("Epoch", fontsize=8)
ax.set_ylabel("Dev SpanCov@5 (%)", fontsize=8)
ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
ax.tick_params(labelsize=7)
ax.legend(
    fontsize=6.5,
    loc="lower right",
    framealpha=0.85,
    handlelength=1.8,
    handletextpad=0.4,
    borderpad=0.5,
    labelspacing=0.3,
)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

fig.tight_layout(pad=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT}")
