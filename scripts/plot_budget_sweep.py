"""Regenerate figs/budget-sweep.pdf for the supplementary material.

Reads rerun-batch analysis JSONs (views.json + residual-vs-seed.json) and
draws the 2x1 Coverage / Full Support panel figure, now including the
tool-field flat Dense-FT (B) curve in addition to flat (A), provenance-unit
(D), and the R-GCN residual.
"""

import json
import pathlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = pathlib.Path(__file__).parent.parent
ANALYSIS = ROOT / "results" / "isetrace" / "wsdm-final" / "analysis"
OUT = (
    ROOT
    / "docs"
    / "raw"
    / "overleaf"
    / "6a0c358cc348651e9a5ce8fd"
    / "wsdm_cr"
    / "figs"
    / "budget-sweep.pdf"
)

BUDGETS = [256, 512, 1024, 2048, 4096, 8192]


def extract(summary, metric_prefix, budgets=BUDGETS):
    """Return (means, stds) in percent for a given summary block."""
    metrics = summary["metrics"]
    means, stds = [], []
    for b in budgets:
        m = metrics[f"{metric_prefix}@{b} Tokens"]
        mean = m.get("mean", m.get("value", 0.0))
        std = m.get("std", 0.0) or 0.0
        means.append(100.0 * mean)
        stds.append(100.0 * std)
    return np.array(means), np.array(stds)


views = json.loads((ANALYSIS / "views.json").read_text())
rseed = json.loads((ANALYSIS / "residual-vs-seed.json").read_text())

# Build a combined summary dict: label -> (means, stds, color, marker, ls)
# Okabe-Ito palette; (B) is added with a distinct marker.
series = [
    ("Flat Dense-FT (A)", views["summary"]["flat"], "#0072B2", "o", "-"),
    ("Tool-field flat (B)", views["summary"]["flat_tool_io"], "#56B4E9", "P", "--"),
    ("Prov.-Unit Dense-FT (D)", views["summary"]["pu"], "#E69F00", "s", "-"),
    ("R-GCN residual", rseed["summary"]["full_rgcn"], "#009E73", "D", "-"),
    ("Seed (D passthrough)", rseed["summary"]["seed"], "#999999", "^", ":"),
]

fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True)

for ax, (prefix, ylabel) in zip(
    axes, [("Coverage", "Coverage (%)"), ("Full Support", "Full Support (%)")]
):
    for label, block, color, marker, ls in series:
        means, stds = extract(block, prefix)
        ax.plot(
            BUDGETS,
            means,
            color=color,
            linestyle=ls,
            linewidth=1.1,
            marker=marker,
            markersize=3.0,
            label=label,
        )
        if any(stds > 0):
            ax.fill_between(
                BUDGETS,
                means - stds,
                means + stds,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

axes[-1].set_xlabel("Context budget (tokens)", fontsize=8)
axes[-1].legend(
    fontsize=6.0,
    loc="best",
    framealpha=0.85,
    handlelength=1.8,
    handletextpad=0.4,
    borderpad=0.5,
    labelspacing=0.25,
    ncol=1,
)

fig.tight_layout(pad=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT}")
