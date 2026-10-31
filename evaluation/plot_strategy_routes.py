"""Shared strategy metrics and bar panels for the retained paper figures."""
import collections
import json
from pathlib import Path

from matplotlib.ticker import PercentFormatter
import numpy as np


NAMES = ["S1", "S2", "S3", "S4", "S5"]
DATASETS = {
    "sudoku": {
        "long": {
            "S1": "S1 naked singles", "S2": "S2 hidden singles",
            "S3": "S3 crosshatching", "S4": "S4 propagation",
            "S5": "S5 backtracking",
        },
        "bars": ["strategy as\nrequested", "trace legal",
                 "trace rule-\nfaithful", "answer\ncorrect"],
    },
    "maze": {
        "long": {
            "S1": "S1 BFS", "S2": "S2 DFS walk", "S3": "S3 A*",
            "S4": "S4 bidirectional", "S5": "S5 dead-end filling",
        },
        # valid: the trace equals one of the five reference traces;
        # faithful: mean fraction of the requested trace shared as a prefix.
        "bars": ["strategy as\nrequested", "trace is a\nstrategy",
                 "prefix of\nrequested", "answer\ncorrect"],
    },
}
COL = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
CONDITIONS = [
    ("described", "Label in prompt,\nrouter free"),
    ("forced", "Neutral prompt,\npath forced"),
]


def load(run):
    records = []
    run = Path(run)
    root = run / "generations.jsonl"
    paths = [root] if root.exists() else sorted(
        run.glob("shard_*/generations.jsonl")
    )
    for path in paths:
        with path.open() as handle:
            records += [json.loads(line) for line in handle]
    by = collections.defaultdict(list)
    for r in records:
        by[r["condition"]].append(r)
    return by


def dataset_of(by):
    """The corpus the records come from, read off their id field."""
    return "maze" if any(
        "maze_id" in r for rows in by.values() for r in rows
    ) else "sudoku"


def rate(rows, fn):
    return sum(fn(r) for r in rows) / len(rows) if rows else None


def metrics(rows):
    from evaluation.generate_strategy_routes import padding_metrics

    return {
        "strategy": rate([r for r in rows if r["expected"] is not None],
                         lambda r: r["strategy"] == r["expected"]),
        "valid": rate(rows, lambda r: r["valid"]),
        "faithful": rate(rows, lambda r: r.get("faithful", False)),
        "answer": rate(rows, lambda r: r["answer_correct"]),
        **padding_metrics(rows),
    }


def per_strategy(by, cond):
    return {s: metrics([r for r in by.get(cond, []) if r["expected"] == s]) for s in NAMES}


def bar_panel(ax, table, title, spec):
    width = 0.8 / len(table)
    keys = ["strategy", "valid", "faithful", "answer", "padding_fraction"]
    for j, strategy in enumerate(table):
        vals = [table[strategy][k] for k in keys]
        bars = ax.bar(
            np.arange(len(keys)) + (j - (len(table) - 1) / 2) * width,
            [np.nan if v is None else v for v in vals], width * 0.92,
            color=COL[j], label=spec["long"][strategy],
        )
        for bar, value in zip(bars, vals):
            label = "N/A" if value is None else f"{100 * value:.1f}%"
            if value is not None and 0 < value < 0.001:
                label = f"{100 * value:.3f}%"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    (0 if value is None else value) + 0.012, label,
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(spec["bars"] + ["padding\nremoved ↓"])
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if any(row["strategy"] is not None for row in table.values()):
        ax.hlines(0.2, -0.45, 0.45, ls=":", color="grey", lw=0.9)
    ax.set_xlim(-0.65, len(keys) - 0.35)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def confusion(rows, key):
    m = np.zeros((5, 6))
    for r in rows:
        i = NAMES.index(r[key])
        j = NAMES.index(r["strategy"]) if r["strategy"] in NAMES else 5
        m[i, j] += 1
    return m / np.maximum(m.sum(1, keepdims=True), 1)
