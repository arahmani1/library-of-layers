#!/usr/bin/env python3
"""Plot seed-averaged active-match training loss against cumulative FLOPs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

if __package__:
    from . import loss_history, loss_router_plot
    from .local_data import ASSETS, load_manifest, provenance
else:
    import loss_history
    import loss_router_plot
    from local_data import ASSETS, load_manifest, provenance

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAPER_RC = loss_router_plot.PAPER_RC


def load_run(specification, data_dir=ASSETS) -> dict[str, object]:
    """Read one complete history from the selected local data directory."""
    return {
        **loss_history.load_curve(specification, data_dir),
        "source": json.dumps(provenance(data_dir, [specification["file"]])[0], sort_keys=True),
    }


def aggregate_group(
    specification, curves, smoothing_window: int,
) -> dict[str, object]:
    """Smooth aligned seeds twice, then compute their pointwise mean and range."""

    if len(curves) != 3:
        raise ValueError("each plotted group must contain exactly three seeds")
    if len({curve["seed"] for curve in curves}) != 3:
        raise ValueError("each plotted group must contain three distinct seeds")
    reference_steps = curves[0]["step"]
    reference_flops = curves[0]["x"]
    for curve in curves[1:]:
        if not np.array_equal(curve["step"], reference_steps):
            raise ValueError(f"{specification['label']} seed steps are not aligned")
        if not np.array_equal(curve["x"], reference_flops):
            raise ValueError(f"{specification['label']} seed FLOPs are not aligned")

    smoothed = np.stack([
        loss_history.running_average(
            loss_history.running_average(curve["y"], smoothing_window), smoothing_window,
        )
        for curve in curves
    ])
    final_val_losses = np.asarray(
        [curve["final_val_nll"] for curve in curves], dtype=np.float64,
    )
    if not np.isfinite(final_val_losses).all() or np.any(final_val_losses <= 0):
        raise ValueError("terminal validation NLLs must be finite and positive")
    return {
        "label": specification["label"],
        "color": specification["color"],
        "x": reference_flops,
        "mean": smoothed.mean(axis=0),
        "minimum": smoothed.min(axis=0),
        "maximum": smoothed.max(axis=0),
        "final_val_mean": final_val_losses.mean(),
        "final_val_minimum": final_val_losses.min(),
        "final_val_maximum": final_val_losses.max(),
        "sources": tuple(curve["source"] for curve in curves),
    }


def make_figure(groups, router_heatmap):
    """Render prepared seed means and ranges with the shared paper layout."""

    curves = []
    for group in groups:
        tail = group["x"] >= 2.5e18
        curves.append({
            **group,
            **{key: group[key][tail] for key in ("x", "minimum", "maximum")},
            "y": group["mean"][tail],
        })
    figure = loss_router_plot.make_figure(
        curves, router_heatmap, loss_ylim=None,
        figsize=(16.5, 11.5), width_ratios=(1.55, 2.2), wspace=0.22,
    )
    if curves:
        figure.axes[0].set_xlim(
            2.5,
            max(group["x"][-1] / 1e18 for group in groups),
        )
        figure.axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    for band in figure.axes[0].collections:
        band.set_alpha(0.60)
    for line in figure.axes[0].lines:
        line.set_linewidth(5.0)
    figure.axes[1].set_title("")
    for annotation in figure.axes[1].texts:
        annotation.set_fontsize(14.0)
    for axes in figure.axes:
        axes.tick_params(axis="both", which="both", labelsize=24.0)
        axes.xaxis.label.set_fontsize(28.0)
        axes.yaxis.label.set_fontsize(28.0)
    figure.legends[0].remove()
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.28),
        borderaxespad=0.0, frameon=False, ncol=2,
        prop={"family": loss_router_plot.FONT_FAMILY, "size": 24.0},
    )
    return figure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window", type=int, default=1000,
        help="trailing average per pass (two passes; default: 1000; 1 disables)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ASSETS,
        help="local data directory containing datasets.json",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".pdf"),
        help="output PDF (default: plotting/lol_vs_moe.pdf)",
    )
    args = parser.parse_args(argv)
    if args.window <= 0:
        parser.error("--window must be positive")
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    specification = load_manifest(args.data_dir)["lol_vs_moe"]
    groups = []
    for group in specification["groups"]:
        curves = [load_run(run, args.data_dir) for run in group["runs"]]
        groups.append(aggregate_group(group, curves, args.window))

    router_specification = specification["router"]
    router_heatmap = loss_router_plot.load_router_heatmap(router_specification, args.data_dir)
    router_source = json.dumps(provenance(args.data_dir, [router_specification["file"]])[0], sort_keys=True)
    figure = make_figure(groups, router_heatmap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = ", ".join(
        str(source) for group in groups for source in group["sources"]
    )
    try:
        with plt.rc_context(PAPER_RC):
            figure.savefig(
                args.output, format="pdf", bbox_inches="tight", transparent=True,
                metadata={
                    "Title": "Seed-averaged active-match training loss and expert utilization",
                    "Subject": (
                        "CE loss against cumulative training FLOPs; pointwise mean "
                        f"and min/max across three seeds after two passes of a trailing {args.window}-"
                        "step average; legend reports terminal-checkpoint validation "
                        "NLL avg/min/max across seeds, evaluated on 100 batches and "
                        "2,662,400 tokens per checkpoint; complete training histories: "
                        f"{sources}; learned router shares exclude mandatory routes; "
                        f"router source {router_source}"
                    ),
                },
            )
    finally:
        plt.close(figure)

    summaries = "; ".join(
        f"{group['label']}: final val avg/min/max "
        f"{group['final_val_mean']:.4f}/{group['final_val_minimum']:.4f}/"
        f"{group['final_val_maximum']:.4f}"
        for group in groups
    )
    print(
        f"Wrote {args.output.name}; trailing average {args.window} steps, two passes; {summaries}; "
        f"learned-router utilization from {router_source}"
    )


if __name__ == "__main__":
    main()
