#!/usr/bin/env python3
"""Dictionary-driven training-loss comparisons; run from the repository root."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from plotting.local_data import ASSETS, load_manifest, provenance
import json

if __package__:
    from . import loss_history
else:
    import loss_history

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.routing_viz import LANGUAGE_COLORS

FONT_PATH = Path(__file__).with_name("assets") / "Times New Roman.ttf"
font_manager.fontManager.addfont(str(FONT_PATH))
FONT_FAMILY = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()

def make_figure(curves, smoothing_window=200):
    """Render the same trailing-smoothed losses in step and compute coordinates."""
    with plt.rc_context({**loss_history.PAPER_RC,
                         "font.family": FONT_FAMILY,
                         "font.size": 12.0,
                         "axes.labelsize": 12.5,
                         "xtick.labelsize": 11.5,
                         "ytick.labelsize": 11.5,
                         "legend.fontsize": 11.5,
                         "axes.prop_cycle": plt.cycler(color=LANGUAGE_COLORS)}):
        figure, axes = plt.subplots(1, 2, sharey=True, figsize=(7.2, 1.85))
        figure.patch.set_alpha(0)
        for curve in curves:
            y = loss_history.running_average(curve["y"], smoothing_window)
            label = curve["label"]
            if curve.get("state", "finished") != "finished":
                label += f"\n({curve['state']}; last step {curve['step'][-1]:,})"
            line, = axes[0].plot(curve["step"], y, label=label, linewidth=1.25)
            axes[1].plot(curve["x"] / 1e19, y, color=line.get_color(), linewidth=1.25)
        axes[0].set_xlabel("Step")
        axes[1].set_xlabel("FLOPs (1e19)")
        axes[0].set_ylabel("CE Loss")
        axes[0].set_yscale("log")
        axes[0].yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 3, 4, 6, 8)))
        axes[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
        axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())
        for axis in axes:
            axis.set_facecolor("none")
            axis.set_xlim(left=0)
            axis.grid(True, linewidth=0.4, alpha=0.3)
            axis.set_axisbelow(True)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.tight_layout(pad=0.5)
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.005), borderaxespad=0,
                      frameon=False, ncol=min(2, len(curves)))
    return figure


def plot_ablation(specification, *, output=None, data_dir=ASSETS):
    """Write a PDF from a named model-to-legend-label dictionary; return its path."""
    name = specification["name"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("plot name must be a non-empty filename stem")
    window = specification.get("smoothing_window", 200)
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("smoothing_window must be a positive integer")
    if not specification["models"]:
        raise ValueError("models must contain at least one model")
    output = Path(output) if output is not None else Path(__file__).with_name(name + ".pdf")
    if output.suffix.lower() != ".pdf":
        raise ValueError("output must end in .pdf")
    models = load_manifest(data_dir)["ablations"]["models"]
    runs = []
    for model, label in specification["models"].items():
        if model not in models:
            raise ValueError(f"unknown model {model!r}; add it to datasets.json")
        run = {**models[model], "label": label}
        runs.append(run)
    curves = [loss_history.load_curve(run, data_dir) for run in runs]
    figure = make_figure(curves, window)
    sources = json.dumps(provenance(data_dir, [run["file"] for run in runs]))
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, transparent=True, bbox_inches="tight", metadata={
            "Title": name,
            "Subject": f"train/ce_loss versus global_step and train/total_flops; trailing average {window}; {sources}",
        })
    finally:
        plt.close(figure)
    print(f"Wrote {output.name}; trailing average {window}; {sources}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation", nargs="?", default="LOAD_BALANCING",
                        help="plot name from the selected datasets.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ASSETS)
    args = parser.parse_args(argv)
    plots = load_manifest(args.data_dir)["ablations"]["plots"]
    if args.ablation not in plots:
        parser.error(f"unknown ablation {args.ablation!r}; choose from {', '.join(plots)}")
    plot_ablation(plots[args.ablation],
                  output=args.output, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
