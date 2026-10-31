"""Plot the three LoL learned-routing snapshots from the MoE comparison.

Run: python -m plotting.lol_vs_moe_routing_seeds
"""
from __future__ import annotations

import argparse
from pathlib import Path


from . import loss_router_plot
from .local_data import ASSETS, load_manifest, provenance

plt = loss_router_plot.plt
SEEDS = (42, 2002, 7402)

def load_heatmaps(data_dir=ASSETS):
    heatmaps = {}
    for specification in load_manifest(data_dir)["lol_vs_moe_routing_seeds"]:
        heatmap = loss_router_plot.load_router_heatmap(specification, data_dir)
        if (heatmap["experts"] != tuple(f"E{i}" for i in range(24))
                or heatmap["layers"] != tuple(f"L{i}" for i in range(16))):
            raise ValueError("unexpected expert or core-layer labels")
        seed = specification["seed"]
        if seed in heatmaps:
            raise ValueError("duplicate seed diagnostic")
        heatmaps[seed] = heatmap
    if set(heatmaps) != set(SEEDS):
        raise ValueError("expected one diagnostic for each seed")
    return heatmaps


def make_figure(heatmaps):
    with plt.rc_context(loss_router_plot.PAPER_RC):
        figure, axes = plt.subplots(3, 1, figsize=(7, 13.5), layout="constrained")
        figure.patch.set_alpha(0.0)
        for row, seed in enumerate(SEEDS):
            ax = axes[row]
            ax.set_facecolor("none")
            mesh = loss_router_plot.draw_router_heatmap(ax, heatmaps[seed], vmax=1.0)
            ax.set_aspect("auto")
            ax.set_title(f"Seed {seed}")
            ax.set_ylabel("Core Layer")
            ax.tick_params(axis="x", labelsize=8, labelrotation=90)
            ax.tick_params(axis="y", labelsize=9)
            for annotation in ax.texts:
                annotation.set_fontsize(7)
        colorbar = figure.colorbar(mesh, ax=axes, fraction=0.02, pad=0.02)
        colorbar.set_label("Window share")
        colorbar.ax.yaxis.set_major_formatter(loss_router_plot.PercentFormatter(1.0))
        colorbar.solids.set_rasterized(False)
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ASSETS)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".pdf"))
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    heatmaps = load_heatmaps(args.data_dir)
    sources = str(provenance(args.data_dir, [s["file"] for s in load_manifest(args.data_dir)["lol_vs_moe_routing_seeds"]]))
    figure = make_figure(heatmaps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with plt.rc_context(loss_router_plot.PAPER_RC):
            figure.savefig(
                args.output, format="pdf", bbox_inches="tight", transparent=True,
                metadata={
                    "Title": "LoL routing by seed for the MoE comparison",
                    "Subject": "Last logged learned-router window shares, excluding mandatory routes; " + sources,
                },
            )
    finally:
        plt.close(figure)
    print(f"Wrote {args.output.name}\n{sources}")


if __name__ == "__main__":
    main()
