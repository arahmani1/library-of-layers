"""Plot pinned LoL learned-routing shares by library size and seed.

Run: python -m plotting.lol_routing_seeds
"""
from __future__ import annotations

import argparse
from pathlib import Path


from . import loss_router_plot
from .local_data import ASSETS, load_manifest, provenance

plt = loss_router_plot.plt
LIBRARY_SIZES = (2, 4, 8)
SEEDS = (42, 2002, 7402)

def load_heatmaps(data_dir=ASSETS):
    heatmaps = {}
    for specification in load_manifest(data_dir)["lol_routing_seeds"]:
        key = (specification["library_size"], specification["seed"])
        heatmap = loss_router_plot.load_router_heatmap(specification, data_dir)
        if (heatmap["experts"] != tuple(f"E{i}" for i in range(key[0]))
                or heatmap["layers"] != tuple(f"L{i}" for i in range(16))):
            raise ValueError("unexpected expert or core-layer labels")
        if key in heatmaps:
            raise ValueError("duplicate library-size/seed diagnostic")
        heatmaps[key] = heatmap
    if set(heatmaps) != {(size, seed) for size in LIBRARY_SIZES for seed in SEEDS}:
        raise ValueError("expected one diagnostic for each library size and seed")
    return heatmaps


def make_figure(heatmaps):
    with plt.rc_context(loss_router_plot.PAPER_RC):
        figure, axes = plt.subplots(3, 3, figsize=(10.5, 12), layout="constrained")
        figure.patch.set_alpha(0.0)
        for row, size in enumerate(LIBRARY_SIZES):
            for column, seed in enumerate(SEEDS):
                ax = axes[row, column]
                ax.set_facecolor("none")
                mesh = loss_router_plot.draw_router_heatmap(
                    ax, heatmaps[size, seed], vmax=1.0,
                )
                ax.set_aspect("auto")
                ax.set_title(f"Seed {seed}" if row == 0 else "")
                ax.set_ylabel(f"L = {size}\nCore Layer" if column == 0 else "")
                ax.set_xlabel("Expert" if row == 2 else "")
                ax.tick_params(axis="x", labelsize=10, labelrotation=0)
                ax.tick_params(axis="y", labelsize=9, labelleft=column == 0)
                for annotation in ax.texts:
                    annotation.set_fontsize(8)
        colorbar = figure.colorbar(mesh, ax=axes, fraction=0.035, pad=0.025)
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
    sources = str(provenance(args.data_dir, [s["file"] for s in load_manifest(args.data_dir)["lol_routing_seeds"]]))
    figure = make_figure(heatmaps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with plt.rc_context(loss_router_plot.PAPER_RC):
            figure.savefig(
                args.output, format="pdf", bbox_inches="tight", transparent=True,
                metadata={
                    "Title": "LoL routing by library size and seed",
                    "Subject": "Last logged learned-router window shares, excluding mandatory routes; " + sources,
                },
            )
    finally:
        plt.close(figure)
    print(f"Wrote {args.output.name}\n{sources}")


if __name__ == "__main__":
    main()
