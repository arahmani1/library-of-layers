#!/usr/bin/env python3
"""Plot 19 multilingual routing snapshots in a five-row, four-column grid.

Fit PCA independently to each snapshot's recorded routes. Snapshots are read from the selected local data directory. No checkpoint inference is needed. Run from the repository root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.multilingual import (  # noqa: E402
    load_checkpoint_pca, load_routing_data,
    pdf_path, plot_multilingual,
)
from plotting.routing_overview import (  # noqa: E402
    FONT_FAMILY, LANGUAGE_NAMES, PAPER_RC,
)
from utils.routing_viz import (  # noqa: E402
    cluster_space_coordinates, language_color,
)

from plotting.local_data import ASSETS, load_manifest, source_path, provenance


def make_figure(snapshots, final):
    """Fit each snapshot's own PCA; preserve the independently fitted final panel."""
    recovered = cluster_space_coordinates(final["routes"], final["n_blocks"])
    if not np.allclose(recovered, final["coordinates"], rtol=1e-7, atol=1e-7):
        raise ValueError("Recovered final PCA does not match the saved main figure")

    projected = []
    for snapshot in snapshots:
        if (snapshot["n_blocks"] != final["n_blocks"]
                or snapshot["n_depths"] != final["n_depths"]
                or set(snapshot["languages"]) != set(final["languages"])):
            raise ValueError("Snapshot dimensions or languages differ from the final checkpoint")
        selected = snapshot["ordinals"] > 0
        if selected.sum() < 2:
            raise ValueError("Snapshot has fewer than two windows after the first")
        data = {**snapshot, **{
            name: snapshot[name][selected]
            for name in ("routes", "lang_ids", "n_tokens", "ordinals")
        }}
        data["coordinates"] = cluster_space_coordinates(
            data["routes"], data["n_blocks"],
        )
        projected.append(data)
    projected.append(final)
    points = np.concatenate([data["coordinates"] for data in projected])
    lower, upper = points.min(0), points.max(0)
    padding = np.maximum(upper - lower, 1e-6) * 0.06
    lower, upper = lower - padding, upper + padding

    with plt.rc_context({**PAPER_RC, "font.family": FONT_FAMILY}):
        figure, grid = plt.subplots(5, 4, figsize=(8.0, 10.1))
        figure.patch.set_alpha(0)
        for index, data in enumerate(projected):
            axes = grid.flat[index]
            plot_multilingual(axes, data, legend=False, marker_area_scale=0.24)
            axes.set_xlim(lower[0], upper[0])
            axes.set_ylim(lower[1], upper[1])
            axes.set_title(f"Step {data['step']:,}", fontsize=12.5, pad=6)
            axes.set_xlabel("")
            axes.set_ylabel("")
            axes.xaxis.set_major_locator(MultipleLocator(2))
            axes.yaxis.set_major_locator(MultipleLocator(2))
            axes.tick_params(
                labelsize=10.5, length=2.5, pad=2,
                labelleft=index % 4 == 0, labelbottom=index // 4 == 4,
            )
        legend_axes = grid.flat[-1]
        legend_axes.set_axis_off()
        handles = [Line2D(
            [], [], marker="o", linestyle="none", markersize=5,
            markeredgewidth=0, color=language_color(language),
            label=LANGUAGE_NAMES[language.split("_", 1)[0]],
        ) for language in sorted(final["languages"])]
        legend_axes.legend(
            handles=handles, loc="center", ncol=2, frameon=False,
            fontsize=10.5, handlelength=0.6, handletextpad=0.3,
            columnspacing=0.8, labelspacing=0.65, borderaxespad=0,
        )
        figure.supxlabel("PC 1", fontsize=12, y=0.017)
        figure.supylabel("PC 2", fontsize=12, x=0.025)
        figure.subplots_adjust(left=0.085, right=0.99, bottom=0.055,
                               top=0.97, wspace=0.18, hspace=0.28)
        for text in figure.findobj(Text):
            text.set_fontfamily(FONT_FAMILY)
    return figure, projected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ASSETS)
    parser.add_argument("--output", type=pdf_path, default=Path(__file__).with_suffix(".pdf"))
    args = parser.parse_args(argv)
    specification = load_manifest(args.data_dir)["multilingual_progression"]
    final, final_source = load_checkpoint_pca(source_path(args.data_dir, specification["final_pca"]))
    snapshots = [load_routing_data(source_path(args.data_dir, item["file"]), item["step"])
                 for item in specification["snapshots"]]
    if len(snapshots) != 18 or any(a["step"] >= b["step"] for a, b in zip(snapshots, snapshots[1:] + [final])):
        raise ValueError("expected 18 ordered snapshots before the final PCA")
    sources = [str(item) for item in provenance(args.data_dir,
               [item["file"] for item in specification["snapshots"]] + [specification["final_pca"]])]
    figure, projected = make_figure(snapshots, final)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="pdf", transparent=True, bbox_inches="tight",
                   metadata={
                       "Title": "Multilingual routing throughout training",
                       "Subject": (
                           "5 rows x 4 columns, 19 snapshots and legend; "
                           "PCA mean/basis fitted independently at each step; "
                           "deterministic per-snapshot jitter and shared axis limits; "
                           "document-window ordinals > 0. "
                           + "; ".join(sources) + "; " + final_source
                           + "; final PCA uses a separate validation sample."
                       ),
                   })
    plt.close(figure)
    print(f"Wrote {args.output.name}; final PCA recovery verified against saved coordinates")
    print("Displayed windows:", [(data["step"], len(data["routes"])) for data in projected])


if __name__ == "__main__":
    main()
