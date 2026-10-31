#!/usr/bin/env python3
"""Render multilingual, Sudoku, and maze routing results as one paper figure.

The multilingual panel reads the local checkpoint PCA by default and circles
the Latin-script group with one ellipse.
The confusion panels use the committed forced-route CSV.

Example:
    python plotting/routing_overview.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.legend_handler import HandlerPatch  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting.multilingual import (  # noqa: E402
    PAPER_RC,
    load_checkpoint_pca,
    pdf_path,
    plot_multilingual,
)
from plotting.local_data import ASSETS, load_manifest, source_path

FONT_PATH = Path(__file__).with_name("assets") / "Times New Roman.ttf"
font_manager.fontManager.addfont(str(FONT_PATH))
FONT_FAMILY = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
PAPER_RC = {**PAPER_RC, "font.family": FONT_FAMILY}

LANGUAGE_NAMES = {
    "arb": "Arabic",
    "cmn": "Mandarin",
    "deu": "German",
    "fra": "French",
    "hin": "Hindi",
    "jpn": "Japanese",
    "kor": "Korean",
    "rus": "Russian",
    "spa": "Spanish",
    "tur": "Turkish",
}

MULTILINGUAL_PANEL = "router/multilingual/cluster_space_w1"
MULTILINGUAL_ORDINAL = 1
STRATEGIES = ("S1", "S2", "S3", "S4", "S5")
GENERATED = (*STRATEGIES, "mixed/invalid")
DATASETS = ("sudoku", "maze")
FIELDS = (
    "dataset", "source_run", "checkpoint", "forced_strategy",
    "generated_strategy", "count", "forced_total", "fraction",
)


def load_confusion_data(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]:
    """Load and validate the complete forced-route confusion table."""

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(
                f"{path} must have exactly these columns: {', '.join(FIELDS)}"
            )
        raw_rows = list(reader)

    expected = {
        (dataset, forced, generated)
        for dataset in DATASETS
        for forced in STRATEGIES
        for generated in GENERATED
    }
    seen = set()
    matrices = {
        dataset: np.zeros((len(STRATEGIES), len(GENERATED)), dtype=float)
        for dataset in DATASETS
    }
    counts = {
        dataset: np.zeros((len(STRATEGIES), len(GENERATED)), dtype=np.int64)
        for dataset in DATASETS
    }
    totals: dict[tuple[str, str], int] = {}
    sources = {dataset: set() for dataset in DATASETS}
    checkpoints = {dataset: set() for dataset in DATASETS}

    for row in raw_rows:
        key = (
            row["dataset"], row["forced_strategy"], row["generated_strategy"],
        )
        if key not in expected:
            raise ValueError(f"unexpected confusion row {key}")
        if key in seen:
            raise ValueError(f"duplicate confusion row {key}")
        seen.add(key)
        try:
            count = int(row["count"])
            forced_total = int(row["forced_total"])
            fraction = float(row["fraction"])
        except ValueError as error:
            raise ValueError(f"non-numeric confusion values in row {key}") from error
        if count < 0 or forced_total <= 0 or count > forced_total:
            raise ValueError(f"invalid count or forced_total in row {key}")
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"invalid fraction in row {key}")
        if not math.isclose(
            fraction, count / forced_total, rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(f"fraction does not match count/total in row {key}")

        dataset, forced, generated = key
        index = (STRATEGIES.index(forced), GENERATED.index(generated))
        matrices[dataset][index] = fraction
        counts[dataset][index] = count
        total_key = (dataset, forced)
        if total_key in totals and totals[total_key] != forced_total:
            raise ValueError(f"inconsistent forced_total for {total_key}")
        totals[total_key] = forced_total
        if not row["source_run"] or not row["checkpoint"]:
            raise ValueError(f"missing provenance in row {key}")
        sources[dataset].add(row["source_run"])
        checkpoints[dataset].add(row["checkpoint"])

    missing = expected - seen
    if missing:
        raise ValueError(f"missing {len(missing)} confusion rows")
    for dataset in DATASETS:
        if len(sources[dataset]) != 1 or len(checkpoints[dataset]) != 1:
            raise ValueError(f"{dataset} must have one source run and checkpoint")
        if not np.allclose(matrices[dataset].sum(axis=1), 1.0):
            raise ValueError(f"{dataset} fractions must sum to one by forced strategy")
        for row, forced in enumerate(STRATEGIES):
            if counts[dataset][row].sum() != totals[(dataset, forced)]:
                raise ValueError(
                    f"{dataset} counts do not match forced_total for {forced}"
                )

    provenance = {
        dataset: (next(iter(sources[dataset])),
                  next(iter(checkpoints[dataset])))
        for dataset in DATASETS
    }
    return matrices, provenance


def format_fraction(value: float, *, compact: bool = False) -> str:
    """Format every nonzero fraction without hiding small failure rates."""

    if value == 0.0:
        return ""
    percentage = 100.0 * value
    if percentage == 100.0:
        return "100%"
    if percentage >= 99.9:
        label = f"{percentage:.3f}"
    elif percentage >= 1.0:
        label = f"{percentage:.1f}"
    else:
        label = f"{percentage:.3f}"
    return f"{label.rstrip('0').rstrip('.') if compact else label}%"


def plot_confusion(
    axes, matrix: np.ndarray, title: str, *, compact: bool = False,
):
    """Render one vector-native forced-route confusion matrix."""

    mesh = axes.pcolormesh(
        np.arange(len(GENERATED) + 1) - 0.5,
        np.arange(len(STRATEGIES) + 1) - 0.5,
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        edgecolors="white",
        linewidth=0.6,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            label = format_fraction(float(matrix[row, column]), compact=compact)
            if label:
                axes.text(
                    column, row, label, ha="center", va="center",
                    fontsize=6.2,
                    color="white" if matrix[row, column] > 0.55 else "black",
                )
    axes.set_xlim(-0.5, len(GENERATED) - 0.5)
    axes.set_ylim(len(STRATEGIES) - 0.5, -0.5)
    axes.set_aspect("equal")
    axes.set_xticks(range(len(GENERATED)))
    axes.set_xticklabels((*STRATEGIES, "mixed/\ninvalid"))
    axes.set_yticks(range(len(STRATEGIES)))
    axes.set_yticklabels(STRATEGIES)
    axes.set_xlabel("Generated strategy")
    axes.set_ylabel("Forced strategy")
    axes.set_title(title, pad=8)
    axes.set_facecolor("none")
    for spine in axes.spines.values():
        spine.set_visible(False)
    return mesh


def make_figure(
    routing_data: dict[str, object],
    confusion: dict[str, np.ndarray],
):
    """Build the transparent three-panel paper figure."""

    if "coordinates" not in routing_data:
        selected = np.asarray(routing_data["ordinals"]) == MULTILINGUAL_ORDINAL
        if int(selected.sum()) < 2:
            raise ValueError(
                f"{MULTILINGUAL_PANEL} needs at least two window-"
                f"{MULTILINGUAL_ORDINAL} routes"
            )
        routing_data = {
            **routing_data,
            **{
                name: np.asarray(routing_data[name])[selected]
                for name in ("routes", "lang_ids", "n_tokens", "ordinals")
            },
        }
    with plt.rc_context(PAPER_RC):
        figure = plt.figure(figsize=(36.5, 9.5))
        figure.patch.set_alpha(0.0)
        grid = figure.add_gridspec(
            1, 4, width_ratios=(1.0, 1.0, 1.0, 0.055), wspace=0.28,
        )
        axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
        colorbar_axes = figure.add_subplot(grid[0, 3])

        plot_multilingual(
            axes[0], routing_data, legend=False, marker_area_scale=2.5,
            circled_languages=tuple(
                language for language in routing_data["languages"]
                if language.endswith("_Latn")
            ),
        )
        multilingual_width_ratio = (
            float(np.ptp(axes[0].get_xlim()))
            / float(np.ptp(axes[0].get_ylim()))
            * len(STRATEGIES)
            / len(GENERATED)
        )
        grid.set_width_ratios((multilingual_width_ratio, 1.0, 1.0, 0.055))
        handles, labels = axes[0].get_legend_handles_labels()
        labels = [
            LANGUAGE_NAMES.get(language.split("_", 1)[0], language.split("_", 1)[0])
            for language in labels
        ]
        handles.append(Ellipse(
            (0, 0), width=2, height=1, fill=False, edgecolor="black",
            linewidth=3.0, linestyle=(0, (3, 2)),
        ))
        labels.append("Latin")
        # Size legend handle boxes separately from the 52 pt text set below.
        figure.legend(
            handles, labels, loc="upper center",
            ncol=2, frameon=False, borderaxespad=0.0, labelspacing=0.20,
            columnspacing=1.0,
            fontsize=28, handlelength=0.9, handleheight=0.7,
            handletextpad=0.45, markerscale=5.0, scatteryoffsets=(0.5,),
            handler_map={Ellipse: HandlerPatch(
                patch_func=lambda **kwargs: Ellipse(
                    (-kwargs["xdescent"] + kwargs["width"] / 2,
                     -kwargs["ydescent"] + kwargs["height"] / 2),
                    width=1.3 * kwargs["width"],
                    height=1.3 * kwargs["height"],
                ),
            )},
        )
        axes[0].set_title("Multilingual routing", pad=8)
        sudoku_mesh = plot_confusion(
            axes[1], confusion["sudoku"], "Sudoku", compact=True,
        )
        plot_confusion(axes[2], confusion["maze"], "Mazes")
        colorbar = figure.colorbar(sudoku_mesh, cax=colorbar_axes)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label("Fraction of generated traces")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        colorbar.outline.set_linewidth(0.65)
        figure.subplots_adjust(
            left=0.14, right=0.93, bottom=0.23, top=0.90,
        )
        # Include the nested multilingual labels and heatmap annotations.
        for text in figure.findobj(Text):
            text.set_fontfamily(FONT_FAMILY)
            text.set_fontsize(text.get_fontsize() + 20.0)
        # Keep three-decimal percentages inside their heatmap cells.
        for axes_item in axes[1:3]:
            for annotation in axes_item.texts:
                annotation.set_fontsize(24)
        for axes_item in figure.axes:
            axes_item.title.set_fontsize(52)
            axes_item.xaxis.label.set_fontsize(50)
            axes_item.yaxis.label.set_fontsize(50)
            for tick in (*axes_item.get_xticklabels(), *axes_item.get_yticklabels()):
                tick.set_fontsize(46)
        colorbar.ax.tick_params(labelsize=56)
        colorbar.ax.yaxis.label.set_fontsize(60)
        for axes_item in axes[1:3]:
            last_tick = axes_item.get_xticklabels()[-1]
            last_tick.set_transform(
                last_tick.get_transform() + ScaledTranslation(
                    0.12, 0, figure.dpi_scale_trans,
                )
            )
        for text in figure.legends[0].get_texts():
            text.set_fontsize(52)
        figure.canvas.draw()
        for axes_item, text in zip(axes[1:3], (
            "S1: Naked singles\nS2: Hidden singles\nS3: Crosshatching\n"
            "S4: Propagation\nS5: Backtracking",
            "S1: BFS\nS2: DFS\nS3: A*\nS4: Bidirectional search\n"
            "S5: Dead-end filling",
        )):
            label_bottom = axes_item.transAxes.inverted().transform((
                0, axes_item.xaxis.label.get_window_extent(
                    figure.canvas.get_renderer()
                ).y0,
            ))[1]
            axes_item.text(
                0.0, label_bottom, text, ha="left", va="top",
                fontsize=48, fontfamily=FONT_FAMILY, linespacing=1.2,
                transform=axes_item.transAxes + ScaledTranslation(
                    0, -0.30, figure.dpi_scale_trans,
                ),
            )
        xlabel_bottom = axes[0].transAxes.inverted().transform((
            0, axes[0].xaxis.label.get_window_extent(
                figure.canvas.get_renderer()
            ).y0,
        ))[1]
        figure.legends[0].set_bbox_to_anchor(
            (0.5, xlabel_bottom),
            transform=axes[0].transAxes + ScaledTranslation(
                0, -0.30, figure.dpi_scale_trans,
            ),
        )
    return figure


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ASSETS)
    parser.add_argument(
        "--multilingual-npz", type=Path,
        help="Use this NPZ; by default use the local checkpoint NPZ if present",
    )
    parser.add_argument(
        "--confusion-csv", type=Path,
        help="forced-route confusion CSV",
    )
    parser.add_argument(
        "--output", type=pdf_path,
        default=Path(__file__).with_suffix(".pdf"),
        help="output PDF (default: plotting/routing_overview.pdf)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    specification = load_manifest(args.data_dir)["routing_overview"]
    local_npz = args.multilingual_npz or source_path(args.data_dir, specification["pca"])
    confusion_csv = args.confusion_csv or source_path(args.data_dir, specification["confusion"])
    confusion, _ = load_confusion_data(confusion_csv)
    routing_data, routing_source = load_checkpoint_pca(local_npz)
    routing_step = int(routing_data["step"])
    figure = make_figure(routing_data, confusion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    multilingual_panel = (
        "checkpoint PCA, windows after first"
        if "coordinates" in routing_data else MULTILINGUAL_PANEL
    )
    subject = (
        f"{multilingual_panel} from {routing_source}, global step {routing_step}; "
        "Sudoku and maze forced-route confusion; "
        f"PCA SHA-256={hashlib.sha256(local_npz.read_bytes()).hexdigest()}; "
        f"confusion SHA-256={hashlib.sha256(confusion_csv.read_bytes()).hexdigest()}"
    )
    figure.savefig(
        args.output,
        format="pdf",
        bbox_inches="tight",
        transparent=True,
        metadata={
            "Title": "Routing across multilingual text, Sudoku, and mazes",
            "Subject": subject,
        },
    )
    plt.close(figure)
    print(
        f"Wrote {args.output.name} from {routing_source} "
        f"(global step {routing_step}, {multilingual_panel}) "
        "and local confusion measurements"
    )


if __name__ == "__main__":
    main()
