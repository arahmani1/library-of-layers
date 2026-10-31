"""Vector appendix figures from saved strategy evaluation outputs only.

Run ``python -m plotting.strategy_routes --task sudoku|maze --data-dir DIR``
to render user-supplied aggregate measurements. Pass an optional
``RUN_DIR`` to score saved generations, or add ``--export-data FILE`` to export
their sanitized aggregates. Soft-policy agreement measures dispatch share for the
expert selected by mean probability; paths are always shown as saved.
Use ``--conditions forced`` for reproductions that measured forced routes only;
the default still requires both described and forced observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from plotting.local_data import ASSETS, load_manifest, source_path

from evaluation.plot_strategy_routes import (
    CONDITIONS, DATASETS, NAMES, bar_panel, dataset_of, load, per_strategy,
)

FONT_PATH = Path(__file__).with_name("assets") / "Times New Roman.ttf"
font_manager.fontManager.addfont(str(FONT_PATH))
PAPER_RC = {
    "font.family": font_manager.FontProperties(fname=str(FONT_PATH)).get_name(),
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 7, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
}
# Published route-panel /MediaBox heights (points).
# These publication page sizes do not imply equality of the missing renderer.
ROUTE_HEIGHT_POINTS = {"sudoku": 352.9375, "maze": 251.0625}


def validate_paths(paths):
    """Validate saved mode paths without recomputing routing."""
    if set(paths) != set(NAMES):
        raise ValueError("route table must contain S1 through S5")
    depth = None
    for strategy in NAMES:
        path = np.asarray(paths[strategy]["path"])
        agreement = np.asarray(paths[strategy]["agreement"], dtype=float)
        if path.ndim != 2 or not all(path.shape) or path.shape != agreement.shape:
            raise ValueError(f"{strategy}: path and agreement need matching nonempty [windows, depth] tables")
        if not np.issubdtype(path.dtype, np.integer) or np.any(path < 0):
            raise ValueError(f"{strategy}: expert IDs must be nonnegative integers")
        if not np.isfinite(agreement).all() or np.any((agreement < 0) | (agreement > 1)):
            raise ValueError(f"{strategy}: agreement must be finite and between 0 and 1")
        if depth is not None and path.shape[1] != depth:
            raise ValueError("all strategies must have the same recurrent depth")
        depth = path.shape[1]


def read_inputs(run: Path, task: str, *, conditions=("described", "forced")):
    """Select the same root/shard outputs as the evaluation plotter."""
    if "forced" not in conditions or set(conditions) - dict(CONDITIONS).keys():
        raise ValueError("paper strategy figures require forced with optional described")
    root = run / "generations.jsonl"
    generations = [root] if root.exists() else sorted(run.glob("shard_*/generations.jsonl"))
    if not generations:
        raise FileNotFoundError(f"no generations.jsonl or shard outputs in {run}")
    tables = list(dict.fromkeys(
        path.with_name("paths.json") if path.with_name("paths.json").exists()
        else run / "paths.json" for path in generations
    ))
    documents = [json.loads(path.read_text()) for path in tables]
    saved = documents[0]
    for path, document in zip(tables[1:], documents[1:]):
        if any(document.get(key) != saved.get(key) for key in ("paths", "checkpoint", "step")):
            raise ValueError(f"shard route tables differ: {tables[0]} and {path}")
    paths = saved["paths"]
    validate_paths(paths)
    saved_conditions = load(run)
    by = {condition: saved_conditions[condition] for condition in conditions
          if saved_conditions.get(condition)}
    rows = [record for records in by.values() for record in records]
    if not rows:
        raise ValueError("no generation records to plot")
    if dataset_of(by) != task or any(("maze_id" in record) != (task == "maze") for record in rows):
        raise ValueError(f"generation records do not all belong to {task}")
    metrics = {}
    for condition in conditions:
        if not by.get(condition):
            raise ValueError(f"paper strategy figures require the {condition} condition")
        metrics[condition] = per_strategy(by, condition)
        if any(metrics[condition][strategy]["strategy"] is None for strategy in NAMES):
            raise ValueError(f"{condition} condition needs observations for all five requested strategies")
    table = metrics["forced"]
    sources = [{"input": index, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for index, path in enumerate(dict.fromkeys(generations + tables))]
    return paths, table, {
        "task": task, "sources": sources,
        "route_source": {key: int(saved[key]) for key in ("step", "batches") if key in saved},
        "conditions": {condition: len(records) for condition, records in by.items()},
        "strategy_counts": {condition: {strategy: sum(record["expected"] == strategy for record in records)
                                        for strategy in NAMES}
                            for condition, records in by.items()},
        "excluded_saved_conditions": {condition: len(records) for condition, records in saved_conditions.items()
                                      if condition not in by},
        "measurement_scope": f"active saved {'/'.join(conditions)} rows only; excluded conditions are not active passes",
        "generation_reexecuted": False,
        "described_metrics": metrics.get("described"),
        "forced_metrics": table,
        "agreement_zero": "zero dispatch share for the selected expert; historical vote tables may also have unobserved cells",
        "agreement_labels": "rounded integer percentages; 100 reserved for exactly perfect agreement",
        "publication_geometry": {
            "width_points": 396, "route_height_points": ROUTE_HEIGHT_POINTS[task],
            "conditions_height_points": 230.4,
            "limitation": "page geometry matches; original paper renderer is absent, no pixel equality implied",
        },
        "path_policy": "show saved expert IDs only; no inference or extrapolation",
        "renderer": "reconstructed paper layout; values replayed from supplied saved outputs",
    }


def route_figure(paths, task, *, agreement):
    """Draw every saved window without padding or recomputing selected paths."""
    heights = [max(len(paths[strategy]["path"]) for strategy in NAMES[start:start + 2]) + 3
               for start in (0, 2, 4)]
    fig, axes = plt.subplots(3, 2, figsize=(5.5, ROUTE_HEIGHT_POINTS[task] / 72),
                             gridspec_kw={"height_ratios": heights})
    expert_count = max(16, 1 + max(max(row) for s in NAMES for row in paths[s]["path"]))
    if agreement:
        cmap, norm = plt.get_cmap("turbo").with_extremes(bad="#eeeeee"), colors.Normalize(0, 1)
    else:
        tab = plt.get_cmap("tab20").colors
        palette = list(tab[::2]) + list(tab[1::2])
        cmap = colors.ListedColormap([palette[i % len(palette)] for i in range(expert_count)])
        norm = colors.BoundaryNorm(np.arange(expert_count + 1) - 0.5, expert_count)
    for ax, strategy in zip(axes.flat, NAMES):
        values = np.asarray(paths[strategy]["agreement" if agreement else "path"])
        missing = np.asarray(paths[strategy]["observations"]) == 0\
            if "observations" in paths[strategy] else values == 0
        displayed = np.ma.masked_where(missing, values) if agreement else values
        mesh = ax.pcolormesh(displayed, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.45,
                             rasterized=False)
        windows, depth = values.shape
        for window in range(windows):
            for layer in range(depth):
                value = values[window, layer]
                # The archive reserves 100 for an exactly perfect saved share.
                label = ("—" if missing[window, layer] else f"{min(99 if value < 1 else 100, round(100 * value))}") if agreement else str(value)
                rgb = cmap(norm(value))[:3]
                luminance = np.dot(rgb, [0.299, 0.587, 0.114])
                ax.text(layer + 0.5, window + 0.5, label, ha="center", va="center",
                        fontsize=5.5, color="white" if luminance < 0.45 and label != "—" else "black")
        ax.set(xticks=np.arange(depth) + 0.5, xticklabels=[f"L{i}" for i in range(depth)],
               yticks=np.arange(windows) + 0.5, yticklabels=[f"W{i}" for i in range(windows)],
               xlim=(0, depth), ylim=(windows, 0), aspect="equal")
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(DATASETS[task]["long"][strategy])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[2, 0].set_xlabel("Recurrent layer")
    axes[2, 1].set_axis_off()
    color_ax = axes[2, 1].inset_axes([0, 0.62, 1, 0.10])
    colorbar = fig.colorbar(mesh, cax=color_ax, orientation="horizontal")
    colorbar.solids.set_rasterized(False)
    if agreement:
        colorbar.set_ticks(np.arange(0, 1.01, 0.2))
        colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1))
        colorbar.set_label("Validation agreement\nwith selected expert")
    else:
        colorbar.set_ticks(range(expert_count))
        colorbar.set_label("Selected expert ID")
    fig.text(0.005, 0.5, "Window ordinal", rotation=90, va="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.94, hspace=0.65, wspace=0.16)
    return fig


def condition_figure(table, task):
    """Use the existing evaluation metrics and five-metric bar renderer."""
    # strategy_conditions_{sudoku,maze}.pdf has a 396 x 230.4 point MediaBox.
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bar_panel(ax, table, "", DATASETS[task])
    # The archive prints up to three decimals, needed for near-perfect maze rates.
    values = [table[s][key] for s in table
              for key in ("strategy", "valid", "faithful", "answer", "padding_fraction")]
    for text, value in zip(ax.texts, values):
        text.set_text("N/A" if value is None else f"{100 * value:.3f}".rstrip("0").rstrip(".") + "%")
        text.set_fontsize(6)
    ax.set_ylabel("Rate (%)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle=":", color="grey", linewidth=0.9))
    labels.append("Chance (1 in 5)")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.30)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--data-dir", type=Path, default=ASSETS)
    parser.add_argument("--export-data", type=Path, help="Export sanitized aggregate JSON without rendering")
    parser.add_argument("--task", choices=tuple(DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--conditions", choices=("described,forced", "forced"),
                        default="described,forced",
                        help="Saved conditions to validate; forced-only reproductions opt in explicitly.")
    args = parser.parse_args(argv)
    if args.run_dir is not None:
        paths, table, provenance = read_inputs(args.run_dir, args.task,
                                             conditions=tuple(args.conditions.split(",")))
    else:
        relative = load_manifest(args.data_dir)["strategy_routes"][args.task]
        path = source_path(args.data_dir, relative)
        data = json.loads(path.read_text())
        if data["task"] != args.task:
            raise ValueError("strategy input task does not match --task")
        paths = data["paths"]
        validate_paths(paths)
        required = tuple(args.conditions.split(","))
        conditions = data["conditions"]
        strategy_counts = data["strategy_counts"]
        if not all(condition in conditions and isinstance(conditions[condition], int)
                   and conditions[condition] > 0 for condition in required):
            raise ValueError("strategy input does not contain all requested conditions")
        metrics = {}
        for condition in required:
            coverage = strategy_counts[condition]
            if (set(coverage) != set(NAMES)
                    or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in coverage.values())
                    or sum(coverage.values()) != conditions[condition]):
                raise ValueError("invalid per-strategy coverage")
            raw = data[condition + "_metrics"]
            if set(raw) != set(NAMES):
                raise ValueError("strategy metrics must contain S1 through S5")
            metrics[condition] = {}
            for strategy in NAMES:
                metrics[condition][strategy] = {}
                for key in ("strategy", "valid", "faithful", "answer", "padding_fraction",
                            "cleanup_fraction", "padding_n", "padding_coverage"):
                    value = raw[strategy][key]
                    if value is None and key not in ("strategy", "padding_n"):
                        metrics[condition][strategy][key] = None
                        continue
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0
                            or (key != "padding_n" and value > 1)
                            or (key == "padding_n" and not isinstance(value, int))):
                        raise ValueError(f"invalid {condition}/{strategy}/{key} metric")
                    metrics[condition][strategy][key] = value
        table = metrics["forced"]
        provenance = {
            "task": args.task,
            "sources": [{"input": 0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
            "route_source": {key: int(data[key]) for key in ("step", "batches") if key in data},
            "conditions": {condition: conditions[condition] for condition in required},
            "strategy_counts": {condition: strategy_counts[condition] for condition in required},
            "generation_reexecuted": False,
            "described_metrics": metrics.get("described"), "forced_metrics": table,
        }
    if args.export_data is not None:
        args.export_data.parent.mkdir(parents=True, exist_ok=True)
        export = {"task": args.task, "paths": {
            strategy: {key: paths[strategy][key] for key in ("path", "agreement")}
            for strategy in NAMES},
            **provenance["route_source"],
            **{key: provenance[key] for key in ("conditions", "strategy_counts", "described_metrics", "forced_metrics")}}
        args.export_data.write_text(json.dumps(export, indent=2) + "\n")
        print(f"Exported {args.export_data.name}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_names = [f"{args.task}_route_agreement_all_windows.pdf",
                    f"{args.task}_layer_paths_all_windows.pdf", f"strategy_conditions_{args.task}.pdf"]
    metadata = {"Title": f"{args.task} strategy routing", "Subject": json.dumps(provenance, sort_keys=True),
                "Creator": "plotting.strategy_routes", "CreationDate": None, "ModDate": None}
    with plt.rc_context(PAPER_RC):
        for name, kind in zip(output_names, ("agreement", "path", "conditions")):
            fig = condition_figure(table, args.task) if kind == "conditions" else route_figure(
                paths, args.task, agreement=kind == "agreement")
            try:
                fig.savefig(args.output_dir / name, transparent=True, metadata=metadata)
            finally:
                plt.close(fig)
    provenance["outputs"] = [{"path": name, "sha256": hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()}
                             for name in output_names]
    manifest = args.output_dir / f"{args.task}_strategy_routes_provenance.json"
    manifest.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Saved {', '.join(output_names)} and {manifest.name}")


if __name__ == "__main__":
    main()
