"""LoL versus Huginn at matched compute, with three-seed mean and min/max.

The p4c4 comparison reads pinned terminal-checkpoint validation from
lol_vs_huginn_final_validation.json (the first 21,299,200 validation tokens).
Its RHS smooths each complete training CE history with two passes of a
1,000-step trailing average
before aggregation at identical logged FLOP coordinates. The viewport fits
seed ranges from 5 ExaFLOPs onward.
Complete histories are read from the selected local data directory.

Run: python -m plotting.lol_vs_huginn --output plotting/lol_vs_huginn.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .loss_history import running_average, validate_history
    from .local_data import ASSETS, load_manifest, source_path, provenance
else:
    from loss_history import running_average, validate_history
    from local_data import ASSETS, load_manifest, source_path, provenance

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STEM = Path(__file__).stem
SEEDS = (42, 2002, 7402)

# Configuration key -> (dataset key, legend label, short mark label, family).
RUN_SETS = {
    "p4c4": {
        "hug-c2r8":  ("hug-c2r8", "Huginn, 2 × 8",  "2×8",  "huginn"),
        "hug-c4r4":  ("hug-c4r4", "Huginn, 4 × 4",  "4×4",  "huginn"),
        "hug-c8r2":  ("hug-c8r2", "Huginn, 8 × 2",  "8×2",  "huginn"),
        "lol-e2":    ("lol-e2", "LoL, L = 2",       "L = 2",   "lol"),
        "lol-e4":    ("lol-e4", "LoL, L = 4",       "L = 4",   "lol"),
        "lol-e8":    ("lol-e8", "LoL, L = 8",       "L = 8",   "lol"),
    },
}
HUGINN_HEADER = "Huginn (Layers × Loops)"
LOL_HEADER = "LoL (Library Size, L)"
MARK = {"lol": "o", "huginn": "^"}
GRID, AXC, INK2 = "#e6e5e1", "#b8b7b2", "#52514e"
FS = dict(note=14)  # shared font family with enlarged paper labels below


def fetch_run(specification: dict, *, data_dir=ASSETS,
              metric: str = "train/ce_loss") -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a complete local history and exported parameter counts."""
    if metric != "train/ce_loss":
        raise ValueError(f"Unsupported history metric {metric}")
    frame = validate_history(
        pd.read_csv(source_path(data_dir, specification["file"]), float_precision="round_trip"),
        {"steps": int(specification["step"]) + 1}, specification["file"],
    )
    if any(not isinstance(specification[key], int) or specification[key] <= 0
           for key in ("params", "layers")):
        raise ValueError("parameter and layer counts must be positive integers")
    return frame["global_step"].to_numpy(), frame[metric].to_numpy(), {
        "final_step": specification["step"], "params": specification["params"],
        "layers": specification["layers"], "train_flops": frame["train/total_flops"].to_numpy(),
    }


def load_seed_data(path: Path, window: int, *, data_dir=ASSETS, configurations=None) -> dict:
    """Validate pinned evaluations and aggregate complete, aligned seed histories."""
    configurations = RUN_SETS["p4c4"] if configurations is None else configurations
    entries = json.loads(path.read_text())
    expected = {(key, seed) for key in configurations for seed in SEEDS}
    if (len(entries) != len(expected)
            or {(e["key"], e["seed"]) for e in entries} != expected
            or len({e["file"] for e in entries}) != len(entries)):
        raise RuntimeError("Expected six configurations, each with distinct runs for seeds 42, 2002, 7402")
    data = {}
    for key in configurations:
        group = sorted((e for e in entries if e["key"] == key), key=lambda e: e["seed"])
        curves, sizes = [], []
        for entry in group:
            rid = f"{key}-seed-{entry['seed']}"
            if entry["val_tokens"] != 21_299_200 or not np.isfinite(entry["nll"]):
                raise RuntimeError(f"{rid}: expected finite NLL over 21,299,200 validation tokens")
            protocol = dict(validation_set="val", batch_size=26, block_size=1024,
                            val_batches=800)
            if any(entry.get(field) != value for field, value in protocol.items()):
                raise RuntimeError(f"{rid}: validation protocol does not match the pinned subset")
            steps, ce, meta = fetch_run(entry, data_dir=data_dir)
            if entry["step"] != meta["final_step"]:
                raise RuntimeError(f"{rid}: pinned checkpoint step does not match completed run")
            flops = np.asarray(meta["train_flops"])
            if not np.isfinite(flops).all() or np.any(np.diff(flops) <= 0):
                raise RuntimeError(f"{rid}: invalid cumulative FLOPs")
            if curves and not np.array_equal(flops, reference_flops):
                raise RuntimeError(f"{key}: seed FLOP coordinates differ; cannot aggregate")
            reference_flops = flops
            sizes.append((meta["params"], meta["layers"]))
            curves.append(running_average(running_average(ce, window), window))
            print(f"{key} seed={entry['seed']} step={entry['step']}: NLL={entry['nll']:.9f}, "
                  f"tokens={entry['val_tokens']}; RHS {len(steps)} complete steps")
        if len(set(sizes)) != 1:
            raise RuntimeError(f"{key}: seed model parameter counts differ")
        final = np.asarray([e["nll"] for e in group])
        curves = np.asarray(curves)
        data[key] = dict(final_nll=final.mean(), final_min=final.min(), final_max=final.max(),
                         train_flops=reference_flops, curve_mean=curves.mean(axis=0),
                         curve_min=curves.min(axis=0), curve_max=curves.max(axis=0),
                         params=sizes[0][0], layers=sizes[0][1], entries=group)
    return data


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.patch.set_alpha(0)


def grouped_legend(axes, runs: dict, data: dict, color: dict, headers: dict, Line2D) -> None:
    """One titled legend per model family below the panels."""
    for ax, family in zip(axes, ("lol", "huginn")):
        ax.set_axis_off()
        keys = sorted((k for k in runs if runs[k][3] == family), key=lambda k: data[k]["params"])
        handles = [
            Line2D([], [], color=color[k], linestyle="-",
                   linewidth=1.7, marker=MARK[runs[k][3]], markersize=7, markeredgewidth=0,
                   label=runs[k][2])
            for k in keys
        ]
        ax.legend(handles=handles, title=headers[family], loc="center", ncol=len(handles),
                  handlelength=1.5, handletextpad=0.4, columnspacing=0.9,
                  borderaxespad=0, title_fontsize=16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, default=1000,
                    help="RHS trailing average per pass (two passes; default: 1000; 1 disables)")
    ap.add_argument("--output", type=Path, default=None, help="PDF path override (must end in .pdf)")
    ap.add_argument("--data-dir", type=Path, default=ASSETS,
                    help="local data directory containing datasets.json")
    args = ap.parse_args()
    if args.window <= 0:
        ap.error("--window must be positive")
    out = args.output or Path(__file__).with_name(f"{STEM}.pdf")
    if out.suffix != ".pdf":
        raise SystemExit("--output must end in .pdf")
    specification = load_manifest(args.data_dir)[STEM]
    runs = specification["configurations"]
    headers = {"lol": LOL_HEADER, "huginn": HUGINN_HEADER}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator, StrMethodFormatter
    from plotting.loss_router_plot import PAPER_RC
    from utils.routing_viz import LANGUAGE_COLORS

    plt.rcParams.update({
        **PAPER_RC,
        "font.size": 16, "axes.labelsize": 18,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.edgecolor": AXC, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    })

    data = load_seed_data(source_path(args.data_dir, specification["validation"]),
                          args.window, data_dir=args.data_dir, configurations=runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out.with_suffix(".csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "mean", "min", "max", *[f"seed_{s}" for s in SEEDS]])
        for key, values in data.items():
            by_seed = {e["seed"]: e["nll"] for e in values["entries"]}
            writer.writerow([key, values["final_nll"], values["final_min"], values["final_max"],
                             *[by_seed[seed] for seed in SEEDS]])
    print(f"wrote {summary_path.name}")
    by_fam = lambda fam: sorted((k for k in runs if runs[k][3] == fam), key=lambda k: data[k]["params"])
    color = {}
    fam_color = {"lol": LANGUAGE_COLORS[1], "huginn": LANGUAGE_COLORS[0]}
    for fam in ("lol", "huginn"):
        keys = by_fam(fam)
        base = np.asarray(matplotlib.colors.to_rgb(fam_color[fam]))
        # Light, base, and dark shades in increasing parameter order.
        color.update(zip(keys, (0.65 + 0.35 * base, base, 0.40 * base)))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 4.3), constrained_layout=True,
                             gridspec_kw={"height_ratios": (1, 0.22)})
    (ax_a, ax_b), legend_axes = axes
    fig.patch.set_alpha(0)

    # ---- panel a: final NLL vs. non-embedding parameters ----
    xs_all = [data[k]["params"] / 1e6 for k in runs]
    ys_all = [data[k][bound] for k in runs for bound in ("final_min", "final_max")]
    x_span = max(xs_all) - min(xs_all)
    xlim = (min(xs_all) - 0.12 * x_span, max(xs_all) + 0.10 * x_span)
    ylim = (np.floor((min(ys_all) - 0.018) * 200) / 200, np.ceil((max(ys_all) + 0.018) * 200) / 200)
    for fam in ("huginn", "lol"):
        keys = by_fam(fam)
        xs = [data[k]["params"] / 1e6 for k in keys]
        ys = [data[k]["final_nll"] for k in keys]
        ax_a.plot(xs, ys, color=fam_color[fam], linewidth=2.0, alpha=0.9, zorder=2)
        for k, x, y in zip(keys, xs, ys):
            ax_a.errorbar(x, y, yerr=[[y - data[k]["final_min"]], [data[k]["final_max"] - y]],
                          capsize=3, elinewidth=1.2, marker=MARK[fam], color=color[k], markersize=7.5, markeredgecolor="white",
                      markeredgewidth=1.0, linestyle="none", zorder=4)
            above = fam == "huginn"
            dy, va = (9, "bottom") if above else (-10, "top")
            label_y = data[k]["final_max" if above else "final_min"]
            ax_a.annotate(runs[k][2], (x, label_y), xytext=(0, dy), textcoords="offset points",
                          ha="center", va=va, fontsize=FS["note"], color=INK2, zorder=5)
    ax_a.set(xlim=xlim, ylim=ylim, xlabel="Non-embedding parameters (M)", ylabel="Eval CE Loss")
    style(ax_a)

    # ---- panel b: training CE vs. training FLOPs ----
    late_losses = []
    for k in runs:
        fam = runs[k][3]
        x = data[k]["train_flops"] / 1e18
        tail = x >= 5
        x, y = x[tail], data[k]["curve_mean"][tail]
        low, high = data[k]["curve_min"][tail], data[k]["curve_max"][tail]
        late_losses.extend(low)
        late_losses.extend(high)
        ax_b.fill_between(x, low, high, color=color[k], alpha=0.60, linewidth=0)
        ax_b.plot(x, y, color=color[k], linestyle="-", linewidth=2.0,
                  label=runs[k][1], zorder=5 if fam == "lol" else 3)
    ax_b.set_xlim(left=0)
    ax_b.set_xlabel("ExaFLOPs")
    ax_b.set_xlim(
        5.0,
        max(data[k]["train_flops"][-1] / 1e18 for k in runs),
    )
    ax_b.set(yscale="log", ylabel="CE Train Loss")
    low, high = np.log([min(late_losses), max(late_losses)])
    padding = max((high - low) * 0.08, 0.005)
    ax_b.set_ylim(np.exp(low - padding), np.exp(high + padding))
    ax_b.set_yticks(np.linspace(min(late_losses), max(late_losses), 5))
    ax_b.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax_b.yaxis.set_minor_locator(NullLocator())
    style(ax_b)
    grouped_legend(legend_axes, runs, data, color, headers, Line2D)

    subject = "Three seeds (42, 2002, 7402): arithmetic mean and min/max; "
    subject += "LHS final checkpoint validation: val, batch_size=26, batches=800, length=1024, tokens=21299200; "
    subject += "; ".join(f"{k}:seed{e['seed']}@step{e['step']}:nll={e['nll']}"
                         + (f":sequence_aux_coeff={e['sequence_aux_coeff']}" if "sequence_aux_coeff" in e else "")
                         for k in runs for e in data[k]["entries"])
    subject += "; inputs=" + json.dumps(provenance(args.data_dir, [specification["validation"],
        *[e["file"] for k in runs for e in data[k]["entries"]]]), sort_keys=True)
    subject += f"; RHS train/ce_loss vs train/total_flops; per-seed trailing running average: {args.window} optimizer steps, 2 passes"
    fig.savefig(out, transparent=True, metadata={"Title": f"{STEM} (p4c4, curves, horizontal)",
                                                 "Subject": subject})
    print(f"wrote {out.name} (RHS training CE, trailing average: {args.window} optimizer steps, 2 passes)")



if __name__ == "__main__":
    main()
