"""Shared paper renderer for training CE and pinned expert utilization.

Curves contain cumulative FLOPs (x), prepared loss (y), label and color;
optional raw_y adds a faint trace, and minimum/maximum add a seed range.
"""
from __future__ import annotations

import json
from pathlib import Path
import hashlib

from plotting.local_data import ASSETS, source_path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import LogLocator, NullLocator, PercentFormatter, StrMethodFormatter  # noqa: E402
import numpy as np  # noqa: E402

if __package__:
    from . import loss_history
else:
    import loss_history

FONT_PATH = Path(__file__).with_name("assets") / "Times New Roman.ttf"
font_manager.fontManager.addfont(str(FONT_PATH))
FONT_FAMILY = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
PAPER_RC = {
    **loss_history.PAPER_RC,
    "font.family": FONT_FAMILY,
    "font.size": 11.5,
    "axes.labelsize": 12.0,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "mathtext.fontset": "custom",
    **{f"mathtext.{style}": FONT_FAMILY for style in ("rm", "it", "bf", "bfit", "cal", "sf", "tt")},
    "mathtext.fallback": None,
}


def read_router_heatmap(path: Path, specification):
    """Read and validate one pinned learned-router Plotly heatmap."""

    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError(f"{path} is not a Plotly figure")
    heatmaps = [
        trace
        for trace in data["data"]
        if isinstance(trace, dict) and trace.get("type") == "heatmap"
    ]
    if len(heatmaps) != 1:
        raise ValueError(f"{path} must contain exactly one heatmap trace")

    trace = heatmaps[0]
    try:
        experts = tuple(str(label) for label in trace["x"])
        layers = tuple(str(label) for label in trace["y"])
        values = np.asarray(trace["z"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} contains invalid heatmap data") from error
    if (
        values.ndim != 2
        or len(layers) != values.shape[0]
        or len(experts) != values.shape[1]
    ):
        raise ValueError(f"{path} heatmap labels do not match its matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite router shares")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError(f"{path} contains router shares outside [0, 1]")
    if not np.allclose(
        values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5,
    ):
        raise ValueError(f"{path} router shares are not normalized by layer")
    return {
        "label": specification["label"],
        "step": specification["router_step"],
        "experts": experts,
        "layers": layers,
        "values": values,
    }


def load_router_heatmap(specification, data_dir=ASSETS):
    """Read a local router diagnostic, verifying a checksum when supplied."""
    path = source_path(data_dir, specification["file"])
    expected = specification.get("sha256")
    if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{specification['file']} has an unexpected SHA-256")
    return read_router_heatmap(path, specification)


def draw_router_heatmap(router_axes, router_heatmap, *, vmax=None):
    """Draw annotated routing cells and axes; return the colorbar mappable."""

    dense_heatmap = max(router_heatmap["values"].shape) > 8
    values = router_heatmap["values"]
    mesh = router_axes.pcolormesh(
        np.arange(values.shape[1] + 1) - 0.5,
        np.arange(values.shape[0] + 1) - 0.5,
        values,
        cmap="turbo",
        vmin=0.0,
        vmax=float(values.max()) if vmax is None else vmax,
        edgecolors="white",
        linewidth=0.45,
    )
    router_axes.set_xlim(-0.5, values.shape[1] - 0.5)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            red, green, blue, _ = mesh.cmap(mesh.norm(value))
            luminance = (
                0.2126 * red + 0.7152 * green + 0.0722 * blue
            )
            label = f"{100.0 * value:.0f}"
            router_axes.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="black" if luminance > 0.52 else "white",
                fontsize=5.0 if dense_heatmap else 7.0,
            )
    router_axes.set_ylim(values.shape[0] - 0.5, -0.5)
    router_axes.set_aspect("equal")
    router_axes.set_xticks(range(values.shape[1]))
    router_axes.set_xticklabels(router_heatmap["experts"])
    router_axes.set_yticks(range(values.shape[0]))
    router_axes.set_yticklabels(router_heatmap["layers"])
    if dense_heatmap:
        router_axes.tick_params(axis="x", labelsize=6.0, labelrotation=90)
        router_axes.tick_params(axis="y", labelsize=7.0)
    router_axes.set_xlabel("Expert")
    router_axes.set_ylabel("Core Layer")
    router_axes.set_title(router_heatmap["label"])
    for spine in router_axes.spines.values():
        spine.set_visible(False)
    return mesh


def make_figure(
    curves, router_heatmap=None, *, loss_ylim=(0.58, 12.0),
    figsize=None, width_ratios=(1.55, 1.0), wspace=None, legend_ncol=1, legend_fontsize=None,
):
    """Build a vector-native loss figure with an optional router panel."""

    dense_heatmap = router_heatmap is not None and max(router_heatmap["values"].shape) > 8
    with plt.rc_context(PAPER_RC):
        if router_heatmap is None:
            figure, loss_axes = plt.subplots(figsize=figsize or (7.2, 3.8))
        else:
            figure, (loss_axes, router_axes) = plt.subplots(
                1, 2, figsize=figsize or ((9.6, 4.6) if dense_heatmap else (7.2, 3.8)),
                gridspec_kw={"width_ratios": width_ratios},
            )
        figure.patch.set_alpha(0.0)
        for axes in figure.axes:
            axes.set_facecolor("none")
        for curve in curves:
            x = curve["x"] / 1e18
            linestyle = curve.get("linestyle", "-")
            if "raw_y" in curve:
                loss_axes.plot(
                    x, curve["raw_y"], color=curve["color"], linestyle=linestyle,
                    linewidth=0.35, alpha=0.10,
                )
            if "minimum" in curve:
                loss_axes.fill_between(
                    x, curve["minimum"], curve["maximum"],
                    color=curve["color"], alpha=0.18, linewidth=0.0,
                )
            loss_axes.plot(
                x, curve["y"], color=curve["color"], linestyle=linestyle,
                linewidth=1.45, label=curve["label"],
            )

        loss_axes.set_xlabel("ExaFLOPs")
        loss_axes.set_ylabel("CE Train Loss")
        loss_axes.set_yscale("log")
        if loss_ylim is None:
            loss_axes.yaxis.set_major_locator(LogLocator(subs=range(1, 10)))
            loss_axes.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            loss_axes.yaxis.set_minor_locator(NullLocator())
        else:
            loss_axes.set_ylim(*loss_ylim)
            loss_axes.set_yticks((0.6, 0.8, 1.0, 2.0, 4.0, 8.0))
            loss_axes.set_yticklabels(("0.6", "0.8", "1", "2", "4", "8"))
        loss_axes.set_xlim(left=0.0)
        loss_axes.grid(
            True, color="#a7a7a7", linewidth=0.4, alpha=0.35,
        )
        loss_axes.set_axisbelow(True)
        loss_axes.spines["left"].set_linewidth(0.65)
        loss_axes.spines["bottom"].set_linewidth(0.65)

        if router_heatmap is not None:
            mesh = draw_router_heatmap(router_axes, router_heatmap)
            colorbar = figure.colorbar(
                mesh, ax=router_axes, fraction=0.045, pad=0.025,
            )
            colorbar.set_label("Window share")
            colorbar.solids.set_rasterized(False)
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        figure.subplots_adjust(
            left=0.09, right=0.97, top=0.88, bottom=0.39,
            wspace=(0.30 if dense_heatmap else 0.08) if wspace is None else wspace,
        )
        handles, labels = loss_axes.get_legend_handles_labels()
        loss_position = loss_axes.get_position()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5 if legend_ncol > 1 else (loss_position.x0 + loss_position.x1) / 2,
                0.27,
            ),
            borderaxespad=0.0,
            frameon=False,
            ncol=legend_ncol,
            fontsize=legend_fontsize,
        )
    return figure


