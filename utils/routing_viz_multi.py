"""Cross-language routing figures over a ``[L, D, K]`` share tensor.

Every function takes ``routing`` normalised within ``(language, depth)`` so a
panel answers "at depth d, where did this language's windows go?" rather than
"how big is this language's eval split?".
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils.routing_viz import (
    PLOT_RC,
    average_linkage,
    language_color,
    modal_block,
    pairwise_language_distance,
    reference_distribution,
    signed_displacement,
)

_DEVIATION_QUANTILE = 0.995


def _panel_grid(count: int) -> tuple[int, int]:
    columns = min(5, max(1, int(np.ceil(np.sqrt(count)))))
    return int(np.ceil(count / columns)), columns


def _ordered(languages, order):
    languages = tuple(languages)
    if order is None:
        return tuple(range(len(languages)))
    order = tuple(int(index) for index in order)
    if sorted(order) != list(range(len(languages))):
        raise ValueError(
            f"language order {order} is not a permutation of "
            f"{len(languages)} languages"
        )
    return order


def _depth_block_axes(axes, depths: int, blocks: int, identity: str) -> None:
    axes.set_xlabel("depth")
    axes.set_ylabel("block")
    axes.plot(
        [-0.5, depths - 0.5],
        [-0.5, depths - 0.5],
        color=identity, alpha=0.5, linewidth=0.9, linestyle=(0, (4, 3)),
    )
    axes.set_xlim(-0.5, depths - 0.5)
    axes.set_ylim(-0.5, blocks - 0.5)


def deviation_vmax(routing, languages, reference="pooled") -> float:
    """Robust symmetric colour limit for :func:`fig_signed_deviation`.

    Computed once at the first eval and reused thereafter, so
    two eval steps' deviation panels are on one scale and a real change in
    specialisation is not hidden by a rescaled colourbar.
    """

    routing = np.asarray(routing, dtype=np.float64)
    deviation = routing - reference_distribution(
        routing, languages, reference
    )[None]
    limit = float(np.quantile(np.abs(deviation), _DEVIATION_QUANTILE))
    return max(limit, 1e-4)


def fig_signed_deviation(
    routing,
    languages,
    n_blocks: int,
    *,
    reference: str = "pooled",
    vmax: float | None = None,
    order=None,
):
    """Per-language ``P[l] - reference`` depth x block panels."""

    routing = np.asarray(routing, dtype=np.float64)
    languages = tuple(languages)
    baseline = reference_distribution(routing, languages, reference)
    deviation = routing - baseline[None]
    limit = (
        float(vmax) if vmax is not None
        else deviation_vmax(routing, languages, reference)
    )
    order = _ordered(languages, order)
    depths = routing.shape[1]
    rows, columns = _panel_grid(len(order))
    unlike = "the consensus" if reference in (None, "", "pooled") else reference

    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(
            rows, columns,
            figsize=(2.5 * columns + 1.4, 2.3 * rows + 0.9),
            squeeze=False,
        )
        image = None
        for position, index in enumerate(order):
            panel = axes[position // columns][position % columns]
            image = panel.imshow(
                deviation[index].T,
                aspect="auto", origin="lower", cmap="RdBu_r",
                vmin=-limit, vmax=limit, interpolation="nearest",
            )
            panel.set_title(
                languages[index], color=language_color(languages[index]),
            )
            # The deviation colormap is white at zero, so the identity
            # reference needs a dark line rather than the map's light one.
            _depth_block_axes(panel, depths, int(n_blocks), "#3d4350")
        for empty in range(len(order), rows * columns):
            axes[empty // columns][empty % columns].axis("off")
        if image is not None:
            figure.colorbar(
                image, ax=axes, shrink=0.75,
                label=f"routing share unlike {unlike}",
            )
        figure.suptitle(
            f"Signed routing deviation from {unlike} "
            f"(symmetric scale +/-{limit:.4f})"
        )
    return figure


def fig_routing_map_multi(routing, languages, n_blocks: int, *, order=None):
    """Per-language depth x block routing maps with the identity diagonal."""

    routing = np.asarray(routing, dtype=np.float64)
    languages = tuple(languages)
    order = _ordered(languages, order)
    depths = routing.shape[1]
    rows, columns = _panel_grid(len(order))
    peak = max(float(routing.max()), 1e-6)

    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(
            rows, columns,
            figsize=(2.5 * columns + 1.4, 2.3 * rows + 0.9),
            squeeze=False,
        )
        image = None
        for position, index in enumerate(order):
            panel = axes[position // columns][position % columns]
            image = panel.imshow(
                routing[index].T,
                aspect="auto", origin="lower", cmap="magma",
                vmin=0.0, vmax=peak, interpolation="nearest",
            )
            panel.set_title(
                languages[index], color=language_color(languages[index]),
            )
            _depth_block_axes(panel, depths, int(n_blocks), "#ffffff")
            # After the identity reference: the two coincide wherever the
            # router has not drifted, and the modal track has to win.
            panel.plot(
                np.arange(depths), modal_block(routing)[index],
                color="#7fe3a0", linewidth=1.0, alpha=0.95,
                marker="o", markersize=2.0, markeredgewidth=0.0,
            )
        for empty in range(len(order), rows * columns):
            axes[empty // columns][empty % columns].axis("off")
        if image is not None:
            figure.colorbar(
                image, ax=axes, shrink=0.75, label="window share at depth",
            )
        figure.suptitle(
            "Routing map by language (green: modal block, dashed: identity)"
        )
    return figure


def fig_utilisation_multi(routing, languages, n_blocks: int, *, order=None):
    """Language x block load, with the pooled load as a marginal bar."""

    routing = np.asarray(routing, dtype=np.float64)
    languages = tuple(languages)
    order = _ordered(languages, order)
    load = routing.mean(1)
    pooled = load.mean(0)

    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(
            2, 1,
            figsize=(max(6.0, 0.22 * int(n_blocks) + 2.4),
                     0.34 * len(order) + 3.2),
            height_ratios=(1, 2.6),
            sharex=True,
        )
        axes[0].bar(
            np.arange(int(n_blocks)), pooled,
            color="#8f9096", width=0.85,
        )
        axes[0].axhline(
            1.0 / max(int(n_blocks), 1),
            color="#e8734c", linewidth=0.8, linestyle="--",
        )
        axes[0].set_ylabel("pooled")
        axes[0].set_title(
            "Block utilisation (mean over depths; dashed: uniform load)"
        )
        image = axes[1].imshow(
            load[list(order)],
            aspect="auto", origin="upper", cmap="viridis",
            vmin=0.0, interpolation="nearest",
        )
        axes[1].set_yticks(np.arange(len(order)))
        axes[1].set_yticklabels([languages[index] for index in order])
        axes[1].set_xlabel("block")
        figure.colorbar(image, ax=axes, shrink=0.7, label="window share")
    return figure


def fig_signed_displacement(routing, languages, *, order=None):
    """Per-depth ``modal block - depth`` drift away from identity routing."""

    routing = np.asarray(routing, dtype=np.float64)
    languages = tuple(languages)
    order = _ordered(languages, order)
    displacement = signed_displacement(routing)
    depths = np.arange(routing.shape[1])

    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(figsize=(7.0, 4.0))
        axes.axhline(0.0, color="#8f9096", linewidth=0.9)
        for index in order:
            axes.plot(
                depths, displacement[index],
                color=language_color(languages[index]),
                marker="o", markersize=2.6, linewidth=1.2,
                label=languages[index],
            )
        axes.set_xlabel("depth")
        axes.set_ylabel("modal block - depth")
        # Keep a full block of headroom so a flat run reads as "no drift"
        # rather than as autoscaled noise.
        reach = max(float(np.abs(displacement).max()), 1.0) * 1.15
        axes.set_ylim(-reach, reach)
        axes.set_title(
            "Signed routing displacement (0 = identity routing)"
        )
        axes.grid(True, linewidth=0.4, alpha=0.3)
        axes.legend(loc="best", frameon=False, fontsize=7, ncols=2)
    return figure


def fig_language_similarity(routing, languages):
    """Dendrogram-ordered language x language mean-over-depth JSD."""

    languages = tuple(languages)
    distance = pairwise_language_distance(routing)
    order, merges = average_linkage(distance)
    order = list(order)
    position = {leaf: index for index, leaf in enumerate(order)}

    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(
            2, 1,
            figsize=(0.52 * len(languages) + 3.4,
                     0.52 * len(languages) + 4.2),
            height_ratios=(1, 3.0),
            sharex=True,
        )
        centres = {leaf: float(position[leaf]) for leaf in range(len(order))}
        peak = 0.0
        for left, right, height, leaves in merges:
            left_x, right_x = centres[left], centres[right]
            axes[0].plot(
                [left_x, left_x, right_x, right_x],
                [0.0, height, height, 0.0],
                color="#8f9096", linewidth=0.9,
            )
            centres[len(centres)] = float(
                np.mean([position[leaf] for leaf in leaves])
            )
            peak = max(peak, height)
        axes[0].set_xlim(-0.5, len(order) - 0.5)
        axes[0].set_ylim(0.0, peak * 1.15 if peak > 0 else 1.0)
        axes[0].set_xticks([])
        axes[0].set_ylabel("JSD")
        axes[0].set_title("Language routing similarity (average linkage)")
        for spine in ("top", "right", "bottom"):
            axes[0].spines[spine].set_visible(False)

        image = axes[1].imshow(
            distance[np.ix_(order, order)],
            cmap="magma", vmin=0.0, interpolation="nearest",
        )
        axes[1].set_xticks(np.arange(len(order)))
        axes[1].set_yticks(np.arange(len(order)))
        labels = [languages[leaf] for leaf in order]
        axes[1].set_xticklabels(labels, rotation=90)
        axes[1].set_yticklabels(labels)
        # Both axes take the colorbar so the dendrogram leaves stay aligned
        # with the heatmap columns underneath them.
        figure.colorbar(
            image, ax=axes, shrink=0.7, label="mean-over-depth JSD",
        )
    return figure
