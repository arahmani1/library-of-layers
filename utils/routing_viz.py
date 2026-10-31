"""Matplotlib routing figures and the distribution math they share.

This module is the leaf of the multilingual routing-diagnostics stack:
``utils/routing_viz_multi.py`` and ``utils/routing_logger.py`` both import its
palette, its distribution helpers, and its clustering. Nothing here touches
W&B, torch, or the training loop, so the figures stay reproducible offline
from a stored ``counts.npz``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Stable colorblind-friendly qualitative palette, most separable colours first.
LANGUAGE_COLORS = (
    "#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc",
    "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9",
)

# Palette slots follow the nesting order of the FineWeb2 presets rather than
# the alphabetical registry, so the languages a two- or five-language run uses
# take the most separable colours and the neutral grey lands on a language
# that only appears once there are ten of them. A language's colour is fixed
# here and never depends on the subset a run happens to evaluate, so two runs'
# figures can be read side by side.
_PALETTE_ORDER = (
    "rus_Cyrl", "cmn_Hani", "spa_Latn", "arb_Arab", "hin_Deva",
    "deu_Latn", "jpn_Jpan", "kor_Hang", "tur_Latn", "fra_Latn",
)

_EPS = 1e-12
_JITTER_SEED = 6_700_417
_BACKGROUND = "#111318"
_FOREGROUND = "#e6e8ec"
_GRID = "#2b2f38"

PLOT_RC = {
    "figure.facecolor": _BACKGROUND,
    "savefig.facecolor": _BACKGROUND,
    "axes.facecolor": _BACKGROUND,
    "axes.edgecolor": _GRID,
    "axes.labelcolor": _FOREGROUND,
    "axes.titlecolor": _FOREGROUND,
    "text.color": _FOREGROUND,
    "xtick.color": _FOREGROUND,
    "ytick.color": _FOREGROUND,
    "grid.color": _GRID,
    "figure.dpi": 110,
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "figure.constrained_layout.use": True,
}


def language_color(language: str) -> str:
    """Return the stable colour a language keeps across every run."""

    if language in _PALETTE_ORDER:
        index = _PALETTE_ORDER.index(language)
    else:
        # Not ``hash``: string hashing is salted per process, so the colour of
        # a non-preset split would change between runs and even between ranks.
        index = len(_PALETTE_ORDER) + sum(language.encode())
    return LANGUAGE_COLORS[index % len(LANGUAGE_COLORS)]


# --------------------------------------------------------------------------
# Distribution math shared by the figures and the scalar metrics.
# --------------------------------------------------------------------------


def entropy_bits(probabilities) -> np.ndarray:
    """Shannon entropy in bits over the last axis."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    return -(
        probabilities * np.log2(np.clip(probabilities, _EPS, None))
    ).sum(-1)


def jensen_shannon(left, right) -> np.ndarray:
    """Base-2 Jensen-Shannon divergence over the last axis, in ``[0, 1]``."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mixture = 0.5 * (left + right)
    divergence = entropy_bits(mixture) - 0.5 * (
        entropy_bits(left) + entropy_bits(right)
    )
    return np.clip(divergence, 0.0, 1.0)


def pooled_distribution(routing) -> np.ndarray:
    """Unweighted mean over languages of ``[L, D, K]`` routing shares."""

    return np.asarray(routing, dtype=np.float64).mean(0)


def reference_distribution(routing, languages, reference) -> np.ndarray:
    """Return the ``[D, K]`` panel every language is compared against."""

    if reference in (None, "", "pooled"):
        return pooled_distribution(routing)
    languages = list(languages)
    if reference not in languages:
        raise ValueError(
            f"pivot {reference!r} is not one of the evaluated languages "
            f"{languages}"
        )
    return np.asarray(routing, dtype=np.float64)[languages.index(reference)]


def pairwise_language_distance(routing) -> np.ndarray:
    """Mean-over-depth JSD between every language pair of ``[L, D, K]``."""

    routing = np.asarray(routing, dtype=np.float64)
    return jensen_shannon(routing[:, None], routing[None, :]).mean(-1)


def modal_block(routing) -> np.ndarray:
    """Most-visited block per ``(language, depth)``; block ids are nominal."""

    return np.asarray(routing, dtype=np.float64).argmax(-1)


def signed_displacement(routing) -> np.ndarray:
    """``argmax_k P[l, d, k] - d``: the only legal scalar over block ids.

    Blocks and depths share one index space, so their difference is
    meaningful even though an average of raw block ids never is.
    """

    modal = modal_block(routing)
    depths = np.arange(modal.shape[-1], dtype=np.float64)
    return modal.astype(np.float64) - depths


def average_linkage(distance):
    """Average-linkage clustering of a square distance matrix.

    Returns ``(leaf_order, merges)`` where ``merges`` holds
    ``(left_id, right_id, height, leaves)`` in merge order and cluster ids
    below ``n`` are leaves. Implemented here rather than pulled from SciPy:
    the language count is at most ten, and the routing stack otherwise has no
    SciPy dependency.
    """

    distance = np.asarray(distance, dtype=np.float64)
    count = distance.shape[0]
    if count < 2:
        return tuple(range(count)), []

    members = {index: (index,) for index in range(count)}
    sizes = {index: 1 for index in range(count)}
    active = list(range(count))
    pairs = {
        (left, right): float(distance[left, right])
        for left in range(count)
        for right in range(left + 1, count)
    }

    def separation(left, right):
        return pairs[(min(left, right), max(left, right))]

    merges = []
    next_id = count
    while len(active) > 1:
        height, left, right = min(
            (separation(left, right), left, right)
            for position, left in enumerate(active)
            for right in active[position + 1:]
        )
        merged = next_id
        next_id += 1
        for other in active:
            if other in (left, right):
                continue
            pairs[(min(merged, other), max(merged, other))] = (
                separation(left, other) * sizes[left]
                + separation(right, other) * sizes[right]
            ) / (sizes[left] + sizes[right])
        members[merged] = members[left] + members[right]
        sizes[merged] = sizes[left] + sizes[right]
        active = [
            other for other in active if other not in (left, right)
        ] + [merged]
        merges.append((left, right, height, members[merged]))
    return members[active[0]], merges


def language_leaf_order(routing) -> tuple[int, ...]:
    """Dendrogram leaf order of the language x language JSD matrix."""

    return tuple(average_linkage(pairwise_language_distance(routing))[0])


def one_hot_routes(routes, n_blocks: int) -> np.ndarray:
    """``[W, D]`` block ids to ``[W, D * K]`` one-hot rows."""

    routes = np.asarray(routes, dtype=np.int64)
    windows, depths = routes.shape
    one_hot = np.zeros((windows, depths * int(n_blocks)), dtype=np.float32)
    flat = routes + np.arange(depths)[None, :] * int(n_blocks)
    one_hot[np.arange(windows)[:, None], flat] = 1.0
    return one_hot


def route_centroid_accuracy(routes, lang_ids, n_blocks: int) -> float:
    """Leave-one-out nearest-centroid accuracy of language from the route.

    Each window's one-hot route is compared against every language's mean
    route, its own language's mean recomputed without it. Chance is
    ``1 / languages``; 1.0 means the route alone identifies the language.
    Returns ``nan`` with fewer than two populated languages.
    """

    one_hot = one_hot_routes(routes, n_blocks).astype(np.float64)
    lang_ids = np.asarray(lang_ids, dtype=np.int64)
    languages = np.unique(lang_ids)
    if languages.size < 2 or one_hot.shape[0] < 2:
        return float("nan")
    sums = np.stack([one_hot[lang_ids == lang].sum(0) for lang in languages])
    counts = np.array([(lang_ids == lang).sum() for lang in languages])
    centroids = sums / counts[:, None]
    # [W, L] squared distances to every centroid, then swap in the
    # leave-one-out centroid for each window's own language.
    distances = ((one_hot[:, None, :] - centroids[None]) ** 2).sum(-1)
    own = np.searchsorted(languages, lang_ids)
    own_count = counts[own]
    loo = (sums[own] - one_hot) / np.maximum(own_count - 1, 1)[:, None]
    loo_distance = ((one_hot - loo) ** 2).sum(-1)
    singleton = own_count <= 1
    distances[np.arange(one_hot.shape[0]), own] = np.where(
        singleton, np.inf, loo_distance
    )
    predicted = distances.argmin(1)
    return float((predicted == own).mean())


def pca_2d(features) -> np.ndarray:
    """Project rows onto their first two principal components."""

    matrix = np.asarray(features, dtype=np.float64)
    matrix = matrix - matrix.mean(0, keepdims=True)
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0], 2))
    components = np.linalg.svd(matrix, full_matrices=False)[2]
    if components.shape[0] < 2:
        components = np.vstack(
            [components, np.zeros((2 - components.shape[0], matrix.shape[1]))]
        )
    components = components[:2]
    # SVD component signs are arbitrary. Pin each one by making its
    # largest-magnitude loading non-negative so separate processes and
    # numerical backends do not reflect an otherwise identical plot.
    pivots = np.abs(components).argmax(1)
    signs = np.where(components[np.arange(2), pivots] < 0.0, -1.0, 1.0)
    return matrix @ (components * signs[:, None]).T


def cluster_space_coordinates(routes, n_blocks: int) -> np.ndarray:
    """PCA coordinates plus deterministic display jitter for route rows."""

    routes = np.asarray(routes, dtype=np.int64)
    if routes.ndim != 2 or routes.shape[0] == 0:
        raise ValueError("cluster space needs a non-empty [W, D] reservoir")
    embedded = pca_2d(one_hot_routes(routes, n_blocks))
    # A hard-argmax router repeats route rows, so many windows land on one
    # exact coordinate and a large cluster looks like a single point. Fan
    # them out by a fixed fraction of the spread, from a fixed seed so the
    # scatter is stable frame to frame. The jitter is presentation only.
    spread = embedded.max(0) - embedded.min(0)
    return embedded + np.random.default_rng(_JITTER_SEED).normal(
        scale=np.maximum(spread, _EPS) * 0.012, size=embedded.shape
    )


# --------------------------------------------------------------------------
# Figures.
# --------------------------------------------------------------------------


def fig_cluster_space(
    routes,
    lang_ids,
    languages,
    n_blocks: int,
    *,
    n_tokens=None,
    title: str = (
        "Window routing cluster space (PCA of one-hot routes, jittered)"
    ),
):
    """Scatter a reservoir of raw window routes in two principal components.

    ``routes`` is ``[W, D]`` of block ids, ``lang_ids`` is ``[W]`` indices
    into ``languages``. Each window becomes a ``D * K`` one-hot vector, so
    proximity means "took the same blocks at the same depths" and no block id
    is ever averaged. Points carry a small fixed-seed jitter so repeated
    route rows stay countable; read cluster position, not point position.
    """

    routes = np.asarray(routes, dtype=np.int64)
    lang_ids = np.asarray(lang_ids, dtype=np.int64)
    languages = tuple(languages)
    windows = routes.shape[0]
    embedded = cluster_space_coordinates(routes, n_blocks)

    if n_tokens is None:
        sizes = np.full(windows, 6.0)
    else:
        weights = np.asarray(n_tokens, dtype=np.float64)
        span = max(float(weights.max() - weights.min()), _EPS)
        sizes = 3.0 + 12.0 * (weights - weights.min()) / span

    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(figsize=(6.6, 5.4))
        for index, language in enumerate(languages):
            selected = lang_ids == index
            if not selected.any():
                continue
            colour = language_color(language)
            axes.scatter(
                embedded[selected, 0],
                embedded[selected, 1],
                s=sizes[selected],
                c=colour,
                alpha=0.45,
                linewidths=0.0,
                label=f"{language} (n={int(selected.sum())})",
            )
            centroid = embedded[selected].mean(0)
            axes.scatter(
                centroid[0], centroid[1], s=120, marker="X",
                c=colour, edgecolors=_FOREGROUND, linewidths=0.8, zorder=5,
            )
        # Frame the bulk so a few rare patterns cannot stretch the axes past
        # the clusters. Outliers still plot; they fall outside the view.
        for axis, limits in (
            (axes.set_xlim, np.quantile(embedded[:, 0], (0.01, 0.99))),
            (axes.set_ylim, np.quantile(embedded[:, 1], (0.01, 0.99))),
        ):
            margin = max(float(limits[1] - limits[0]) * 0.12, _EPS)
            axis(limits[0] - margin, limits[1] + margin)
        axes.set_title(title)
        axes.set_xlabel("PC 1")
        axes.set_ylabel("PC 2")
        axes.grid(True, linewidth=0.4, alpha=0.35)
        axes.legend(
            loc="best", frameon=False, fontsize=7, markerscale=2.0,
        )
    return figure
