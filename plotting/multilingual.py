"""Shared multilingual routing data and PCA drawing for paper figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from utils.routing_viz import (  # noqa: E402
    cluster_space_coordinates,
    language_color,
)

ARTIFACT_SCHEMA_VERSION = 2
REQUIRED_FIELDS = {
    "routes", "lang_ids", "n_tokens", "ordinals", "langs", "K", "D",
    "step", "schema_version",
}
PAPER_RC = {
    "font.family": "sans-serif",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
}


def load_routing_data(path: Path, expected_step: int) -> dict[str, object]:
    """Load and validate a reconstructable routing-reservoir NPZ."""

    with np.load(path) as stored:
        missing = sorted(REQUIRED_FIELDS - set(stored.files))
        if missing:
            raise ValueError(
                "routing artifact predates reconstructable schema 2; "
                f"missing {', '.join(missing)}. Run validation with the "
                "updated multilingual logger."
            )
        data = {key: stored[key].copy() for key in REQUIRED_FIELDS}

    scalar_values = {}
    for name in ("schema_version", "step", "K", "D"):
        value = np.asarray(data[name])
        if value.shape != () or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must be a scalar integer; got {value.dtype} {value.shape}")
        scalar_values[name] = int(value)
    schema_version = scalar_values["schema_version"]
    step = scalar_values["step"]
    n_blocks = scalar_values["K"]
    n_depths = scalar_values["D"]
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported routing artifact schema {schema_version}; expected "
            f"schema {ARTIFACT_SCHEMA_VERSION}"
        )
    if step != expected_step:
        raise ValueError(
            f"artifact contains global step {step}, expected {expected_step}"
        )

    integer_arrays = {}
    for name in ("routes", "lang_ids", "n_tokens", "ordinals"):
        value = np.asarray(data[name])
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must contain integers; got {value.dtype}")
        integer_arrays[name] = value.astype(np.int64, copy=False)
    routes = integer_arrays["routes"]
    lang_ids = integer_arrays["lang_ids"]
    n_tokens = integer_arrays["n_tokens"]
    ordinals = integer_arrays["ordinals"]

    stored_languages = np.asarray(data["langs"])
    if stored_languages.ndim != 1 or stored_languages.size == 0:
        raise ValueError(
            f"langs must be a non-empty one-dimensional array; "
            f"got {stored_languages.shape}"
        )
    raw_languages = stored_languages.tolist()
    if any(not isinstance(item, (str, bytes)) for item in raw_languages):
        raise ValueError("langs must contain only strings")
    languages = tuple(
        item.decode() if isinstance(item, bytes) else item
        for item in raw_languages
    )
    if any(not language for language in languages):
        raise ValueError("language names must be non-empty")
    if len(set(languages)) != len(languages):
        raise ValueError("language names must be unique")

    if n_depths <= 0:
        raise ValueError(f"D must be positive; got {n_depths}")
    if n_blocks <= 0:
        raise ValueError(f"K must be positive; got {n_blocks}")
    if routes.ndim != 2 or routes.shape[1] != n_depths or routes.shape[0] < 2:
        raise ValueError(
            f"routes must have shape [W, {n_depths}] with W >= 2; "
            f"got {routes.shape}"
        )
    windows = routes.shape[0]
    for name, values in (
        ("lang_ids", lang_ids), ("n_tokens", n_tokens),
        ("ordinals", ordinals),
    ):
        if values.shape != (windows,):
            raise ValueError(
                f"{name} must have shape [{windows}]; got {values.shape}"
            )
    if routes.min() < 0 or routes.max() >= n_blocks:
        raise ValueError(f"route block ids must lie in [0, {n_blocks})")
    if lang_ids.min() < 0 or lang_ids.max() >= len(languages):
        raise ValueError(
            f"language ids must lie in [0, {len(languages)})"
        )
    if (n_tokens <= 0).any():
        raise ValueError("every retained window must contain a valid token")
    if (ordinals < -1).any():
        raise ValueError("window ordinals must be -1 or non-negative")

    return {
        "routes": routes,
        "lang_ids": lang_ids,
        "n_tokens": n_tokens,
        "ordinals": ordinals,
        "languages": languages,
        "n_blocks": n_blocks,
        "n_depths": n_depths,
        "step": step,
    }


def load_checkpoint_pca(path: Path) -> tuple[dict[str, object], str]:
    """Read the plotted points saved by ``make_pca.py`` without rerunning PCA."""

    required = {
        "coordinates", "routes", "lang_ids", "n_tokens", "ordinals",
        "languages", "n_blocks", "n_depths",
        "checkpoint_step", "model",
    }
    with np.load(path, allow_pickle=False) as stored:
        missing = sorted(required - set(stored.files))
        if missing:
            raise ValueError(f"PCA data is missing {', '.join(missing)}")
        data = {name: stored[name].copy() for name in required}

    routes = data["routes"]
    coordinates = data["coordinates"]
    lang_ids = data["lang_ids"]
    n_tokens = data["n_tokens"]
    ordinals = data["ordinals"]
    if data["languages"].ndim != 1:
        raise ValueError("PCA languages must be a one-dimensional list")
    languages = tuple(str(name) for name in data["languages"])
    n_blocks = int(data["n_blocks"])
    n_depths = int(data["n_depths"])
    if (routes.ndim != 2 or routes.shape[0] < 2
            or routes.shape[1] != n_depths
            or not np.issubdtype(routes.dtype, np.integer)
            or n_blocks <= 0 or routes.min() < 0 or routes.max() >= n_blocks):
        raise ValueError("PCA routes must be valid [W, D] expert IDs")
    windows = routes.shape[0]
    if coordinates.shape != (windows, 2) or not np.isfinite(coordinates).all():
        raise ValueError("PCA coordinates must be finite [W, 2] points")
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("PCA languages must be a nonempty unique list")
    for name, values in (("lang_ids", lang_ids), ("n_tokens", n_tokens),
                         ("ordinals", ordinals)):
        if values.shape != (windows,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"PCA {name} must be an integer [W] array")
    if (lang_ids.min() < 0 or lang_ids.max() >= len(languages)
            or (n_tokens <= 0).any() or (ordinals <= 0).any()):
        raise ValueError("PCA rows must have valid language, tokens, and ordinal")
    step = int(data["checkpoint_step"])
    model = str(data["model"])
    if not model or step < 0:
        raise ValueError("PCA checkpoint provenance is incomplete")
    return {
        "coordinates": coordinates,
        "routes": routes,
        "lang_ids": lang_ids,
        "n_tokens": n_tokens,
        "ordinals": ordinals,
        "languages": languages,
        "n_blocks": n_blocks,
        "n_depths": n_depths,
        "step": step,
    }, f"routing PCA, step {step}"


def plot_multilingual(
    axes,
    data: dict[str, object],
    *,
    legend: bool = True,
    circled_languages: tuple[str, ...] = (),
    marker_area_scale: float = 1.0,
):
    """Render routing and ellipse the selected languages as one group."""

    routes = data["routes"]
    lang_ids = data["lang_ids"]
    n_tokens = data["n_tokens"]
    languages = data["languages"]
    embedded = data.get("coordinates")
    if embedded is None:
        embedded = cluster_space_coordinates(routes, data["n_blocks"])

    token_span = max(float(np.ptp(n_tokens)), 1.0)
    sizes = marker_area_scale * (
        4.0 + 12.0 * (n_tokens - n_tokens.min()) / token_span
    )
    with plt.rc_context(PAPER_RC):
        axes.set_facecolor("none")
        for index, language in enumerate(languages):
            selected = lang_ids == index
            if not selected.any():
                continue
            colour = language_color(language)
            axes.scatter(
                embedded[selected, 0], embedded[selected, 1],
                s=sizes[selected], c=colour, alpha=0.58,
                linewidths=0.0, label=language,
            )
            centroid = embedded[selected].mean(0)
            axes.scatter(
                centroid[0], centroid[1], s=90 * marker_area_scale,
                marker="X", c=colour,
                edgecolors="black", linewidths=0.55, zorder=5,
            )

        quantiles = np.stack([
            np.quantile(embedded[:, axis], (0.01, 0.99))
            for axis in range(2)
        ])
        spans = quantiles[:, 1] - quantiles[:, 0]
        reference_span = max(float(spans.max()), 1e-6)
        circle_indices = [
            index for index, language in enumerate(languages)
            if language in circled_languages
        ]
        selected = np.isin(lang_ids, circle_indices)
        outline_extent = None
        if selected.any():
            points = embedded[selected]
            centre = np.median(points, axis=0)
            offsets = points - centre
            eigenvalues, eigenvectors = np.linalg.eigh(
                offsets.T @ offsets / len(points)
            )
            order = np.argsort(eigenvalues)[::-1]
            basis = eigenvectors[:, order]
            scales = np.maximum(
                np.sqrt(np.maximum(eigenvalues[order], 0.0)),
                0.04 * reference_span,
            )
            radius = max(
                float(np.quantile(
                    np.linalg.norm(
                        ((embedded[lang_ids == index] - centre) @ basis) / scales,
                        axis=1,
                    ),
                    0.9,
                ))
                for index in circle_indices
                if np.any(lang_ids == index)
            )
            semiaxes = np.maximum(
                radius * scales + 0.025 * reference_span,
                0.08 * reference_span,
            )
            angle = np.degrees(np.arctan2(basis[1, 0], basis[0, 0]))
            ellipse = Ellipse(
                centre, width=2 * semiaxes[0], height=2 * semiaxes[1],
                angle=angle, fill=False, edgecolor="black", linewidth=1.0,
                linestyle=(0, (3, 2)), alpha=0.85, zorder=6,
            )
            ellipse.set_gid(
                "circled-languages:"
                + ",".join(languages[index] for index in circle_indices)
            )
            axes.add_patch(ellipse)
            half_extents = np.sqrt((basis**2) @ (semiaxes**2))
            outline_extent = (centre - half_extents, centre + half_extents)
        for axis, (setter, limits) in enumerate(zip(
            (axes.set_xlim, axes.set_ylim), quantiles,
        )):
            lower, upper = limits
            if outline_extent is not None:
                lower = min(lower, outline_extent[0][axis])
                upper = max(upper, outline_extent[1][axis])
            display_span = max(float(upper - lower), 0.05 * reference_span)
            centre = float((lower + upper) / 2)
            margin = 0.10 * display_span
            setter(
                centre - 0.5 * display_span - margin,
                centre + 0.5 * display_span + margin,
            )
        axes.set_xlabel("PC 1")
        axes.set_ylabel("PC 2")
        axes.set_aspect("equal", adjustable="box")
        axes.grid(True, color="#a7a7a7", linewidth=0.4, alpha=0.35)
        axes.set_axisbelow(True)
        axes.spines["left"].set_linewidth(0.65)
        axes.spines["bottom"].set_linewidth(0.65)
        if legend:
            axes.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5,
                frameon=False, columnspacing=1.25, handletextpad=0.35,
                markerscale=1.6,
            )


def pdf_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".pdf":
        raise argparse.ArgumentTypeError("output path must end in .pdf")
    return path
