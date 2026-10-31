"""Complete local optimizer-step histories and shared loss-plot styling."""
from pathlib import Path

from plotting.local_data import ASSETS, source_path

import numpy as np
import pandas as pd

X_KEY = "train/total_flops"
Y_KEY = "train/ce_loss"

PAPER_RC = {
    "font.family": "sans-serif",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
}


def running_average(values, window: int) -> np.ndarray:
    """Return a causal trailing average with an expanding warmup."""

    if not isinstance(window, int) or window <= 0:
        raise ValueError("running-average window must be a positive integer")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    cumulative = np.cumsum(values)
    if values.size > window:
        cumulative[window:] -= cumulative[:-window]
    counts = np.minimum(np.arange(1, values.size + 1), window)
    return cumulative / counts


def validate_history(frame, specification, source: str):
    """Validate and canonically order one complete optimizer-step history."""

    required = ("global_step", X_KEY, Y_KEY)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{source} is missing columns {missing}")
    history = frame.loc[:, required].copy()
    if history.isna().any().any():
        raise ValueError(f"{source} contains missing required values")
    try:
        steps = pd.to_numeric(history["global_step"]).to_numpy()
        x = pd.to_numeric(history[X_KEY]).to_numpy(np.float64)
        y = pd.to_numeric(history[Y_KEY]).to_numpy(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} contains non-numeric history values") from error
    expected_steps = int(specification["steps"])
    if history.shape[0] != expected_steps:
        raise ValueError(
            f"{source} has {history.shape[0]} rows; expected {expected_steps}"
        )
    expected = np.arange(expected_steps, dtype=np.int64)
    if not np.array_equal(steps, expected):
        raise ValueError(f"{source} does not contain every optimizer step once")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{source} contains non-finite metric values")
    if (x < 0.0).any() or (y <= 0.0).any():
        raise ValueError(f"{source} contains invalid metric values")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(
            f"{source} does not have strictly increasing {X_KEY}"
        )
    history["global_step"] = steps
    history[X_KEY] = x
    history[Y_KEY] = y
    return history


def load_history(specification, data_dir=ASSETS):
    """Read the complete local CSV; absent inputs never trigger network access."""
    path = source_path(data_dir, specification["file"])
    return validate_history(pd.read_csv(path, float_precision="round_trip"), specification, specification["file"])


def load_curve(specification: dict[str, object], data_dir: Path = ASSETS) -> dict[str, object]:
    """Load a validated full-history curve from the selected dataset directory."""
    history = load_history(specification, data_dir)
    return {
        **specification,
        "step": history["global_step"].to_numpy(dtype=np.int64),
        "x": history[X_KEY].to_numpy(dtype=np.float64),
        "y": history[Y_KEY].to_numpy(dtype=np.float64),
    }
