"""Shared model construction for checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from model.factory import build_model


def build_evaluation_model(
    state_dict: Mapping[str, torch.Tensor],
    cfg: Mapping[str, Any],
):
    """Build an evaluation model and strictly load normalized weights."""

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        normalized[key.replace("._orig_mod.", ".")] = value

    model, arch = build_model(cfg["model"])
    model.load_state_dict(normalized, strict=True)
    return model, arch
