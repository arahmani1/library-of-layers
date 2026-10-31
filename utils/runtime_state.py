"""Run identity and exact distributed-resume helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from omegaconf import OmegaConf
import torch.distributed as dist

from model.attention_strategies import parse_attention_strategy_name


MOE_FFN_MUON_FORMAT = 1


def config_fingerprint(cfg: dict[str, Any], world_size: int) -> str:
    """Hash every current setting that affects the training trajectory."""

    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError("Config fingerprint input must resolve to a mapping")

    model = dict(cfg["model"])
    if "attention_strategy" in model:
        model["attention_strategy"] = parse_attention_strategy_name(
            model["attention_strategy"]
        ).value
    if "normalization" in model:
        model["normalization"] = str(model["normalization"]).lower()
    if "first_window_routing" in model:
        model["first_window_routing"] = str(
            model["first_window_routing"]
        ).lower()
    if model.get("normalization", "affine_rmsnorm") == "affine_rmsnorm":
        model.pop("normalization", None)
    if model.get("first_window_length") is None:
        model.pop("first_window_length", None)
    if model.get("window") is None:
        model.pop("window", None)
    if model.get("window_stride") is None:
        model.pop("window_stride", None)
    if model.get("hidden_key_span") is None:
        model.pop("hidden_key_span", None)
    else:
        model["hidden_key_span"] = [
            int(bound) for bound in model["hidden_key_span"]
        ]
    training = dict(cfg["training"])
    if bool(model.get("moe_ffn", False)):
        model["moe_ffn_muon_format"] = MOE_FFN_MUON_FORMAT
    recurrence = dict(training.get("recurrence_curriculum") or {})
    if not recurrence.get("first_window_full_backprop", False):
        recurrence.pop("first_window_full_backprop", None)
    if recurrence.get("backprop_depth") is None:
        recurrence.pop("backprop_depth", None)
    if recurrence:
        training["recurrence_curriculum"] = recurrence
    elif "recurrence_curriculum" in training:
        training.pop("recurrence_curriculum")
    data = dict(cfg["data"])
    if list(data.get("validation_sets", ["val"])) == ["val"]:
        data.pop("validation_sets", None)
    if data.get("loss_sections") is None:
        data.pop("loss_sections", None)
    if data.get("tag_span") is None:
        data.pop("tag_span", None)
    else:
        data["tag_span"] = [int(bound) for bound in data["tag_span"]]
    identity = {
        "format_version": 4,
        "model": model,
        "training": training,
        "data": data,
        "seed": int(cfg.get("seed", 42)),
        "world_size": int(world_size),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:12]


def capture_rank_runtime_state(
    device: str | torch.device,
    tensors: dict[str, torch.Tensor | None] | None = None,
) -> dict[str, Any]:
    """Capture RNG streams and optional trainer tensors for one rank."""

    state: dict[str, Any] = {
        "format_version": 1,
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state": None,
        "tensors": {},
    }
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state(device_obj).cpu()

    for name, value in (tensors or {}).items():
        state["tensors"][name] = (
            None if value is None else value.detach().cpu().clone()
        )
    return state


def gather_rank_runtime_states(
    local_state: dict[str, Any],
    world_size: int,
) -> list[dict[str, Any]]:
    """Collect every rank's runtime state on every rank."""

    gathered: list[dict[str, Any] | None] = [None] * int(world_size)
    dist.all_gather_object(gathered, local_state)
    if any(state is None for state in gathered):
        raise RuntimeError("Failed to gather runtime state from every DDP rank")
    return [state for state in gathered if state is not None]


def restore_rank_runtime_state(
    states: list[dict[str, Any]],
    rank: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor | None]:
    """Restore one rank and return its optional trainer tensors."""

    if not isinstance(states, (list, tuple)) or rank >= len(states):
        raise ValueError(
            f"Checkpoint runtime state does not contain DDP rank {rank}"
        )
    state = states[rank]
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint runtime-state format")

    torch.set_rng_state(state["torch_rng_state"].cpu())

    device_obj = torch.device(device)
    cuda_state = state.get("cuda_rng_state")
    if cuda_state is not None:
        if device_obj.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device_obj)

    restored = {}
    for name, value in state.get("tensors", {}).items():
        restored[name] = None if value is None else value.to(device_obj)
    return restored
