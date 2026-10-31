"""Hydra resolution and shared distributed-entrypoint setup."""

from __future__ import annotations

from dataclasses import dataclass
import os
from collections.abc import Collection
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from data.sudoku_sections import LOSS_SECTIONS, layout_for
from model.factory import architecture_is_routed
from model.config import validate_huginn_config, validate_paper_normalization


def _drop_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_nulls(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_nulls(item) for item in value if item is not None]
    return value


def resolve_config(cfg: DictConfig) -> DictConfig:
    """Return a fully resolved, null-free Hydra config for runtime use."""

    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise TypeError("The Hydra job config must resolve to a mapping")
    return OmegaConf.create(_drop_nulls(resolved))


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert runtime config to the plain mapping stored in checkpoints."""

    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise TypeError("The runtime config must resolve to a mapping")
    return resolved


def validate_training_entrypoint(
    cfg: DictConfig, *, supported_arches: Collection[str]
) -> None:
    """Validate the model/config contract for the unified trainer."""

    for section in ("model", "shape", "training", "data", "logging"):
        if OmegaConf.select(cfg, section) is None:
            raise ValueError(f"Missing required config section {section!r}")

    for path in (
        "training.max_steps",
        "training.batch_size",
        "training.grad_accum_steps",
        "logging.log_every",
        "logging.eval_every",
        "logging.save_every",
    ):
        value = OmegaConf.select(cfg, path)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{path} must be a positive integer")

    model_cfg = cfg.model
    arch = str(model_cfg.arch)
    if arch not in supported_arches:
        supported = ", ".join(sorted(supported_arches))
        raise ValueError(
            f"Unsupported model.arch={arch!r}; expected one of: {supported}"
        )
    validate_paper_normalization(model_cfg)
    if arch == "huginn":
        validate_huginn_config(model_cfg)
    shape_cfg = cfg.shape
    shape_name = str(shape_cfg.name)
    shape_arches = {
        str(name) for name in shape_cfg.get("supported_arches", ())
    }
    if arch not in shape_arches:
        raise ValueError(
            f"model shape {shape_name!r} does not support "
            f"model.arch={arch!r}."
        )
    routed = model_cfg.routed
    expected_routed = architecture_is_routed(arch)
    if not isinstance(routed, bool) or routed is not expected_routed:
        raise ValueError(
            f"model.routed must be {expected_routed!r} for model.arch={arch!r}."
        )
    if expected_routed and str(model_cfg.get("first_window_routing", "")).lower() != "learned_token_depth":
        raise ValueError("model.first_window_routing must be 'learned_token_depth'")
    hidden_key_span = OmegaConf.select(cfg, "model.hidden_key_span")
    if expected_routed and hidden_key_span is not None:
        hidden_bounds = [int(bound) for bound in hidden_key_span]
        if (
            len(hidden_bounds) != 2
            or not 0 <= hidden_bounds[0] < hidden_bounds[1]
        ):
            raise ValueError(
                "model.hidden_key_span must be [start, stop) with "
                f"0 <= start < stop, got {hidden_key_span!r}."
            )
        hidden_stop = hidden_bounds[1]
        first_window_length = OmegaConf.select(
            cfg, "model.first_window_length"
        )
        if (
            first_window_length is None
            or int(first_window_length) != hidden_stop
        ):
            raise ValueError(
                "On routed architectures model.hidden_key_span must end at "
                "model.first_window_length so routed window attention cannot "
                "expose hidden keys to later tokens."
            )
    loss_sections = OmegaConf.select(cfg, "data.loss_sections")
    if loss_sections is not None:
        if str(loss_sections) not in LOSS_SECTIONS:
            expected = ", ".join(sorted(LOSS_SECTIONS))
            raise ValueError(
                f"Unknown data.loss_sections={loss_sections!r}; "
                f"expected one of: {expected}"
            )
        if arch != "lol_dynamic_window":
            # This model's own loss mask is all true, so the trainer can count
            # the final supervised set before forward and normalize it over
            # the whole optimizer step. Other model masks are not known yet.
            raise ValueError(
                "data.loss_sections is only supported for "
                f"model.arch='lol_dynamic_window', not {arch!r}."
            )
    tag_span = OmegaConf.select(cfg, "data.tag_span")
    if loss_sections is not None:
        # Only the strategy corpora have sections, and the maze label is
        # ordinary cell syntax that nothing but the span can mark.
        layout = layout_for(OmegaConf.select(cfg, "data.dataset"))
        if tag_span is None:
            raise ValueError(
                f"data.dataset={layout.name!r} marks its strategy label by "
                "position only; set data.tag_span to the packed manifest's "
                "layout.tag_span together with data.loss_sections."
            )
    if tag_span is not None:
        bounds = [int(bound) for bound in tag_span]
        if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1]:
            raise ValueError(
                "data.tag_span must be [start, stop) with 0 <= start < stop, "
                f"got {tag_span!r}."
            )
        if loss_sections is None:
            raise ValueError(
                "data.tag_span only labels sudoku or maze sections; set "
                "data.loss_sections as well."
            )
        hidden = OmegaConf.select(cfg, "model.hidden_key_span")
        if hidden is not None and [int(b) for b in hidden] != bounds:
            raise ValueError(
                "model.hidden_key_span must equal data.tag_span when both "
                f"are set, got {list(hidden)!r} and {bounds!r}."
            )


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    is_master: bool
    device: str


def initialize_distributed(cfg: DictConfig) -> DistributedContext:
    """Initialize NCCL, bind the local CUDA device, and seed this rank."""

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    cfg.device = device
    torch.manual_seed(int(cfg.seed) + rank)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_master=rank == 0,
        device=device,
    )


def ddp_find_unused_parameters(
    model: torch.nn.Module,
    model_cfg: DictConfig,
    *,
    recurrence_backprop_depth: int | None = None,
) -> bool:
    """Choose the DDP graph policy from routing and recurrence behavior."""

    if not bool(model_cfg.routed):
        return False
    router_parameters_may_be_unused = (
        not bool(model_cfg.get("weight_tie_router", True))
    )
    return (
        recurrence_backprop_depth is not None
        or router_parameters_may_be_unused
        or not bool(getattr(model, "grouped_window_dispatch", False))
    )


def clear_fresh_checkpoints(cfg: DictConfig, resume_id: str) -> list[str]:
    """Remove only this run's latest checkpoint and temporary file."""

    if not bool(cfg.run.fresh):
        return []
    save_dir = cfg.save_dir
    latest_path = os.path.join(save_dir, f"latest_{resume_id}.pt")
    candidates = [latest_path, latest_path + ".tmp"]
    removed = []
    for checkpoint_path in candidates:
        if os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
            removed.append(checkpoint_path)
    return removed
