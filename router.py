"""Training support for routed and depth-recurrent architectures.

The executable training lifecycle belongs in :mod:`train`.  This module keeps
the routed model's optimizer partitioning, load balancing and extra checkpoint
state out of that shared loop, together with the recurrence curriculum that
``lol_dynamic_window`` and the unrouted ``huginn`` middle share.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from model.lol_window import LoLWindow
from model.config import ROUTER_OWNED_FIRST_WINDOW
from model.recurrence_curriculum import (
    RecurrenceSample,
    one_minus_sqrt_curriculum_mean,
)

RECURRENCE_EVAL_DEPTHS = (2, 4, 8, 16, 32, 64)


def needs_runtime(cfg) -> bool:
    """True when the trainer must build a :class:`RouterRuntime`."""

    return (
        bool(cfg.model.routed)
        or bool(cfg.model.get("moe_ffn", False))
        or str(cfg.model.arch) == "huginn"
    )


def recurrence_kwargs(depth: int | None, backprop_depth: int | None = None):
    """Model keyword arguments selecting one executed recurrence."""

    kwargs = {}
    if backprop_depth is not None:
        kwargs["backprop_steps"] = backprop_depth
    if depth is not None:
        kwargs["num_steps"] = depth
    return kwargs


@dataclass(frozen=True)
class LoadBalancingConfig:
    """Validated settings for router load balancing."""

    strategy: str = "aux_loss"
    bias_update_speed: float = 0.001
    bias_update_stop_fraction: float = 0.966
    sequence_aux_coeff: float = 0.0001

    @classmethod
    def from_training_config(cls, training_cfg) -> "LoadBalancingConfig":
        values = training_cfg.get("load_balancing") or {}
        return cls(
            strategy=str(values.get("strategy", cls.strategy)),
            bias_update_speed=float(
                values.get("bias_update_speed", cls.bias_update_speed)
            ),
            bias_update_stop_fraction=float(
                values.get(
                    "bias_update_stop_fraction",
                    cls.bias_update_stop_fraction,
                )
            ),
            sequence_aux_coeff=float(
                values.get("sequence_aux_coeff", cls.sequence_aux_coeff)
            ),
        )

    def __post_init__(self) -> None:
        if self.strategy not in {
            "aux_loss",
            "loss_free",
        }:
            raise ValueError(
                "training.load_balancing.strategy must be 'aux_loss', "
                "or 'loss_free', "
                f"got {self.strategy!r}."
            )
        if self.bias_update_speed < 0.0:
            raise ValueError(
                "training.load_balancing.bias_update_speed must be nonnegative"
            )
        if not 0.0 <= self.bias_update_stop_fraction <= 1.0:
            raise ValueError(
                "training.load_balancing.bias_update_stop_fraction must be "
                "in [0, 1]"
            )
        if self.sequence_aux_coeff < 0.0:
            raise ValueError(
                "training.load_balancing.sequence_aux_coeff must be nonnegative"
            )

    @property
    def is_loss_free(self) -> bool:
        return self.strategy == "loss_free"

    def summary(self) -> str:
        return (
            "Load balancing: DeepSeek-V3 "
            f"{self.strategy.replace('_', '-')} bias updates "
            f"(speed={self.bias_update_speed:g}, "
            f"stop_fraction={self.bias_update_stop_fraction:g}) plus "
            f"sequence aux coefficient={self.sequence_aux_coeff:g}"
        )


@dataclass
class RouterAuxPool:
    """Sufficient statistics pooled across gradient-accumulation batches."""

    oh_sum: torch.Tensor | None = None
    p_sum: torch.Tensor | None = None
    count: torch.Tensor | None = None
    gradient_count: torch.Tensor | None = None
    ffn_load_sum: torch.Tensor | None = None
    ffn_importance_sum: torch.Tensor | None = None


@dataclass
class RouterRuntime:
    """Resolved router configuration plus mutable resumable trainer state."""

    max_lr: float
    min_lr: float
    load_balancing: LoadBalancingConfig
    max_steps: int
    recurrence_enabled: bool
    recurrence_backprop_depth: int | None
    recurrence_start_depth: int
    recurrence_target_mean: float
    recurrence_growth_steps: int
    recurrence_hard_cap: int
    first_window_full_backprop: bool = False
    # False for a recurrent middle that samples its depth but routes nothing.
    uses_router: bool = True
    f_hat: torch.Tensor | None = None
    n_hat: torch.Tensor | None = None
    ffn_f_hat: torch.Tensor | None = None
    ffn_n_hat: torch.Tensor | None = None
    load_max_violation: torch.Tensor | None = None
    expert_bias_abs_max: torch.Tensor | None = None
    ffn_auxiliary_loss: torch.Tensor | None = None
    ffn_load_max_violation: torch.Tensor | None = None
    ffn_expert_bias_abs_max: torch.Tensor | None = None


    def bias_update_active(self, step: int) -> bool:
        stop_step = int(
            self.max_steps * self.load_balancing.bias_update_stop_fraction
        )
        return (
            self.load_balancing.is_loss_free
            and self.load_balancing.bias_update_speed > 0.0
            and step < stop_step
        )

    def sample_depth(self, step: int) -> RecurrenceSample | None:
        if not self.recurrence_enabled:
            return None
        mean = one_minus_sqrt_curriculum_mean(
            step,
            self.recurrence_growth_steps,
            self.recurrence_start_depth,
            self.recurrence_target_mean,
        )
        depth = max(1, min(round(mean), self.recurrence_hard_cap))
        gradient_depth = (
            depth if self.recurrence_backprop_depth is None
            else min(depth, self.recurrence_backprop_depth)
        )
        return RecurrenceSample(
            total_depth=depth,
            gradient_depth=gradient_depth,
            scheduled_mean=mean,
        )

    def validation_depths(self, current_depth: int | None) -> tuple[int | None, ...]:
        if not self.recurrence_enabled:
            return (None,)
        cap = self.recurrence_hard_cap
        depths = {*RECURRENCE_EVAL_DEPTHS, current_depth, cap}
        return tuple(sorted(depth for depth in depths if depth <= cap))

    def restore_tensors(self, tensors: dict[str, torch.Tensor | None]) -> None:
        self.f_hat = tensors.get("f_hat")
        self.n_hat = tensors.get("n_hat")
        self.ffn_f_hat = tensors.get("ffn_f_hat")
        self.ffn_n_hat = tensors.get("ffn_n_hat")

    def runtime_tensors(self) -> dict[str, torch.Tensor | None]:
        tensors = {
            "f_hat": self.f_hat,
            "n_hat": self.n_hat,
            "ffn_f_hat": self.ffn_f_hat,
            "ffn_n_hat": self.ffn_n_hat,
        }
        return tensors

    def log_fields(self, sample: RecurrenceSample | None) -> dict[str, object]:
        fields: dict[str, object] = {}
        if self.recurrence_enabled:
            fields["recurrent_depth"] = sample.total_depth
            fields["recurrent_depth_target_mean"] = sample.scheduled_mean
            fields["recurrent_gradient_depth"] = sample.gradient_depth
            fields["recurrent_no_grad_depth"] = sample.no_grad_depth
            if self.first_window_full_backprop:
                fields["recurrent_first_window_gradient_depth"] = sample.total_depth
        if self.load_balancing.is_loss_free:
            if self.load_max_violation is not None:
                fields["load_max_violation"] = self.load_max_violation
            if self.expert_bias_abs_max is not None:
                fields["expert_bias_abs_max"] = self.expert_bias_abs_max
        return fields

    def ffn_log_fields(self) -> dict[str, torch.Tensor]:
        fields: dict[str, torch.Tensor] = {}
        if self.ffn_auxiliary_loss is not None:
            fields["auxiliary_loss"] = self.ffn_auxiliary_loss
        if self.load_balancing.is_loss_free:
            if self.ffn_load_max_violation is not None:
                fields["load_max_violation"] = self.ffn_load_max_violation
            if self.ffn_expert_bias_abs_max is not None:
                fields["expert_bias_abs_max"] = self.ffn_expert_bias_abs_max
        return fields


def build_runtime(cfg, is_master: bool) -> RouterRuntime:
    """Validate routing settings and construct their runtime state."""

    has_outer_router = (
        cfg.model.arch == "lol_dynamic_window" and bool(cfg.model.routed)
    )
    uses_ffn_router = bool(cfg.model.get("moe_ffn", False))
    # huginn samples its executed depth like LoL but routes nothing.
    has_recurrent_middle = (
        has_outer_router or str(cfg.model.arch) == "huginn"
    )
    if not (has_outer_router or uses_ffn_router or has_recurrent_middle):
        raise ValueError(
            "RouterRuntime requires outer routing, FFN routing, or a "
            "recurrent middle"
        )

    router_max_lr = float(
        cfg.training.get("router_max_lr", cfg.training.max_lr)
    )
    configured_min = cfg.training.get("router_min_lr")
    if configured_min is None:
        main_max = float(cfg.training.max_lr)
        router_min_lr = (
            router_max_lr * float(cfg.training.min_lr) / main_max
            if main_max > 0
            else router_max_lr
        )
    else:
        router_min_lr = float(configured_min)

    lb_mode = str(cfg.training.lb_mode)
    if lb_mode != "global":
        raise ValueError(
            f"training.lb_mode must be 'global', got {lb_mode!r}."
        )

    load_balancing = LoadBalancingConfig.from_training_config(cfg.training)
    if load_balancing.is_loss_free and is_master:
        print(load_balancing.summary())

    recurrence_cfg = cfg.training.get("recurrence_curriculum") or {}
    recurrence_enabled = has_recurrent_middle and bool(
        recurrence_cfg.get("enabled", False)
    )
    recurrence_mode = (
        str(recurrence_cfg.get("mode", "linear"))
        if recurrence_enabled
        else None
    )
    mean_schedule = str(recurrence_cfg.get("mean_schedule", "one_minus_sqrt"))
    backprop_value = recurrence_cfg.get("backprop_depth")
    if backprop_value is not None and type(backprop_value) is not int:
        raise ValueError(
            "recurrence_curriculum.backprop_depth must be an integer or null."
        )
    backprop_depth = (
        None if backprop_value is None else int(backprop_value)
    )
    if backprop_depth is not None and not recurrence_enabled:
        raise ValueError(
            "recurrence_curriculum.backprop_depth requires enabled=true."
        )
    truncated_recurrence = backprop_depth is not None
    first_window_full_backprop = recurrence_cfg.get("first_window_full_backprop", False)
    if type(first_window_full_backprop) is not bool:
        raise ValueError("first_window_full_backprop must be a boolean")
    if first_window_full_backprop and (
        not recurrence_enabled or not truncated_recurrence
        or str(cfg.model.arch) != "lol_dynamic_window"
    ):
        raise ValueError("first_window_full_backprop requires a LoLDynamicWindow curriculum with backprop_depth")
    first_window_routing = (
        str(cfg.model.get("first_window_routing", "")).lower()
        if has_outer_router else None
    )
    if has_outer_router and first_window_routing not in ROUTER_OWNED_FIRST_WINDOW:
        raise ValueError("model.first_window_routing must be 'learned_token_depth'")
    hard_cap = int(cfg.model.get("recurrent_depth", 1))
    start_value = recurrence_cfg.get("start_depth")
    start_depth = 1 if start_value is None else int(start_value)
    target_value = recurrence_cfg.get("target_mean")
    target_mean = hard_cap if target_value is None else float(target_value)
    warmup_fraction = float(recurrence_cfg.get("warmup_frac", 0.25))
    growth_value = recurrence_cfg.get("growth_steps")
    growth_steps = max(
        1,
        int(
            warmup_fraction * cfg.training.max_steps
            if growth_value is None
            else growth_value
        ),
    )

    if recurrence_enabled:
        if recurrence_mode != "linear":
            raise ValueError(
                "recurrence_curriculum.mode must be 'linear', "
                f"got {recurrence_mode!r}."
            )
        if mean_schedule != "one_minus_sqrt":
            raise ValueError(
                "recurrence_curriculum.mean_schedule must be "
                f"'one_minus_sqrt', got {mean_schedule!r}."
            )
        if not 1 <= start_depth <= hard_cap:
            raise ValueError(
                "recurrence_curriculum.start_depth must be in "
                f"[1, {hard_cap}], got {start_depth}."
            )
        if target_mean < start_depth:
            raise ValueError(
                "recurrence_curriculum.target_mean must be >= start_depth, "
                f"got {target_mean}."
            )
        if target_mean > hard_cap:
            raise ValueError(
                "linear recurrence_curriculum.target_mean cannot exceed "
                f"recurrent_depth ({hard_cap}), got {target_mean}."
            )
        if truncated_recurrence:
            injection = cfg.model.get("recurrent_input_injection", "none")
            injection = getattr(injection, "value", injection)
            requirements = (
                (1 <= backprop_depth <= hard_cap, f"1 <= backprop_depth <= {hard_cap}"),
            )
            if not has_outer_router:
                # The gradient suffix reaches the prelude only through the
                # injected input, so a plain looped middle cannot truncate.
                requirements += ((
                    injection in {"adapter", "diagonal"},
                    "model.recurrent_input_injection=adapter or diagonal",
                ),)
            else:
                requirements += (
                    (injection == "diagonal", "model.recurrent_input_injection=diagonal"),
                    (
                        bool(cfg.model.get("weight_tie_router", True)),
                        "model.weight_tie_router=true",
                    ),
                    (
                        not cfg.model.get("moe_ffn", False)
                        or str(
                            cfg.model.get("moe_ffn_scope", "all")
                        ).lower() == "prelude_coda",
                        "model.moe_ffn_scope=prelude_coda or "
                        "model.moe_ffn=false",
                    ),
                    (
                        first_window_full_backprop,
                        "first_window_full_backprop=true with learned_token_depth",
                    ),
                )
            for valid, requirement in requirements:
                if not valid:
                    raise ValueError(f"Truncated recurrence requires {requirement}.")
        if is_master:
            print(
                f"Recurrence curriculum: mode={recurrence_mode} "
                f"start={start_depth} target={target_mean:g} "
                f"growth_steps={growth_steps} "
                f"hard_cap={hard_cap}"
            )

    return RouterRuntime(
        max_lr=router_max_lr,
        min_lr=router_min_lr,
        load_balancing=load_balancing,
        recurrence_backprop_depth=backprop_depth,
        max_steps=int(cfg.training.max_steps),
        recurrence_enabled=recurrence_enabled,
        recurrence_start_depth=start_depth,
        recurrence_target_mean=target_mean,
        recurrence_growth_steps=growth_steps,
        recurrence_hard_cap=hard_cap,
        first_window_full_backprop=first_window_full_backprop,
        uses_router=has_outer_router or uses_ffn_router,
    )


def optimizer_param_groups(model, cfg, runtime: RouterRuntime):
    """Partition all routing parameters into their decoupled LR group."""

    raw_model = model.module if hasattr(model, "module") else model
    router_modules = []
    if bool(cfg.model.routed):
        router_modules.append(raw_model.outer_routers)
    if bool(cfg.model.get("moe_ffn", False)):
        router_modules.extend(
            module.router for module in raw_model._ffn_moe_modules()
        )
    if not router_modules:
        raise ValueError("Model does not expose any routing modules")

    router_ids = {
        id(parameter)
        for module in router_modules
        for parameter in module.parameters()
    }
    router_params = [
        parameter for parameter in model.parameters()
        if id(parameter) in router_ids
    ]
    main_params = [
        parameter for parameter in model.parameters()
        if id(parameter) not in router_ids
    ]
    if not router_params or not main_params:
        raise ValueError("Router/main optimizer partition must contain both groups")

    muon_router = bool(cfg.training.get("muon_router", True))
    router_metadata = {
        "lr": runtime.max_lr,
        "_max_lr": runtime.max_lr,
        "_min_lr": runtime.min_lr,
        "_lr_group": "router",
        "weight_decay": 0.0,
    }
    groups = [{
        "params": main_params,
        "lr": float(cfg.training.max_lr),
        "_max_lr": float(cfg.training.max_lr),
        "_min_lr": float(cfg.training.min_lr),
        "_lr_group": "main",
    }]
    router_partitions = ((router_params, False),)
    if muon_router:
        first_window_query_ids = {
            id(query)
            for module in router_modules
            for child in module.modules()
            if (query := getattr(child, "first_window_query", None)) is not None
        }
        router_partitions = (
            ([p for p in router_params if id(p) not in first_window_query_ids], True),
            ([p for p in router_params if id(p) in first_window_query_ids], False),
        )
    groups.extend(
        {
            "params": params,
            **router_metadata,
            "_muon_eligible": muon_eligible,
        }
        for params, muon_eligible in router_partitions
        if params
    )
    return groups, router_params


def _pool_router_stats(pool: RouterAuxPool, stats, *, include_probs: bool):
    if pool.oh_sum is None:
        pool.oh_sum = stats["oh_sum"]
        pool.count = stats["count"]
        if include_probs:
            pool.p_sum = stats["p_sum"].detach()
        return
    pool.oh_sum = pool.oh_sum + stats["oh_sum"]
    pool.count = pool.count + stats["count"]
    if include_probs:
        pool.p_sum = pool.p_sum + stats["p_sum"].detach()


def _pool_ffn_stats(pool: RouterAuxPool, stats) -> None:
    loads = stats["ffn_load_sum"].detach()
    importance = stats["ffn_importance_sum"].detach()
    if pool.ffn_load_sum is None:
        pool.ffn_load_sum = loads
        pool.ffn_importance_sum = importance
    else:
        pool.ffn_load_sum = pool.ffn_load_sum + loads
        pool.ffn_importance_sum = pool.ffn_importance_sum + importance


def _ffn_global_surrogate(stats, cfg, runtime: RouterRuntime):
    importance = stats["ffn_importance_sum"]
    experts = importance.size(-1)
    if (
        runtime.ffn_f_hat is not None
        and runtime.ffn_f_hat.shape != importance.shape
    ):
        runtime.ffn_f_hat = None
        runtime.ffn_n_hat = None
    if runtime.ffn_f_hat is None:
        runtime.ffn_f_hat = torch.full_like(importance, 1.0 / experts)
        global_count = importance.sum(-1).detach().clone()
        _all_reduce_sum(global_count)
        runtime.ffn_n_hat = (
            global_count.clamp_min(1.0)
            / _distributed_world_size()
            * cfg.training.grad_accum_steps
        )
    return cfg.training.grad_accum_steps * experts * (
        (runtime.ffn_f_hat * importance)
        / runtime.ffn_n_hat.unsqueeze(-1)
    ).sum(-1).mean()


def _gradient_depth_stats(stats):
    """Select only score statistics from gradient-bearing recurrences."""

    if "gradient_p_sum" not in stats:
        return stats
    gradient_depth = stats["gradient_p_sum"].size(0)
    return {
        **stats,
        "oh_sum": stats["oh_sum"][-gradient_depth:],
        "p_sum": stats["gradient_p_sum"],
        "count": stats["gradient_count"],
    }




def _outer_global_surrogate(
    stats, cfg, runtime: RouterRuntime, n_choices: int
):
    f_hat = runtime.f_hat
    if f_hat is not None and f_hat.shape != stats["p_sum"].shape:
        runtime.f_hat = None
        runtime.n_hat = None
        f_hat = None
    if f_hat is None:
        f_hat = torch.full_like(stats["oh_sum"], 1.0 / n_choices)
        global_count = stats["count"].detach().clone()
        _all_reduce_sum(global_count)
        n_hat = (
            global_count.clamp_min(1.0)
            / _distributed_world_size()
            * cfg.training.grad_accum_steps
        )
        runtime.f_hat = f_hat
        runtime.n_hat = n_hat
    else:
        n_hat = runtime.n_hat
    return cfg.training.grad_accum_steps * n_choices * (
        (f_hat * stats["p_sum"]) / n_hat.unsqueeze(-1)
    ).sum(dim=-1).mean()


def _outer_aux_metric(stats, n_choices: int):
    count = stats["count"].clamp_min(1.0).unsqueeze(-1)
    return (
        n_choices
        * ((stats["oh_sum"] / count) * (stats["p_sum"] / count)).sum(-1)
    ).mean().detach()




def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1





def forward_loss(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    doc_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cfg,
    runtime: RouterRuntime,
    depth: int | None,
    pool: RouterAuxPool,
    backprop_depth: int | None = None,
    token_loss_mask: torch.Tensor | None = None,
    ce_denominator: torch.Tensor | None = None,
    input_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total loss, auxiliary metric, and cross-entropy.

    ``token_loss_mask`` narrows the supervised targets further than the
    model's own mask, which is how a dataset restricts the loss to part of a
    sequence. It never widens it. ``ce_denominator`` lets the trainer
    normalize those masked losses over every token in the optimizer step.
    """

    kwargs = recurrence_kwargs(depth, backprop_depth)
    if input_valid is not None:
        kwargs["input_valid"] = input_valid
    if backprop_depth is not None and getattr(
        runtime, "first_window_full_backprop", False
    ):
        kwargs["first_window_full_backprop"] = True

    raw_model = model.module if hasattr(model, "module") else model
    uses_ffn_moe = bool(getattr(raw_model, "uses_ffn_moe", False))
    has_outer_router = bool(cfg.model.routed)
    # The chunked path folds the cross entropy into the model, where only the
    # model's own mask is known. A section mask therefore falls back to
    # explicit logits so both masks combine in one place. That is cheap here:
    # the datasets that restrict sections have a small vocabulary.
    uses_chunked_ce = (
        bool(getattr(raw_model, "supports_chunked_linear_cross_entropy", False))
        and token_loss_mask is None
    )
    if uses_chunked_ce:
        kwargs["targets"] = targets
    lm_output, aux_loss, loss_mask, stats = model(
        inputs,
        doc_id=doc_ids,
        position_ids=position_ids,
        return_router_stats=True,
        return_router_diagnostics=False,
        **kwargs,
    )

    if uses_ffn_moe:
        _pool_ffn_stats(pool, stats)
        ffn_aux = stats["ffn_aux_loss"]
    else:
        ffn_aux = aux_loss * 0.0

    # Models without a mask of their own (the vanilla shell with FFN MoE, the
    # recurrent-depth middle) supervise every valid target.
    valid_targets = targets != -100
    loss_mask = (
        valid_targets if loss_mask is None else loss_mask & valid_targets
    )
    if token_loss_mask is not None:
        loss_mask = loss_mask & token_loss_mask

    if uses_chunked_ce:
        ce_loss = lm_output
    else:
        per_token = F.cross_entropy(
            lm_output.view(-1, lm_output.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view_as(targets)
        denominator = (
            loss_mask.float().sum().clamp_min(1.0)
            if ce_denominator is None else ce_denominator
        )
        ce_loss = (per_token * loss_mask.float()).sum() / denominator

    if runtime.load_balancing.is_loss_free:
        sequence_aux = ce_loss * 0.0
        if has_outer_router:
            _pool_router_stats(pool, stats, include_probs=False)
            sequence_aux = stats["sequence_aux_loss"]
        return (
            ce_loss + runtime.load_balancing.sequence_aux_coeff * sequence_aux,
            sequence_aux,
            ce_loss,
        )

    ffn_surrogate = (
        _ffn_global_surrogate(stats, cfg, runtime)
        if uses_ffn_moe else ce_loss * 0.0
    )
    if not has_outer_router:
        return (
            ce_loss + cfg.model.moe_ffn_aux_loss_coeff * ffn_surrogate,
            ffn_aux,
            ce_loss,
        )

    gradient_stats = _gradient_depth_stats(stats)
    _pool_router_stats(pool, stats, include_probs=True)
    gradient_count = gradient_stats["count"].detach()
    pool.gradient_count = (
        gradient_count if pool.gradient_count is None
        else pool.gradient_count + gradient_count
    )
    n_choices = raw_model.n_experts
    surrogate = _outer_global_surrogate(
        gradient_stats, cfg, runtime, n_choices
    )
    total = ce_loss + cfg.model.aux_loss_coeff * surrogate
    if uses_ffn_moe:
        total = (
            total
            + cfg.model.moe_ffn_aux_loss_coeff * ffn_surrogate
        )
    return total, aux_loss + ffn_aux, ce_loss


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor




def _finish_ffn_step(
    model, runtime: RouterRuntime, pool: RouterAuxPool, step: int,
) -> torch.Tensor | None:
    if pool.ffn_load_sum is None or pool.ffn_importance_sum is None:
        return None
    loads = pool.ffn_load_sum
    importance = pool.ffn_importance_sum
    global_load = loads.clone()
    global_importance = importance.clone()
    _all_reduce_sum(global_load)
    _all_reduce_sum(global_importance)
    global_active = global_load.sum(-1) > 0
    global_load_fraction = global_load / global_load.sum(
        -1, keepdim=True
    ).clamp_min(1.0)
    global_importance_fraction = global_importance / global_importance.sum(
        -1, keepdim=True
    ).clamp_min(1e-12)
    aux = global_load.size(-1) * (
        global_load_fraction * global_importance_fraction
    ).sum(-1)
    runtime.ffn_auxiliary_loss = aux[global_active].mean().detach()
    mean_load = global_load.mean(-1, keepdim=True)
    violation = (
        (global_load.max(-1).values - mean_load.squeeze(-1))
        / mean_load.squeeze(-1).clamp_min(1.0)
    )
    if global_active.any():
        runtime.ffn_load_max_violation = (
            violation[global_active].max().detach()
        )

    raw_model = model.module if hasattr(model, "module") else model
    modules = raw_model._ffn_moe_modules()
    if runtime.load_balancing.is_loss_free:
        if runtime.bias_update_active(step):
            signs = torch.sign(global_load - mean_load)
            updates = signs - signs.mean(-1, keepdim=True)
            with torch.no_grad():
                for index, module in enumerate(modules):
                    if global_active[index]:
                        module.expert_bias.sub_(
                            runtime.load_balancing.bias_update_speed
                            * updates[index].to(module.expert_bias)
                        ).clamp_(-16.0, 16.0)
        runtime.ffn_expert_bias_abs_max = torch.stack([
            module.expert_bias.abs().max() for module in modules
        ]).max().detach()
    else:
        runtime.ffn_f_hat = global_load / global_load.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        runtime.ffn_n_hat = (
            global_importance.sum(-1).clamp_min(1.0)
            / _distributed_world_size()
        )
    return runtime.ffn_auxiliary_loss


def finish_aux_step(
    model, runtime: RouterRuntime, pool: RouterAuxPool,
    micro_aux: torch.Tensor, step: int,
) -> torch.Tensor:
    """Finalize auxiliary metrics and all routing-balance state."""

    ffn_aux = _finish_ffn_step(model, runtime, pool, step)
    raw_model = model.module if hasattr(model, "module") else model
    has_outer_router = hasattr(raw_model, "expert_bias")

    if runtime.load_balancing.is_loss_free:
        if has_outer_router:
            if pool.oh_sum is None:
                raise RuntimeError(
                    "Loss-free balancing did not collect assignments"
                )
            per_depth_load = pool.oh_sum.clone()
            _all_reduce_sum(per_depth_load)
            global_load = per_depth_load.sum(dim=0)
            mean_load = global_load.mean()
            global_signs = torch.sign(global_load - mean_load)
            global_update = global_signs - global_signs.mean()
            if runtime.bias_update_active(step):
                with torch.no_grad():
                    raw_model.expert_bias.sub_(
                        runtime.load_balancing.bias_update_speed
                        * global_update.to(raw_model.expert_bias)
                    ).clamp_(-16.0, 16.0)
            runtime.load_max_violation = (
                (global_load.max() - mean_load)
                / mean_load.clamp_min(1.0)
            ).detach()
            runtime.expert_bias_abs_max = raw_model.expert_bias.abs().max().detach()
        return micro_aux

    outer_aux = None
    if has_outer_router:
        if pool.oh_sum is None or pool.p_sum is None or pool.count is None:
            raise RuntimeError(
                "Global load balancing did not collect router statistics"
            )
        n_choices = raw_model.n_experts
        pooled_stats = {
            "oh_sum": pool.oh_sum,
            "p_sum": pool.p_sum,
            "count": pool.count,
        }
        outer_aux = _outer_aux_metric(pooled_stats, n_choices)
        global_oh = pool.oh_sum.clone()
        global_count = pool.count.clone()
        _all_reduce_sum(global_oh)
        _all_reduce_sum(global_count)
        global_gradient_count = (
            pool.gradient_count.clone()
            if pool.gradient_count is not None else global_count
        )
        if pool.gradient_count is not None:
            _all_reduce_sum(global_gradient_count)
        runtime.f_hat = (
            global_oh / global_count.clamp_min(1.0).unsqueeze(-1)
        )
        runtime.n_hat = (
            global_gradient_count.clamp_min(1.0)
            / _distributed_world_size()
        )

    if outer_aux is None:
        if ffn_aux is None:
            raise RuntimeError("No routing statistics were collected")
        return ffn_aux
    return outer_aux if ffn_aux is None else outer_aux + ffn_aux


def validation_forward(
    model,
    inputs: torch.Tensor,
    doc_ids: torch.Tensor,
    position_ids: torch.Tensor,
    depth: int | None,
    *,
    return_router_stats: bool = False,
    return_router_trace: bool = False,
    input_valid=None,
):
    return_router_stats = return_router_stats or return_router_trace
    kwargs = {"num_steps": depth} if depth is not None else {}
    return model(
        inputs,
        doc_id=doc_ids,
        position_ids=position_ids,
        **({"input_valid": input_valid} if input_valid is not None else {}),
        return_router_stats=return_router_stats,
        return_router_diagnostics=return_router_stats,
        return_router_trace=return_router_trace,
        **kwargs,
    )


def window_routing_rows(
    token_paths, depth0_window_ids, doc_ids,
    *, input_valid=None,
):
    """Collapse per-token expert paths onto the depth-zero window axis.

    Shifted-window depths do not share a window layout, so ``routes[w, d]``
    is the block chosen by the depth-``d`` window that owns window ``w``'s
    first token. Returns ``(routes [W, D], n_tokens [W], doc_of_window [W],
    valid [W])``; ``valid`` drops empty windows. Every retained first window
    is router-owned and participates in the sample. For top-k traces, every
    rank becomes a separate route row and the window metadata is repeated.
    """

    token_paths = torch.as_tensor(token_paths)
    paths = token_paths.flatten(0, 1)
    window_of_token = torch.as_tensor(depth0_window_ids).reshape(-1)
    total = int(window_of_token.max().item()) + 1
    counted_windows = (
        window_of_token if input_valid is None else
        window_of_token[input_valid.reshape(-1)]
    )
    tokens = torch.bincount(counted_windows, minlength=total)
    order = torch.arange(
        window_of_token.numel(), device=window_of_token.device
    )
    first_token = torch.full_like(tokens, window_of_token.numel())
    first_token.scatter_reduce_(
        0, window_of_token, order, reduce="amin", include_self=True
    )
    first_token.clamp_(max=window_of_token.numel() - 1)

    sequences, length = doc_ids.shape
    global_doc_id = (
        torch.arange(sequences, device=doc_ids.device).unsqueeze(-1)
        * (length + 1)
        + doc_ids
    ).reshape(-1)
    valid = tokens > 0
    routes = paths[first_token]
    documents = global_doc_id[first_token]
    if routes.ndim == 3:
        top_k = routes.size(-1)
        routes = routes.permute(0, 2, 1).reshape(
            routes.size(0) * top_k, routes.size(1)
        )
        tokens = tokens.repeat_interleave(top_k)
        documents = documents.repeat_interleave(top_k)
        valid = valid.repeat_interleave(top_k)
    return (
        routes,
        tokens,
        documents,
        valid,
    )


def accumulate_captured_routing_trace(
    accumulator,
    lang_index,
    stats,
    doc_ids,
    *,
    input_valid=None,
    validation_name="validation",
):
    """Add captured routing rows to one language's accumulator."""

    try:
        routes, tokens, documents, valid = window_routing_rows(
            stats["routing_token_paths"],
            stats["routing_depth0_window_ids"],
            doc_ids,
            input_valid=input_valid,
        )
        # Windows are in token order, so a document's first window is where
        # the document id changes.
        first_of_doc = torch.ones_like(documents, dtype=torch.bool)
        first_of_doc[1:] = documents[1:] != documents[:-1]
        accumulator.update(
            routes,
            torch.full_like(tokens, int(lang_index)),
            tokens,
            doc_ids=documents,
            valid=valid,
            ordinals=LoLWindow._win_in_doc_idx(
                first_of_doc, documents.numel(), documents.device
            ),
        )
    except Exception as error:
        print(
            "WARNING: routing sample for "
            f"{validation_name} failed "
            f"({type(error).__name__}: {error}); continuing."
        )


@dataclass(frozen=True)
class RouterPlotData:
    n_experts: int
    actual_utilization: list[list[float]]
    token_ids: list[int]
    token_paths: list[list[int]]
    transition_counts: list[list[list[int]]]


@dataclass(frozen=True)
class RouterValidationDiagnostics:
    metrics: dict[str, float]
    plot_data: RouterPlotData | FFNRouterPlotData


@dataclass(frozen=True)
class FFNRouterPlotData:
    n_experts: int
    assignment_utilization: list[list[float]]
    importance_share: list[list[float]]
    # None when no single-pass token trace exists, as in a recurrent middle
    # whose FFN sites run once per recurrence.
    token_ids: list[int] | None
    token_paths: list[list[int]] | None
    transition_counts: list[list[list[int]]] | None


def _load_coefficient_of_variation(utilization: torch.Tensor) -> torch.Tensor:
    mean = utilization.mean(dim=-1)
    return utilization.std(dim=-1, unbiased=False) / mean.clamp_min(1e-12)


def summarize_validation_diagnostics(
    stats: dict[str, torch.Tensor],
) -> tuple[dict[str, float], torch.Tensor]:
    """Summarize full-depth validation routing."""

    dispatch_sum = stats["oh_sum"].float()
    dispatch_count = stats["count"].float()
    actual_utilization = (
        dispatch_sum / dispatch_count.clamp_min(1.0).unsqueeze(-1)
    )
    actual_layer_cv = _load_coefficient_of_variation(actual_utilization)

    actual_global = dispatch_sum.sum(dim=0) / dispatch_count.sum().clamp_min(1.0)
    actual_global_cv = _load_coefficient_of_variation(
        actual_global.unsqueeze(0)
    ).squeeze(0)
    actual_violation = (
        actual_utilization.max(dim=-1).values
        - actual_utilization.mean(dim=-1)
    ) / actual_utilization.mean(dim=-1).clamp_min(1e-12)
    decision_count = dispatch_count.sum().clamp_min(1.0)
    mean_entropy = stats["router_entropy_sum"].float().sum() / decision_count
    mean_confidence = (
        stats["router_confidence_sum"].float().sum() / decision_count
    )
    experts = actual_utilization.size(-1)
    normalized_entropy = (
        mean_entropy / math.log(experts)
        if experts > 1
        else mean_entropy * 0.0
    )
    metrics = {
        "mean_router_entropy": float(mean_entropy.item()),
        "normalized_router_entropy": float(normalized_entropy.item()),
        "mean_top1_probability": float(mean_confidence.item()),
        "active_expert_fraction": float(
            (dispatch_sum.sum(dim=0) > 0).float().mean().item()
        ),
        "unused_layer_expert_fraction": float(
            (dispatch_sum == 0).float().mean().item()
        ),
        "load_coefficient_of_variation": float(
            actual_layer_cv.mean().item()
        ),
        "max_layer_expert_utilization": float(
            actual_utilization.max().item()
        ),
        "actual_global_load_coefficient_of_variation": float(
            actual_global_cv.item()
        ),
        "actual_max_layer_load_violation": float(
            actual_violation.max().item()
        ),
    }
    return metrics, actual_utilization


def summarize_ffn_validation_diagnostics(
    loads: torch.Tensor,
    importance: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Summarize token-level FFN-MoE assignments and gate importance."""

    loads = loads.float()
    importance = importance.float()
    utilization = loads / loads.sum(dim=-1, keepdim=True).clamp_min(1.0)
    importance_share = (
        importance
        / importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    )
    layer_cv = _load_coefficient_of_variation(utilization)
    importance_cv = _load_coefficient_of_variation(importance_share)
    metrics = {
        "active_expert_fraction": float(
            (loads.sum(dim=0) > 0).float().mean().item()
        ),
        "unused_layer_expert_fraction": float(
            (loads == 0).float().mean().item()
        ),
        "load_coefficient_of_variation": float(layer_cv.mean().item()),
        "max_layer_expert_utilization": float(utilization.max().item()),
        "importance_load_coefficient_of_variation": float(
            importance_cv.mean().item()
        ),
        "max_layer_expert_importance": float(
            importance_share.max().item()
        ),
    }
    metrics.update({
        f"layer_{index}_load_coefficient_of_variation": float(value.item())
        for index, value in enumerate(layer_cv)
    })
    metrics.update({
        f"layer_{index}_importance_coefficient_of_variation": float(
            value.item()
        )
        for index, value in enumerate(importance_cv)
    })
    return metrics, utilization, importance_share


def _route_path_plot_data(
    paths: torch.Tensor,
    n_experts: int,
    path_ids: torch.Tensor,
) -> tuple[list[int], list[list[int]], list[list[list[int]]]]:
    """Sample scalar route paths and count adjacent-site transitions."""

    paths = paths.detach().long().cpu()
    path_ids = path_ids.detach().long().cpu()
    valid_paths = (paths[0] >= 0).all(dim=-1)
    available_ids = path_ids[valid_paths].unique(sorted=True)
    sample_count = min(32, available_ids.numel())
    if sample_count:
        selected = torch.linspace(
            0, available_ids.numel() - 1, steps=sample_count
        ).round().long().unique()
        sampled_ids = available_ids[selected]
        sampled = valid_paths & (
            path_ids.unsqueeze(-1) == sampled_ids
        ).any(dim=-1)
        token_paths = paths[0, sampled].tolist()
        token_ids = path_ids[sampled].tolist()
    else:
        token_paths = []
        token_ids = []

    transition_counts = []
    for site in range(paths.size(-1) - 1):
        source = paths[..., site]
        destination = paths[..., site + 1]
        valid = (source >= 0) & (destination >= 0)
        encoded = source[valid] * n_experts + destination[valid]
        transition_counts.append(
            torch.bincount(
                encoded, minlength=n_experts * n_experts
            ).reshape(n_experts, n_experts).tolist()
        )
    return token_ids, token_paths, transition_counts


def _route_change_rate(paths: torch.Tensor) -> float:
    if paths.size(-1) <= 1:
        return 0.0
    source = paths[..., :-1]
    destination = paths[..., 1:]
    valid = (source >= 0) & (destination >= 0)
    return (
        float((source[valid] != destination[valid]).float().mean().item())
        if valid.any()
        else 0.0
    )


def build_ffn_validation_diagnostics(
    diagnostic_totals: dict[str, torch.Tensor] | None,
    paths: torch.Tensor | None,
    *,
    n_experts: int,
) -> RouterValidationDiagnostics:
    """Build FFN-MoE validation metrics and top-k route plot data."""

    if diagnostic_totals is None:
        raise RuntimeError("FFN-MoE validation did not collect diagnostics")
    metrics, utilization, importance_share = (
        summarize_ffn_validation_diagnostics(
            diagnostic_totals["ffn_load_sum"],
            diagnostic_totals["ffn_importance_sum"],
        )
    )
    token_ids = token_paths = transition_counts = None
    if paths is not None:
        paths = torch.as_tensor(paths)
        ranked_paths = paths.permute(0, 1, 3, 2).reshape(
            paths.size(0), -1, paths.size(2)
        )
        path_ids = torch.arange(paths.size(1)).repeat_interleave(
            paths.size(3)
        )
        metrics["route_change_rate"] = _route_change_rate(ranked_paths)
        token_ids, token_paths, transition_counts = _route_path_plot_data(
            ranked_paths, n_experts, path_ids
        )
    return RouterValidationDiagnostics(
        metrics=metrics,
        plot_data=FFNRouterPlotData(
            n_experts=n_experts,
            assignment_utilization=(
                utilization.detach().float().cpu().tolist()
            ),
            importance_share=(
                importance_share.detach().float().cpu().tolist()
            ),
            token_ids=token_ids,
            token_paths=token_paths,
            transition_counts=transition_counts,
        ),
    )


def build_validation_diagnostics(
    diagnostic_totals: dict[str, torch.Tensor] | None,
    paths: torch.Tensor | None,
    *,
    n_experts: int,
) -> RouterValidationDiagnostics:
    """Build routed validation metrics and interactive plot data."""

    if diagnostic_totals is None or paths is None:
        raise RuntimeError("Routed validation did not collect diagnostics")
    paths = torch.as_tensor(paths)
    metrics, actual_utilization = summarize_validation_diagnostics(diagnostic_totals)
    path_ids = torch.arange(paths.size(1))
    if paths.ndim == 4:
        path_ids = path_ids.repeat_interleave(paths.size(3))
        paths = paths.permute(0, 1, 3, 2).reshape(paths.size(0), -1, paths.size(2))
    metrics["route_change_rate"] = _route_change_rate(paths)
    token_ids, token_paths, transition_counts = _route_path_plot_data(
        paths, n_experts, path_ids
    )
    return RouterValidationDiagnostics(
        metrics=metrics,
        plot_data=RouterPlotData(
            n_experts=n_experts,
            actual_utilization=actual_utilization.detach().float().cpu().tolist(),
            token_ids=token_ids,
            token_paths=token_paths,
            transition_counts=transition_counts,
        ),
    )
