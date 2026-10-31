"""Unified multi-GPU trainer for vanilla and routed language models.

Examples:
    torchrun --nproc_per_node=8 train.py arch=vanilla_neo
    torchrun --nproc_per_node=8 train.py arch=lol_dynamic_window
"""

from __future__ import annotations

import math
import os
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import hydra
from omegaconf import DictConfig
import torch
import torch.distributed as dist
from torch._dynamo import config as dynamo_config
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from data import sudoku_sections
from data.dataloader_multiproc import DataLoaderLit_MultiProc
from utils.flop_counter import (
    build_runtime as build_flop_runtime,
    training_step_flops,
)
from model.factory import build_model
from optimizer import build_optimizer, restore_lr_bounds
import router
from utils.configuration import (
    clear_fresh_checkpoints,
    config_to_dict,
    ddp_find_unused_parameters,
    initialize_distributed,
    resolve_config,
    validate_training_entrypoint,
)
from utils.routing_logger import (
    RoutingAccumulator,
    RoutingLogger,
    RoutingLoggerConfig,
    active_languages,
)
from utils.runtime_state import (
    capture_rank_runtime_state,
    config_fingerprint,
    gather_rank_runtime_states,
    restore_rank_runtime_state,
)
from utils.wandb_plots import ffn_router_wandb_plots, router_wandb_plots
from utils.validation_utils import (
    reset_validation_loader,
    dataset_section_masks,
    target_byte_count,
    token_nll_sum,
    unpack_validation_outputs,
    validation_batch_count,
)


SUPPORTED_ARCHES = frozenset({
    "vanilla_neo", "lol_dynamic_window", "huginn",
})


def loss_sections_arm(cfg):
    """Return the configured strategy-corpus loss arm, or ``None`` when unset.

    Configs assembled by hand for a single training or validation concern
    carry no ``data`` block at all, which reads the same as leaving the arm
    unset: train and score on every target.
    """

    data_cfg = getattr(cfg, "data", None)
    return None if data_cfg is None else data_cfg.get("loss_sections")


def section_masks(cfg, inputs, targets, doc_ids, position_ids):
    """Return (loss mask, per-metric masks) for the configured sections.

    Both are ``None`` unless ``data.loss_sections`` selects an arm of the
    sudoku or maze layout (``data.dataset``), so every other dataset keeps
    training and scoring on every target.
    """

    arm = loss_sections_arm(cfg)
    if arm is None:
        return None, None
    return dataset_section_masks(
        cfg.data.get("dataset"), arm, cfg.data.get("tag_span"),
        inputs, targets, doc_ids, position_ids,
    )


@dataclass(frozen=True)
class StepResult:
    loss: torch.Tensor
    ce_loss: torch.Tensor
    aux_loss: torch.Tensor
    grad_norm: torch.Tensor
    main_lr: float
    adamw_lr: float | None
    router_lr: float | None
    step_flops: int
    recurrence_sample: router.RecurrenceSample | None
    flop_depth_measured: bool

    @property
    def depth(self) -> int | None:
        if self.recurrence_sample is not None:
            return self.recurrence_sample.total_depth
        return None


def distributed_step_metrics(
    result: StepResult, world_size: int,
) -> torch.Tensor:
    """Average detached step metrics across DDP ranks for logging."""
    metrics = torch.stack(
        (result.loss, result.ce_loss, result.aux_loss, result.grad_norm)
    )
    dist.reduce(metrics, dst=0, op=dist.ReduceOp.SUM)
    return metrics / int(world_size)


@dataclass(frozen=True)
class EvaluationResult:
    nll: float
    full_depth_nll: float
    depth_nlls: dict[int, float] = field(default_factory=dict)
    validation_name: str = "val"
    router_namespace: str = "router"
    byte_nll: float | None = None
    bits_per_byte: float | None = None
    full_depth_byte_nll: float | None = None
    full_depth_bits_per_byte: float | None = None
    log_full_depth: bool = False
    # Teacher-forced per-token accuracy per sequence section, when configured.
    section_accuracies: dict[str, float] | None = None
    router_metrics: dict[str, float] | None = None
    router_plots: dict[str, object] | None = None


@dataclass
class TrainingState:
    resume_step: int = 0
    wandb_run_id: str | None = None
    runtime_state_by_rank: list[dict] | None = None


def learning_rate(step: int, cfg, maximum: float, minimum: float) -> float:
    """Warm up linearly, then cosine decay between the supplied bounds."""

    warmup_steps = int(cfg.training.warmup_steps)
    max_steps = int(cfg.training.max_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return maximum * (step + 1) / warmup_steps
    if step >= max_steps or max_steps <= warmup_steps:
        return minimum
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return minimum + coefficient * (maximum - minimum)


def initialize_wandb(cfg, world_size: int, is_master: bool, run_id):
    if is_master:
        resolved = config_to_dict(cfg)
        kwargs = {
            "project": cfg.wandb_project,
            "name": cfg.wandb_run_name,
            "config": {
                "architecture": cfg.model.arch,
                **resolved["model"],
                "world_size": world_size,
            },
            "resume": "allow",
        }
        if run_id is not None:
            kwargs["id"] = run_id
        try:
            wandb.init(**kwargs)
        except Exception as error:
            print(
                f"WARNING: wandb.init failed with stored id {run_id!r} "
                f"({type(error).__name__}: {error}). Starting a new run."
            )
            kwargs.pop("id", None)
            wandb.init(**kwargs)
        run_id = wandb.run.id
        wandb.define_metric("global_step", hidden=True)
        for namespace in ("train", "eval", "router", "router_ffn"):
            wandb.define_metric(
                f"{namespace}/*", step_metric="global_step"
            )

    run_ids = [run_id] if is_master else [None]
    dist.broadcast_object_list(run_ids, src=0)
    return run_ids[0]


def setup_model_and_optimizer(
    model,
    cfg,
    router_runtime,
    local_rank: int,
    is_master: bool,
):
    """Move/compile/wrap the model and construct architecture-aware groups."""

    routed = router_runtime is not None and router_runtime.uses_router
    torch.set_float32_matmul_precision("high")
    model.to(cfg.device)
    if cfg.training.compile:
        if is_master:
            print("Compiling model blocks (this can take about a minute)...")
        model.compile_blocks()

    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=(
            ddp_find_unused_parameters(
                model,
                cfg.model,
                recurrence_backprop_depth=(
                    None if router_runtime is None
                    else router_runtime.recurrence_backprop_depth
                ),
            )
        ),
    )
    if not routed:
        return model, build_optimizer(model, cfg.training)

    param_groups, router_params = router.optimizer_param_groups(
        model, cfg, router_runtime
    )
    optimizer = build_optimizer(model, cfg.training, param_groups)
    if is_master:
        print(
            f"Decoupled router LR: peak={router_runtime.max_lr:.2e} "
            f"floor={router_runtime.min_lr:.2e} "
            f"({sum(p.numel() for p in router_params) / 1e6:.3f}M "
            f"router params) | main peak={cfg.training.max_lr:.2e} "
            f"floor={cfg.training.min_lr:.2e}"
        )
    return model, optimizer


def load_checkpoint(
    path,
    model,
    optimizer,
    train_loader,
    val_loader,
    cfg,
    router_runtime,
    flop_runtime,
    rank: int,
    world_size: int,
    is_master: bool,
) -> TrainingState:
    """Resume the latest trajectory, or start a new run."""

    if not os.path.exists(path):
        return TrainingState()
    if is_master:
        print(f"Resuming from {path} ...")
    checkpoint = torch.load(
        path, map_location=cfg.device, weights_only=False
    )
    saved_world_size = int(checkpoint["ddp_world_size"])
    if saved_world_size != world_size:
        raise RuntimeError(
            "Checkpoint world size mismatch: "
            f"saved={saved_world_size}, current={world_size}."
        )

    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_bounds = {
        "main": (
            float(cfg.training.max_lr),
            float(cfg.training.min_lr),
        )
    }
    if router_runtime is not None:
        lr_bounds["router"] = (
            router_runtime.max_lr,
            router_runtime.min_lr,
        )
    flop_runtime.restore_checkpoint(checkpoint)
    restore_lr_bounds(optimizer, lr_bounds)

    run_id = checkpoint["wandb_run_id"]
    saved_config = checkpoint.get("config", {})
    saved_project = saved_config.get(
        "wandb_project",
        saved_config.get("logging", {}).get("wandb_project"),
    )
    if (
        run_id is not None
        and saved_project is not None
        and saved_project != cfg.wandb_project
    ):
        if is_master:
            print(
                f"wandb project changed ({saved_project} -> "
                f"{cfg.wandb_project}); starting a new run."
            )
        run_id = None

    train_loader.current_position = int(
        checkpoint["train_loader_pos_by_rank"][rank]
    )
    val_loader.current_position = int(
        checkpoint["val_loader_pos_by_rank"][rank]
    )
    state = TrainingState(
        resume_step=int(checkpoint["step"]) + 1,
        wandb_run_id=run_id,
        runtime_state_by_rank=checkpoint["runtime_state_by_rank"],
    )
    if is_master:
        print(f"Resumed at step {state.resume_step}")
    return state


def save_latest_checkpoint(
    path,
    step: int,
    model,
    optimizer,
    train_loader,
    val_loader,
    cfg,
    router_runtime,
    flop_runtime,
    state: TrainingState,
    world_size: int,
    is_master: bool,
) -> None:
    """Collect every rank's state and atomically write the latest checkpoint."""

    def gather_position(loader):
        local = torch.tensor(
            [int(loader.current_position)],
            device=cfg.device,
            dtype=torch.long,
        )
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        return [int(position.item()) for position in gathered]

    train_positions = gather_position(train_loader)
    val_positions = gather_position(val_loader)
    trainer_tensors = (
        router_runtime.runtime_tensors()
        if router_runtime is not None
        else None
    )
    local_runtime = capture_rank_runtime_state(
        cfg.device, trainer_tensors
    )
    runtime_states = gather_rank_runtime_states(
        local_runtime, world_size
    )
    if not is_master:
        return

    checkpoint = {
        "step": step,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "wandb_run_id": state.wandb_run_id,
        "ddp_world_size": world_size,
        "runtime_state_by_rank": runtime_states,
        "train_loader_pos_by_rank": train_positions,
        "val_loader_pos_by_rank": val_positions,
        "config": config_to_dict(cfg),
    }
    checkpoint.update(flop_runtime.checkpoint_fields())
    temporary_path = path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)
    print(f"Saved latest checkpoint at step {step} -> {path}")


def run_optimizer_step(
    model,
    optimizer,
    train_loader,
    cfg,
    router_runtime,
    flop_runtime,
    step: int,
) -> StepResult:
    """Run one optimizer step without synchronizing metrics to the host."""

    routed = router_runtime is not None and router_runtime.uses_router
    model.train()
    rates = {}
    for group in optimizer.param_groups:
        rate = learning_rate(
            step,
            cfg,
            float(group["_max_lr"]),
            float(group["_min_lr"]),
        )
        group["lr"] = rate
        name = group["_lr_group"]
        if group.get("_optimizer_component") == "adamw" and name == "main":
            name = "adamw"
        rates[name] = rate
    optimizer.zero_grad(set_to_none=True)
    recurrence_sample = (
        router_runtime.sample_depth(step)
        if router_runtime is not None
        else None
    )
    depth = (
        recurrence_sample.total_depth
        if recurrence_sample is not None else None
    )
    zero = torch.zeros((), device=cfg.device, dtype=torch.float32)
    loss_accum = zero
    ce_accum = zero
    aux_accum = zero
    aux_pool = router.RouterAuxPool() if routed else None
    gradient_depth = (
        recurrence_sample.gradient_depth
        if recurrence_sample is not None
        and router_runtime.recurrence_backprop_depth is not None else None
    )
    measure_flops = flop_runtime.needs_measurement(depth, gradient_depth)
    local_step_flops = (
        training_step_flops(
            cfg, depth, gradient_depth=gradient_depth
        )
        if measure_flops else 0
    )

    masked_batches = None
    ce_denominator = None
    if loss_sections_arm(cfg) is not None:
        masked_batches = []
        local_token_count = 0
        for _ in range(cfg.training.grad_accum_steps):
            batch = (
                train_loader.get_batch_with_meta("cpu", include_input_valid=True)
                if getattr(train_loader, "validity", None) is not None
                else (*train_loader.get_batch_with_meta("cpu"), None)
            )
            token_loss_mask = section_masks(cfg, *batch[:4])[0]
            masked_batches.append((*batch, token_loss_mask))
            local_token_count += int(token_loss_mask.sum())
        global_token_count = torch.tensor(
            float(local_token_count), device=cfg.device
        )
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        # DDP averages gradients across ranks and the loop below averages
        # across microbatches. Dividing each CE sum by this value makes those
        # two averages equal one global supervised-token mean. Clamp the raw
        # global count so a fully unsupervised step stays differentiable zero.
        ce_denominator = global_token_count.clamp_min(1.0) / (
            world_size * cfg.training.grad_accum_steps
        )

    for micro_step in range(cfg.training.grad_accum_steps):
        if masked_batches is None:
            inputs, targets, doc_ids, position_ids, input_valid = (
                train_loader.get_batch_with_meta(
                    cfg.device, include_input_valid=True
                ) if getattr(train_loader, "validity", None) is not None
                else (*train_loader.get_batch_with_meta(cfg.device), None)
            )
            token_loss_mask = None
        else:
            (
                inputs,
                targets,
                doc_ids,
                position_ids,
                input_valid,
                token_loss_mask,
            ) = DataLoaderLit_MultiProc._to_device(
                masked_batches[micro_step], cfg.device
            )
        last_micro_step = micro_step == cfg.training.grad_accum_steps - 1
        sync_context = nullcontext if last_micro_step else model.no_sync
        with sync_context():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if routed:
                    loss, raw_aux, raw_ce = router.forward_loss(
                        model,
                        inputs,
                        targets,
                        doc_ids,
                        position_ids,
                        cfg,
                        router_runtime,
                        depth,
                        aux_pool,
                        backprop_depth=gradient_depth,
                        token_loss_mask=token_loss_mask,
                        ce_denominator=ce_denominator,
                        input_valid=input_valid,
                    )
                else:
                    loss = model(
                        inputs,
                        doc_id=doc_ids,
                        position_ids=position_ids,
                        targets=targets,
                        **({"input_valid": input_valid} if input_valid is not None else {}),
                        **router.recurrence_kwargs(depth, gradient_depth),
                    )
                    raw_aux = zero
                    raw_ce = loss
            loss = loss / cfg.training.grad_accum_steps
            loss.backward()
        loss_accum = loss_accum + loss.detach()
        ce_accum = (
            ce_accum
            + raw_ce.detach() / cfg.training.grad_accum_steps
        )
        if routed:
            aux_accum = (
                aux_accum
                + raw_aux.detach() / cfg.training.grad_accum_steps
            )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        (
            cfg.training.grad_clip
            if cfg.training.grad_clip > 0.0
            else float("inf")
        ),
    ).detach()
    optimizer.step()
    if routed:
        aux_accum = router.finish_aux_step(
            model, router_runtime, aux_pool, aux_accum, step
        )
    step_flops, flop_depth_measured = flop_runtime.record_step(
        depth,
        local_step_flops if measure_flops else None,
        cfg.device,
        gradient_depth=gradient_depth,
    )

    return StepResult(
        loss=loss_accum,
        ce_loss=ce_accum,
        aux_loss=aux_accum,
        grad_norm=grad_norm,
        main_lr=rates["main"],
        adamw_lr=rates.get("adamw"),
        router_lr=rates.get("router"),
        step_flops=step_flops,
        recurrence_sample=recurrence_sample,
        flop_depth_measured=flop_depth_measured,
    )


def evaluation_log_fields(result: EvaluationResult) -> dict[str, object]:
    eval_prefix = (
        "eval" if result.validation_name == "val"
        else f"eval/{result.validation_name}"
    )
    router_prefix = (
        result.router_namespace if result.validation_name == "val"
        else f"{result.router_namespace}/{result.validation_name}"
    )
    fields: dict[str, object] = {
        f"{eval_prefix}/nll": result.nll,
        f"{eval_prefix}/perplexity": math.exp(result.nll),
    }
    for depth, nll in sorted(result.depth_nlls.items()):
        fields.update({
            f"{eval_prefix}/nll_depth_{depth}": nll,
            f"{eval_prefix}/perplexity_depth_{depth}": math.exp(nll),
        })
    if result.byte_nll is not None:
        fields.update({
            f"{eval_prefix}/byte_nll": result.byte_nll,
            f"{eval_prefix}/bits_per_byte": result.bits_per_byte,
        })
    if result.log_full_depth or result.full_depth_nll != result.nll:
        fields.update({
            f"{eval_prefix}/nll_full_depth": result.full_depth_nll,
            f"{eval_prefix}/perplexity_full_depth": math.exp(
                result.full_depth_nll
            ),
        })
        if result.full_depth_byte_nll is not None:
            fields.update({
                f"{eval_prefix}/byte_nll_full_depth": (
                    result.full_depth_byte_nll
                ),
                f"{eval_prefix}/bits_per_byte_full_depth": (
                    result.full_depth_bits_per_byte
                ),
            })
    if result.section_accuracies is not None:
        fields.update({
            f"{eval_prefix}/accuracy_{name}": value
            for name, value in result.section_accuracies.items()
        })
    if result.router_metrics is not None:
        fields.update({
            f"{router_prefix}/{key}": value
            for key, value in result.router_metrics.items()
        })
    if result.router_plots is not None:
        fields.update({
            f"{router_prefix}/{key}": value
            for key, value in result.router_plots.items()
        })
    return fields


def evaluate_validation_sets(
    model, validation_loaders, cfg, router_runtime, depth: int | None,
    routing_accumulator=None, routing_languages=(), routing_trace_batches=0,
) -> tuple[EvaluationResult, ...]:
    """Evaluate every configured validation split at one optimizer step."""

    return tuple(
        evaluate(
            model, val_loader, cfg, router_runtime, depth,
            validation_name=validation_name,
            routing_accumulator=routing_accumulator,
            routing_lang_index=(
                tuple(routing_languages).index(validation_name)
                if routing_accumulator is not None else None
            ),
            routing_trace_batches=routing_trace_batches,
        )
        for validation_name, val_loader in validation_loaders.items()
    )


def evaluate(
    model,
    val_loader,
    cfg,
    router_runtime,
    depth: int | None,
    validation_name: str = "val",
    routing_accumulator=None,
    routing_lang_index: int | None = None,
    routing_trace_batches: int = 0,
) -> EvaluationResult:
    """Compute validation loss and routing diagnostics for one named split."""

    outer_routed = bool(cfg.model.routed)
    uses_ffn_moe = bool(cfg.model.get("moe_ffn", False))
    is_master = not dist.is_initialized() or dist.get_rank() == 0
    model.eval()
    reset_validation_loader(val_loader)
    batch_count = validation_batch_count(
        val_loader, cfg.logging.eval_batches
    )
    # Both recurrent middles, routed or not, validate at every curriculum depth.
    recurrent = router_runtime is not None and router_runtime.recurrence_enabled
    depths = (
        router_runtime.validation_depths(depth)
        if recurrent else (None,)
    )
    current_depth_index = depths.index(depth)
    full_depth_index = (
        depths.index(router_runtime.recurrence_hard_cap)
        if recurrent
        else current_depth_index
    )
    diagnostic_depth_index = (
        full_depth_index if recurrent else len(depths) - 1
    )
    # Column 2 holds the UTF-8 byte count behind the same targets, and stays
    # zero unless the loader carries byte lengths. Columns 3 onwards hold
    # a (hits, targets) pair per section metric.
    scored_sections = loss_sections_arm(cfg) is not None
    metric_names = (
        sudoku_sections.layout_for(cfg.data.get("dataset")).metric_names
        if scored_sections else ()
    )
    totals = torch.zeros(
        (len(depths), 3 + 2 * len(metric_names)),
        device=cfg.device,
        dtype=torch.float64,
    )
    diagnostic_totals = None
    diagnostic_paths = None
    # Sample routing paths from the normal full-depth validation forward on
    # every rank, so the gathered reservoir covers the whole eval shard set.
    collect_routing = outer_routed and routing_accumulator is not None
    with (
        torch.no_grad(),
        dynamo_config.patch(optimize_ddp=False),
    ):
        for batch_index in range(batch_count):
            has_validity = getattr(val_loader, "validity", None) is not None
            if getattr(val_loader, "byte_lengths", None) is not None:
                inputs, targets, doc_ids, position_ids, byte_lengths, input_valid = (
                    val_loader.get_batch_with_byte_meta(
                        cfg.device, include_input_valid=True
                    ) if has_validity else
                    (*val_loader.get_batch_with_byte_meta(cfg.device), None)
                )
            else:
                inputs, targets, doc_ids, position_ids, input_valid = (
                    val_loader.get_batch_with_meta(
                        cfg.device, include_input_valid=True
                    ) if has_validity else
                    (*val_loader.get_batch_with_meta(cfg.device), None)
                )
                byte_lengths = None
            # Depth-independent, so build them once per batch rather than
            # once per evaluated depth.
            section_mask, section_metrics = section_masks(
                cfg, inputs, targets, doc_ids, position_ids
            )
            capture_outer_trace = outer_routed and (
                (is_master and batch_index == 0)
                or (collect_routing and batch_index < routing_trace_batches)
            )
            # Token paths need every FFN site to run once per forward, which
            # a recurrent middle does not do.
            capture_ffn_trace = (
                uses_ffn_moe
                and not outer_routed
                and str(cfg.model.arch) != "huginn"
                and is_master
                and batch_index == 0
            )
            for depth_index, evaluation_depth in enumerate(depths):
                collect_diagnostics = (
                    (outer_routed or uses_ffn_moe)
                    and depth_index == diagnostic_depth_index
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if outer_routed:
                        outputs = router.validation_forward(
                            model,
                            inputs,
                            doc_ids,
                            position_ids,
                            evaluation_depth,
                            return_router_stats=collect_diagnostics,
                            return_router_trace=(
                                collect_diagnostics and capture_outer_trace
                            ),
                            input_valid=input_valid,
                        )
                    else:
                        outputs = model(
                            inputs, doc_id=doc_ids,
                            position_ids=position_ids,
                            **({"input_valid": input_valid} if input_valid is not None else {}),
                            return_router_stats=collect_diagnostics,
                            return_router_trace=(
                                collect_diagnostics and capture_ffn_trace
                            ),
                            **router.recurrence_kwargs(evaluation_depth),
                        )
                    if collect_diagnostics:
                        stats = outputs[3]
                        outputs = outputs[:3]
                        if outer_routed and capture_outer_trace:
                            token_paths = stats["routing_token_paths"]
                            if (
                                collect_routing
                                and batch_index < routing_trace_batches
                            ):
                                router.accumulate_captured_routing_trace(
                                    routing_accumulator,
                                    routing_lang_index,
                                    stats,
                                    doc_ids,
                                    input_valid=input_valid,
                                    validation_name=validation_name,
                                )
                            if is_master and batch_index == 0:
                                diagnostic_paths = token_paths
                        elif capture_ffn_trace:
                            diagnostic_paths = stats[
                                "ffn_routing_token_paths"
                            ]
                        names = (
                            (
                                "oh_sum",
                                "router_entropy_sum",
                                "router_confidence_sum", "count",
                            )
                            if outer_routed
                            else ("ffn_load_sum", "ffn_importance_sum")
                        )
                        if diagnostic_totals is None:
                            diagnostic_totals = {
                                name: stats[name].detach().double().clone()
                                for name in names
                            }
                        else:
                            for name in names:
                                diagnostic_totals[name] += (
                                    stats[name].detach().double()
                                )
                    logits, loss_mask = unpack_validation_outputs(
                        outputs, targets, cfg.model.arch
                    )
                    correct = None
                    if section_mask is not None:
                        # Score exactly the targets training supervises.
                        loss_mask = loss_mask & section_mask
                        correct = logits.argmax(dim=-1) == targets
                    nll_sum, token_count = token_nll_sum(
                        logits, targets, loss_mask
                    )
                    byte_count = target_byte_count(
                        byte_lengths, loss_mask
                    )
                totals[depth_index, 0] += nll_sum.double()
                totals[depth_index, 1] += token_count.double()
                if byte_count is not None:
                    totals[depth_index, 2] += byte_count.double()
                for offset, name in enumerate(metric_names):
                    mask = section_metrics[name]
                    column = 3 + 2 * offset
                    totals[depth_index, column] += (
                        (correct & mask).sum().double()
                    )
                    totals[depth_index, column + 1] += mask.sum().double()

    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if diagnostic_totals is not None:
        for value in diagnostic_totals.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if (totals[:, 1] <= 0).any().item():
        raise RuntimeError("Validation produced no valid target tokens")
    nlls = totals[:, 0] / totals[:, 1]
    byte_nlls = None
    if (totals[:, 2] > 0).all().item():
        byte_nlls = totals[:, 0] / totals[:, 2]
    section_accuracies = {}
    for offset, name in enumerate(metric_names):
        column = 3 + 2 * offset
        target_count = totals[current_depth_index, column + 1]
        # A per-strategy split can have none of a token class at all: only
        # S2 and S3 name rows, columns and boxes.
        if target_count > 0:
            section_accuracies[name] = (
                totals[current_depth_index, column] / target_count
            ).item()
    if collect_routing:
        try:
            # ``oh_sum`` is the active-window dispatch histogram this
            # validation already reduced across ranks, so the whole-eval
            # per-depth counts cost nothing extra.
            routing_accumulator.add_reduced_counts(
                routing_lang_index, diagnostic_totals["oh_sum"]
            )
        except Exception as error:
            print(
                f"WARNING: routing diagnostics for {validation_name} failed "
                f"({type(error).__name__}: {error}); continuing."
            )
    router_metrics = None
    router_plots = None
    if outer_routed and is_master:
        diagnostics = router.build_validation_diagnostics(
            diagnostic_totals,
            diagnostic_paths,
            n_experts=int(cfg.model.n_experts),
        )
        router_metrics = diagnostics.metrics
        router_plots = router_wandb_plots(diagnostics.plot_data)
    elif uses_ffn_moe and is_master:
        diagnostics = router.build_ffn_validation_diagnostics(
            diagnostic_totals,
            diagnostic_paths,
            n_experts=int(cfg.model.moe_ffn_n_experts),
        )
        router_metrics = diagnostics.metrics
        router_plots = ffn_router_wandb_plots(diagnostics.plot_data)
    return EvaluationResult(
        nll=nlls[current_depth_index].item(),
        full_depth_nll=nlls[full_depth_index].item(),
        depth_nlls={
            int(evaluation_depth): nlls[depth_index].item()
            for depth_index, evaluation_depth in enumerate(depths)
            if (
                recurrent
                and evaluation_depth is not None
                and evaluation_depth in router.RECURRENCE_EVAL_DEPTHS
            )
        },
        validation_name=validation_name,
        router_namespace=(
            "router_ffn" if uses_ffn_moe and not outer_routed else "router"
        ),
        byte_nll=(byte_nlls[current_depth_index].item() if byte_nlls is not None else None),
        bits_per_byte=(
            byte_nlls[current_depth_index].item() / math.log(2.0)
            if byte_nlls is not None else None
        ),
        full_depth_byte_nll=(
            byte_nlls[full_depth_index].item() if byte_nlls is not None else None
        ),
        full_depth_bits_per_byte=(
            byte_nlls[full_depth_index].item() / math.log(2.0)
            if byte_nlls is not None else None
        ),
        log_full_depth=recurrent,
        section_accuracies=section_accuracies or None,
        router_metrics=router_metrics,
        router_plots=router_plots,
    )


def print_training_summary(cfg, world_size: int, resume_step: int) -> None:
    effective_per_gpu = (
        cfg.training.batch_size * cfg.training.grad_accum_steps
    )
    print(
        f"Gradient accumulation steps: {cfg.training.grad_accum_steps} | "
        f"Effective per-GPU batch: {effective_per_gpu} | "
        f"Global effective batch: {effective_per_gpu * world_size}"
    )
    if resume_step > 0:
        print(
            f"Continuing training from step {resume_step} to "
            f"{cfg.training.max_steps}"
        )


def train(
    model,
    train_loader,
    val_loader,
    cfg,
    ddp_rank,
    ddp_local_rank,
    ddp_world_size,
    is_master,
    resume_id=None,
):
    """Train either supported architecture from one shared DDP lifecycle."""

    validation_loaders = (
        val_loader if isinstance(val_loader, dict) else {"val": val_loader}
    )
    validation_names = tuple(validation_loaders)
    if not validation_names:
        raise ValueError("At least one validation loader is required")
    val_loader = validation_loaders[validation_names[0]]

    router_runtime = (
        router.build_runtime(cfg, is_master)
        if router.needs_runtime(cfg)
        else None
    )
    flop_runtime = build_flop_runtime(cfg)
    if flop_runtime.enabled and is_master:
        print(
            f"FLOP budget: {flop_runtime.budget:.3e} FLOPs "
            f"({flop_runtime.budget / 1e18:.2f} EFLOPs)."
        )
    model, optimizer = setup_model_and_optimizer(
        model, cfg, router_runtime, ddp_local_rank, is_master,
    )
    os.makedirs(cfg.save_dir, exist_ok=True)
    latest_path = os.path.join(cfg.save_dir, f"latest_{resume_id}.pt")
    state = load_checkpoint(
        latest_path,
        model,
        optimizer,
        train_loader,
        val_loader,
        cfg,
        router_runtime,
        flop_runtime,
        ddp_rank,
        ddp_world_size,
        is_master,
    )
    state.wandb_run_id = initialize_wandb(
        cfg, ddp_world_size, is_master, state.wandb_run_id
    )
    # Multilingual routing diagnostics only mean something when more than one
    # monolingual split shares one routed checkpoint.
    routing_config = RoutingLoggerConfig.from_config(cfg)
    routing_languages = active_languages(validation_names)
    routing_enabled = (
        routing_config.enabled
        and bool(cfg.model.routed)
        and len(routing_languages) > 1
    )
    routing_logger = (
        RoutingLogger(
            routing_config,
            routing_languages,
            int(cfg.model.recurrent_depth),
            int(cfg.model.n_experts),
        )
        if routing_enabled and is_master else None
    )
    if state.runtime_state_by_rank is not None:
        tensors = restore_rank_runtime_state(
            state.runtime_state_by_rank, ddp_rank, cfg.device
        )
        if router_runtime is not None:
            router_runtime.restore_tensors(tensors)

    sigterm_received = False

    def sigterm_handler(signum, frame):
        del signum, frame
        nonlocal sigterm_received
        sigterm_received = True

    signal.signal(signal.SIGTERM, sigterm_handler)
    if is_master:
        print_training_summary(cfg, ddp_world_size, state.resume_step)

    start_time = time.time()
    budget_exhausted = flop_runtime.budget_reached()
    if budget_exhausted and is_master:
        print(
            "The restored checkpoint has already reached its FLOP "
            "budget; keeping it without running another step."
        )

    last_completed_step = state.resume_step - 1
    last_saved_step = last_completed_step
    for step in range(state.resume_step, cfg.training.max_steps):
        if budget_exhausted:
            break

        sigterm_tensor = torch.tensor(
            [1 if sigterm_received else 0],
            device=cfg.device,
            dtype=torch.int,
        )
        dist.all_reduce(sigterm_tensor, op=dist.ReduceOp.MAX)
        if bool(sigterm_tensor.item()):
            if is_master:
                print(
                    f"SIGTERM received at step {step}; saving and exiting."
                )
            save_latest_checkpoint(
                latest_path,
                last_completed_step,
                model,
                optimizer,
                train_loader,
                val_loader,
                cfg,
                router_runtime,
                flop_runtime,
                state,
                ddp_world_size,
                is_master,
            )
            if is_master:
                wandb.finish()
            return

        optimizer_step_started = time.perf_counter() if is_master else None
        result = run_optimizer_step(
            model,
            optimizer,
            train_loader,
            cfg,
            router_runtime,
            flop_runtime,
            step,
        )
        last_completed_step = step
        should_log = step % cfg.logging.log_every == 0
        should_evaluate = step % cfg.logging.eval_every == 0 and step > 0
        loss = ce_loss = aux_loss = grad_norm = None
        if should_log or should_evaluate:
            metrics = distributed_step_metrics(result, ddp_world_size)
            if is_master:
                loss, ce_loss, aux_loss, grad_norm = metrics.tolist()
        optimizer_step_seconds = (
            time.perf_counter() - optimizer_step_started
            if should_log and is_master
            else None
        )

        if should_log and is_master:
            elapsed = time.time() - start_time
            steps_done = step - state.resume_step + 1
            steps_left = cfg.training.max_steps - step - 1
            seconds_per_step = elapsed / steps_done
            eta_seconds = seconds_per_step * steps_left
            elapsed_h, elapsed_remainder = divmod(int(elapsed), 3600)
            elapsed_m, elapsed_s = divmod(elapsed_remainder, 60)
            eta_h, eta_remainder = divmod(int(eta_seconds), 3600)
            eta_m, eta_s = divmod(eta_remainder, 60)
            routed = router_runtime is not None and router_runtime.uses_router
            aux_text = f"  aux={aux_loss:.4f}" if routed else ""
            router_lr_text = (
                f"  rlr={result.router_lr:.2e}" if routed else ""
            )
            adamw_lr_text = (
                f"  alr={result.adamw_lr:.2e}"
                if result.adamw_lr is not None else ""
            )
            print(
                f"[Step {step}/{cfg.training.max_steps}] "
                f"loss={loss:.4f} ce={ce_loss:.4f}{aux_text} "
                f"gnorm={grad_norm:.3f} "
                f"flops={result.step_flops:.3e} "
                f"lr={result.main_lr:.2e}{adamw_lr_text}{router_lr_text} "
                f"elapsed={elapsed_h}h{elapsed_m:02d}m{elapsed_s:02d}s "
                f"ETA={eta_h}h{eta_m:02d}m{eta_s:02d}s "
                f"({seconds_per_step * 1000:.0f}ms/step)"
            )
            global_tokens = (
                ddp_world_size
                * int(cfg.training.batch_size)
                * int(cfg.training.grad_accum_steps)
                * int(cfg.model.block_size)
            )
            fields = {
                "global_step": step,
                "train/loss": loss,
                "train/ce_loss": ce_loss,
                "train/learning_rate": result.main_lr,
                "train/gradient_norm": grad_norm,
                "train/optimizer_step_seconds": optimizer_step_seconds,
                "train/tokens_per_second": (
                    global_tokens / max(optimizer_step_seconds, 1e-12)
                ),
            }
            if result.adamw_lr is not None:
                fields["train/adamw_learning_rate"] = result.adamw_lr
            fields.update(
                {f"train/{key}": value for key, value in
                 flop_runtime.log_fields(
                     result.step_flops, result.flop_depth_measured
                 ).items()}
            )
            if routed:
                fields.update({"router/learning_rate": result.router_lr})
                if bool(cfg.model.routed):
                    fields["router/auxiliary_loss"] = aux_loss
                fields.update(
                    {f"router_ffn/{key}": value for key, value in
                     router_runtime.ffn_log_fields().items()}
                )
            if router_runtime is not None:
                # Recurrence fields keep the router/* names so depth curves
                # compare across the routed and unrouted recurrent middles.
                fields.update(
                    {f"router/{key}": value for key, value in
                     router_runtime.log_fields(result.recurrence_sample).items()}
                )
            wandb.log(fields)

        if should_evaluate:
            routing_accumulator = (
                RoutingAccumulator(
                    routing_languages,
                    int(cfg.model.recurrent_depth),
                    int(cfg.model.n_experts),
                    max_windows=routing_config.max_windows,
                    rank=ddp_rank,
                    device=cfg.device,
                )
                if routing_enabled else None
            )
            evaluations = evaluate_validation_sets(
                model,
                validation_loaders,
                cfg,
                router_runtime,
                result.depth,
                routing_accumulator=routing_accumulator,
                routing_languages=routing_languages,
                routing_trace_batches=routing_config.trace_batches,
            )
            routing_fields = None
            if routing_accumulator is not None:
                # Collective, so every rank has to reach it before the
                # master-only logging below.
                proceed = routing_accumulator.all_reduce()
                if proceed and routing_logger is not None:
                    try:
                        routing_fields = routing_logger.log_fields(
                            routing_accumulator, step
                        )
                    except Exception as error:
                        print(
                            "WARNING: multilingual routing logging failed "
                            f"({type(error).__name__}: {error}); continuing."
                        )
            if is_master:
                for evaluation in evaluations:
                    validation_name = evaluation.validation_name
                    if (
                        router_runtime is not None
                        and router_runtime.recurrence_enabled
                    ):
                        print(
                            f"Step {step} | Loss: {loss:.4f} | "
                            f"Val {validation_name} NLL (depth {result.depth}): "
                            f"{evaluation.nll:.4f} | "
                            f"Full-depth Val NLL "
                            f"(depth {cfg.model.recurrent_depth}): "
                            f"{evaluation.full_depth_nll:.4f} | "
                            f"LR: {result.main_lr:.2e}"
                        )
                    else:
                        print(
                            f"Step {step} | Loss: {loss:.4f} | "
                            f"Val {validation_name} NLL: {evaluation.nll:.4f} | "
                            f"LR: {result.main_lr:.2e}"
                        )
                    wandb.log({
                        "global_step": step,
                        **evaluation_log_fields(evaluation),
                    })
                if routing_fields:
                    # Scalars, figures and the table go out together: two
                    # calls at one step silently drop panels.
                    wandb.log({"global_step": step, **routing_fields})

        if step > 0 and step % cfg.logging.save_every == 0:
            save_latest_checkpoint(
                latest_path,
                step,
                model,
                optimizer,
                train_loader,
                val_loader,
                cfg,
                router_runtime,
                flop_runtime,
                state,
                ddp_world_size,
                is_master,
            )
            last_saved_step = step

        if flop_runtime.budget_reached():
            if is_master:
                print(
                    f"FLOP budget reached at step {step}: "
                    f"total_flops={flop_runtime.total_flops:.4e} >= "
                    f"budget={flop_runtime.budget:.4e}. "
                    "Stopping training."
                )
            if last_saved_step != step:
                save_latest_checkpoint(
                    latest_path,
                    step,
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    cfg,
                    router_runtime,
                    flop_runtime,
                    state,
                    ddp_world_size,
                    is_master,
                )
                last_saved_step = step
            budget_exhausted = True
            break

    if last_completed_step > last_saved_step:
        save_latest_checkpoint(
            latest_path,
            last_completed_step,
            model,
            optimizer,
            train_loader,
            val_loader,
            cfg,
            router_runtime,
            flop_runtime,
            state,
            ddp_world_size,
            is_master,
        )

    dist.barrier()
    if is_master:
        if os.path.exists(latest_path):
            print(f"Last checkpoint kept at: {latest_path}")
        wandb.finish()
    dist.barrier()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(hydra_cfg: DictConfig):
    cfg = resolve_config(hydra_cfg)
    validate_training_entrypoint(cfg, supported_arches=SUPPORTED_ARCHES)
    context = initialize_distributed(cfg)
    try:
        model, _ = build_model(cfg.model, allowed_arches=SUPPORTED_ARCHES)
        if context.is_master:
            parameters = sum(
                parameter.numel() for parameter in model.parameters()
            )
            print(
                f"{cfg.model.arch} parameters: {parameters / 1e6:.1f}M"
            )

        train_loader = DataLoaderLit_MultiProc(
            cfg.data.data_dir,
            cfg.training.batch_size,
            cfg.model.block_size,
            split="train",
            process_rank=context.rank,
            num_processes=context.world_size,
            eot_token_id=int(cfg.model.eot_token_id),
            model_vocab_size=int(cfg.model.vocab_size),
        )
        validation_names = tuple(
            str(name) for name in cfg.data.get("validation_sets", ["val"])
        )
        if not validation_names:
            raise ValueError("data.validation_sets must not be empty")
        val_loader = {
            name: DataLoaderLit_MultiProc(
                cfg.data.data_dir,
                cfg.training.batch_size,
                cfg.model.block_size,
                split=name,
                process_rank=context.rank,
                num_processes=context.world_size,
                eot_token_id=int(cfg.model.eot_token_id),
                model_vocab_size=int(cfg.model.vocab_size),
            )
            for name in validation_names
        }

        resume_id = config_fingerprint(cfg, context.world_size)
        if context.is_master:
            print(
                f"Run fingerprint: {resume_id} "
                f"(checkpoint: latest_{resume_id}.pt)"
            )
            for checkpoint_path in clear_fresh_checkpoints(cfg, resume_id):
                print(f"[run.fresh] Removed {checkpoint_path}")
        dist.barrier()

        train(
            model,
            train_loader,
            val_loader,
            cfg,
            context.rank,
            context.local_rank,
            context.world_size,
            context.is_master,
            resume_id=resume_id,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
