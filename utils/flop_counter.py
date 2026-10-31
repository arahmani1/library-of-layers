"""Low-overhead, shape-theoretical counting of training arithmetic.

Counts use the configured per-rank batch and sequence shapes, treating every
row as one full document. Dynamic-window counts include padding executed by
shifted partial windows. This is an algorithmic count, not a hardware counter:
floating arithmetic and transcendentals each cost one FLOP, multiply-add costs
two, and indexing, comparisons, casts, movement, RNG, and integer operations
cost zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from model.attention_strategies import (
    canonical_window_counts,
    parse_attention_strategy_name,
)


from model.config import validate_huginn_config, validate_paper_normalization


COUNT_VERSION = 14


def _swiglu_hidden(n_embd: int) -> int:
    """Match :class:`model.transformer.SwiGLU` hidden-size rounding."""

    hidden = int(2 * (4 * n_embd) / 3)
    return 256 * ((hidden + 255) // 256)


def _enum_value(value) -> str:
    return str(getattr(value, "value", value)).lower()


def _linear(rows: int, input_width: int, output_width: int, bias: bool) -> int:
    """Forward plus autograd backward for a dense linear projection."""

    matmul = 6 * rows * input_width * output_width
    if not bias:
        return matmul
    # Forward bias broadcast plus bias-gradient reduction.
    return matmul + rows * output_width + max(rows - 1, 0) * output_width


def _norm(groups: int, width: int) -> int:
    """RMS/QK normalization forward and primitive autograd arithmetic."""

    values = groups * width
    forward = groups * (4 * width + 2)
    input_backward = groups * (7 * width + 2)
    weight_backward = values + max(groups - 1, 0) * width
    return forward + input_backward + weight_backward


def _softmax(rows: int, width: int) -> int:
    """Softmax forward and backward; max comparisons are deliberately free."""

    return rows * max(8 * width - 2, 0)


def _silu(values: int) -> int:
    # sigmoid(x) * x forward and its primitive backward expression.
    return 13 * values



def _linear_forward(
    rows: int, input_width: int, output_width: int, bias: bool,
) -> int:
    """Forward-only arithmetic for a dense linear projection."""

    total = 2 * rows * input_width * output_width
    return total + (rows * output_width if bias else 0)


def _norm_forward(groups: int, width: int) -> int:
    """Forward-only RMS/QK normalization arithmetic."""

    return groups * (4 * width + 2)


def _silu_forward(values: int) -> int:
    """Forward sigmoid-and-multiply SiLU arithmetic."""

    return 2 * values

def _attention(
    token_rows: int,
    n_head: int,
    head_dim: int,
    causal_pairs: int,
) -> int:
    """Attention forward/backward over the allowed query-key pairs."""

    # Two forward and five backward matrix multiplications.
    matrix = 14 * n_head * head_dim * causal_pairs
    # Score scaling forward/backward plus softmax forward/backward.
    scalar = n_head * (10 * causal_pairs - 2 * token_rows)
    return matrix + scalar


def _attention_only_block(
    rows: int, causal_pairs: int, cfg, *, forward_only: bool = False,
) -> int:
    """Arithmetic for one pre-norm attention block."""

    width = int(cfg.model.n_embd)
    heads = int(cfg.model.n_head)
    head_dim = width // heads
    norm = _norm_forward if forward_only else _norm
    linear = _linear_forward if forward_only else _linear
    attention = _attention_forward if forward_only else _attention
    total = norm(rows, width)
    total += linear(rows, width, 3 * width, bias=False)
    total += 2 * norm(rows * heads, head_dim)
    total += (6 if forward_only else 12) * rows * width
    total += attention(rows, heads, head_dim, causal_pairs)
    total += linear(rows, width, width, bias=False)
    total += rows * width
    return total


def _block(
    rows: int, causal_pairs: int, cfg, ffn_rows: int | None = None,
    *, forward_only: bool = False,
    sandwich: bool = False, moe_ffn: bool | None = None,
) -> int:
    width = int(cfg.model.n_embd)
    ffn_rows = rows if ffn_rows is None else int(ffn_rows)
    linear = _linear_forward if forward_only else _linear
    silu = _silu_forward if forward_only else _silu

    total = _attention_only_block(
        rows, causal_pairs, cfg, forward_only=forward_only
    )
    norm = _norm_forward if forward_only else _norm
    total += norm(rows, width)
    if sandwich:
        # Sandwich blocks re-normalize the residual stream after the
        # attention and FFN sublayers.
        total += 2 * norm(rows, width)
    uses_moe_ffn = (
        bool(cfg.model.get("moe_ffn", False))
        if moe_ffn is None else bool(moe_ffn)
    )
    if uses_moe_ffn:
        experts = int(cfg.model.moe_ffn_n_experts)
        top_k = int(cfg.model.moe_ffn_top_k)
        dense_hidden = _swiglu_hidden(width)
        divided = (dense_hidden + top_k - 1) // top_k
        hidden = 256 * ((divided + 255) // 256)
        if cfg.model.get("moe_ffn_hidden_dim") is not None:
            hidden = int(cfg.model.moe_ffn_hidden_dim)
        total += linear(ffn_rows, width, experts, bias=False)
        total += 8 * ffn_rows * experts
        active_rows = ffn_rows * top_k
        total += linear(active_rows, width, 2 * hidden, bias=False)
        total += silu(active_rows * hidden)
        total += 3 * active_rows * hidden
        total += linear(active_rows, hidden, width, bias=False)
        total += 8 * active_rows + 5 * active_rows * width
        total += 6 * active_rows + 4 * experts
    else:
        hidden = _swiglu_hidden(width)
        total += linear(ffn_rows, width, 2 * hidden, bias=False)
        total += silu(ffn_rows * hidden)
        total += (1 if forward_only else 3) * ffn_rows * hidden
        total += linear(ffn_rows, hidden, width, bias=False)
    total += rows * width
    return total


def _causal_pairs(
    batch: int, length: int, window: int | None = None,
    hidden_span=None,
) -> int:
    """Allowed causal query-key pairs for full canonical rows.

    ``hidden_span`` is ``model.hidden_key_span``: keys in ``[start, stop)``
    are visible only to queries inside the span, so every later query that
    the causal window would otherwise let see them loses those pairs.
    """

    visible = (
        int(length)
        if window is None
        else min(int(length), int(window))
    )
    row_pairs = (
        visible * (visible + 1) // 2
        + (int(length) - visible) * visible
    )
    if hidden_span is not None:
        start, stop = (min(int(bound), int(length)) for bound in hidden_span)
        for key in range(start, stop):
            last_query = int(length) if window is None else min(
                int(length), key + int(window)
            )
            row_pairs -= max(0, last_query - stop)
    return int(batch) * row_pairs


def _cross_entropy(rows: int, vocab_size: int) -> int:
    # Log-softmax, target selection/negation, mean reduction, and backward.
    log_softmax = rows * max(8 * vocab_size - 2, 0)
    target_and_mean = rows + max(rows - 1, 0) + 1
    backward_target = rows * vocab_size + rows
    return log_softmax + target_and_mean + backward_target


def _router(rows: int, cfg, *, forward_only: bool = False) -> int:
    width = int(cfg.model.n_embd)
    experts = int(cfg.model.n_experts)
    linear = _linear_forward if forward_only else _linear
    total = linear(rows, width, experts, bias=True)
    total += linear(rows, experts, experts, bias=True)
    total += (4 if forward_only else 8) * rows * experts
    softmax = (
        rows * max(3 * experts - 1, 0)
        if forward_only else _softmax(rows, experts)
    )
    return total + softmax


def _routing_glue(
    tokens: int,
    windows: int,
    cfg,
    *,
    forward_only: bool = False,
) -> int:
    width = int(cfg.model.n_embd)
    experts = int(cfg.model.n_experts)
    passes = 1 if forward_only else 2
    # Window lengths come from integer bincount; only hidden sums/divisions
    # execute floating arithmetic (no per-token float count accumulation).
    pooling = passes * (tokens + windows) * width
    prefix = 2 * passes * windows * width
    if _enum_value(cfg.model.get("first_window_routing", "")) != "learned_token_depth":
        raise ValueError("LoL FLOPs require first_window_routing='learned_token_depth'")
    feature_projection = 0 if forward_only else max(windows - 1, 0) * width
    feature_projection += (2 if forward_only else 3) * windows * width
    balance = passes * windows * (3 * experts + 4)
    blend = (3 if forward_only else 7) * tokens * width
    return (
        pooling + prefix + feature_projection
        + _router(windows, cfg, forward_only=forward_only)
        + balance + blend
    )


def _injection(rows: int, cfg, *, forward_only: bool = False) -> int:
    width = int(cfg.model.n_embd)
    parameter_transform = (5 if forward_only else 19) * width
    recurrent_update = (3 if forward_only else 9) * rows * width
    return parameter_transform + recurrent_update


def _attention_forward(
    token_rows: int,
    n_head: int,
    head_dim: int,
    causal_pairs: int,
) -> int:
    """Forward-only attention over the allowed query-key pairs."""

    matrix = 4 * n_head * head_dim * causal_pairs
    scalar = n_head * (4 * causal_pairs - token_rows)
    return matrix + scalar


def training_microbatch_flops(
    cfg, depth: int | None,
    gradient_depth: int | None = None,
) -> int:
    """Count one configured local microbatch under the FLOP convention.

    ``gradient_depth=None`` counts full BPTT. A smaller value counts a final
    gradient-bearing suffix and is accepted only for the stationary diagonal
    LoL profile enforced by the training runtime.
    """

    validate_paper_normalization(cfg.model)
    batch = int(cfg.training.batch_size)
    length = int(cfg.model.block_size)
    rows = int(batch * length)
    width = int(cfg.model.n_embd)
    vocab = int(cfg.model.vocab_size)
    strategy = parse_attention_strategy_name(
        cfg.model.attention_strategy
    ).value
    hidden_span = cfg.model.get("hidden_key_span")
    full_pairs = _causal_pairs(batch, length, hidden_span=hidden_span)
    local_pairs = _causal_pairs(
        batch,
        length,
        window=(
            int(cfg.model.window)
            if strategy == "sliding_window"
            else None
        ),
        hidden_span=hidden_span,
    )

    injection = _enum_value(
        cfg.model.get("recurrent_input_injection", "none")
    )
    if str(cfg.model.arch) == "vanilla_neo":
        blocks = (
            int(cfg.model.n_layer) if depth is None else int(depth)
        )
        if not 1 <= blocks <= int(cfg.model.n_layer):
            raise ValueError(
                "Vanilla active depth must be in "
                f"[1, {int(cfg.model.n_layer)}], got {blocks}"
            )
        if strategy == "shifted_window":
            middle_blocks = blocks - 2
            total = 2 * _block(
                rows, full_pairs, cfg,
            )
            window = int(cfg.model.window)
            window_pairs = window * (window + 1) // 2
            for windows in canonical_window_counts(
                batch,
                length,
                window=int(cfg.model.window),
                stride=int(cfg.model.window_stride),
                depth=middle_blocks,
            ):
                padded_rows = windows * window
                total += _block(
                    padded_rows,
                    windows * window_pairs,
                    cfg,
                    ffn_rows=(
                        rows
                        if bool(cfg.model.get("moe_ffn", False))
                        else padded_rows
                    ),
                    )
        else:
            if strategy == "full_causal":
                local_blocks = 0
                global_blocks = blocks
            elif strategy == "sliding_window":
                local_blocks = blocks
                global_blocks = 0
            else:
                raise ValueError(
                    f"Unsupported Vanilla attention strategy {strategy!r} for FLOPs."
                )
            total = local_blocks * _block(
                rows, local_pairs, cfg,
            )
            total += global_blocks * _block(
                rows, full_pairs, cfg,
            )
        total += _norm(rows, width)
    elif str(cfg.model.arch) == "lol_dynamic_window":
        if _enum_value(cfg.model.get("first_window_routing", "")) != "learned_token_depth":
            raise ValueError("LoL FLOPs require first_window_routing='learned_token_depth'")
        recurrent_depth = (
            int(depth)
            if depth is not None
            else int(cfg.model.recurrent_depth)
        )
        if gradient_depth is None:
            gradient_recurrent_depth = recurrent_depth
        else:
            requested_gradient_depth = int(gradient_depth)
            if requested_gradient_depth < 1:
                raise ValueError("gradient_depth must be at least 1")
            gradient_recurrent_depth = min(
                requested_gradient_depth, recurrent_depth
            )
        no_grad_recurrent_depth = (
            recurrent_depth - gradient_recurrent_depth
        )
        mixed_backprop = no_grad_recurrent_depth > 0 and bool(
            (cfg.training.get("recurrence_curriculum") or {}).get("first_window_full_backprop", False)
        )
        prelude_blocks = int(cfg.model.get("n_prelude_layers", 1))
        uses_ffn_moe = bool(cfg.model.get("moe_ffn", False))
        if uses_ffn_moe and str(cfg.model.get("moe_ffn_scope", "all")) != "prelude_coda":
            raise ValueError("Paper LoL inner MoE requires moe_ffn_scope='prelude_coda'.")
        coda_blocks = int(cfg.model.get("n_coda_layers", 1))
        total = prelude_blocks * _block(
            rows, full_pairs, cfg,
            moe_ffn=uses_ffn_moe,
        )
        if injection == "diagonal":
            total += _norm(rows, width)
            total += _linear(rows, width, width, bias=False)
        # Windows are packed into slots of pack_width tokens: model.window,
        # or the pinned first window when it is wider (attention_strategies).
        # Padding in those slots executes attention and dense FFN arithmetic.
        first_window = cfg.model.get("first_window_length")
        window = max(
            int(cfg.model.window),
            0 if first_window is None else int(first_window),
        )
        window_pairs = window * (window + 1) // 2
        for current_depth, windows in enumerate(canonical_window_counts(
            batch,
            length,
            window=int(cfg.model.window),
            stride=int(cfg.model.window_stride),
            depth=recurrent_depth,
            first_window=first_window,
        )):
            forward_only = current_depth < no_grad_recurrent_depth
            if injection == "diagonal":
                total += _injection(
                    rows, cfg, forward_only=forward_only and not mixed_backprop
                )
            padded_rows = windows * window
            if bool(cfg.model.get("depth_expert_film", False)):
                if forward_only and not mixed_backprop:
                    total += 4 * padded_rows * width
                elif forward_only:
                    gradient_rows = batch * window
                    total += (
                        4 * padded_rows * width
                        + 8 * gradient_rows * width
                        - 4 * width
                    )
                else:
                    total += 12 * padded_rows * width - 4 * width
            total += _routing_glue(
                rows,
                windows,
                cfg,
                # Router feature/head kernels and residual combination remain
                # batched, with detached scores/states outside the active mask.
                forward_only=forward_only and not mixed_backprop,
            )
            total += _block(
                padded_rows,
                windows * window_pairs,
                cfg,
                forward_only=forward_only,
                moe_ffn=False,
            )
            if mixed_backprop:
                # Per-window balancing gradient rescaling (forward + backward).
                total += 4 * windows * int(cfg.model.n_experts)
                if forward_only:
                    # One first window per canonical document/row. The prefix
                    # executes all expert forwards, but backward only for these
                    # padded windows; no first-window forward is duplicated.
                    first_rows = batch * window
                    total += _block(
                        first_rows, batch * window_pairs, cfg,
                        moe_ffn=False,
                    ) - _block(
                        first_rows, batch * window_pairs, cfg,
                        forward_only=True, moe_ffn=False,
                    )
        total += coda_blocks * _block(
            rows, full_pairs, cfg,
            moe_ffn=uses_ffn_moe,
        )
        total += _norm(rows, width)
    elif str(cfg.model.arch) == "huginn":
        validate_huginn_config(cfg.model)
        recurrent_depth = (
            int(depth)
            if depth is not None
            else int(cfg.model.recurrent_depth)
        )
        if gradient_depth is None:
            gradient_recurrent_depth = recurrent_depth
        else:
            if int(gradient_depth) < 1:
                raise ValueError("gradient_depth must be at least 1")
            gradient_recurrent_depth = min(
                int(gradient_depth), recurrent_depth
            )
        no_grad_recurrent_depth = recurrent_depth - gradient_recurrent_depth
        sandwich = True
        core_blocks = int(cfg.model.n_layer)
        total = int(cfg.model.get("n_prelude_layers", 1)) * _block(
            rows, full_pairs, cfg, sandwich=sandwich
        )
        # Scaling the std-1 sampled initial state; the draw itself is RNG.
        total += rows * width
        for current_depth in range(recurrent_depth):
            forward_only = current_depth < no_grad_recurrent_depth
            linear = _linear_forward if forward_only else _linear
            total += linear(rows, 2 * width, width, bias=False)
            for core_index in range(core_blocks):
                block_rows = rows
                causal_pairs = full_pairs
                total += _block(
                    block_rows,
                    causal_pairs,
                    cfg,
                    sandwich=sandwich,
                    forward_only=forward_only,
                )
        # The recurrence exit reuses the final norm before the coda.
        total += _norm(rows, width)
        total += int(cfg.model.get("n_coda_layers", 1)) * _block(
            rows, full_pairs, cfg, sandwich=sandwich
        )
        total += _norm(rows, width)
    else:
        raise ValueError(
            f"Unsupported FLOP-calibrated architecture {cfg.model.arch!r}"
        )

    total += _linear(rows, width, vocab, bias=False)
    total += _cross_entropy(rows, vocab)
    # Loss scaling by gradient accumulation in forward and backward.
    total += 2
    return int(total)


def training_step_flops(
    cfg, depth=None, gradient_depth=None,
) -> int:
    """Count one configured local optimizer step."""

    return int(cfg.training.grad_accum_steps) * training_microbatch_flops(
        cfg, depth, gradient_depth=gradient_depth
    )


@dataclass
class FlopRuntime:
    """Process-local per-(total, gradient) counts and cumulative totals."""

    enabled: bool
    budget: float
    first_window_full_backprop: bool = False
    measured_step_flops: dict[tuple[int, ...], int] = field(
        default_factory=dict
    )
    calibrated_flops: int = 0

    def _key(
        self, depth: int | None, gradient_depth: int | None = None,
    ) -> tuple[int, ...]:
        total = -1 if depth is None else int(depth)
        gradient = total if gradient_depth is None else int(gradient_depth)
        return (total, gradient, 1) if self.first_window_full_backprop else (total, gradient)

    @property
    def total_flops(self) -> int:
        return self.calibrated_flops

    def needs_measurement(
        self, depth: int | None, gradient_depth: int | None = None,
    ) -> bool:
        key = self._key(depth, gradient_depth)
        return key not in self.measured_step_flops

    def record_step(
        self,
        depth: int | None,
        local_measurement: int | None,
        device: torch.device | str,
        gradient_depth: int | None = None,
    ) -> tuple[int, bool]:
        key = self._key(depth, gradient_depth)
        measured = key not in self.measured_step_flops
        if measured:
            if local_measurement is None:
                raise ValueError(
                    "First step at a depth pair requires a FLOP measurement"
                )
            value = torch.tensor(
                int(local_measurement), device=device, dtype=torch.int64
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            step_flops = int(value.item())
            self.measured_step_flops[key] = step_flops
        else:
            step_flops = self.measured_step_flops[key]
        self.calibrated_flops += step_flops
        return step_flops, measured

    def restore_checkpoint(self, checkpoint: dict) -> None:
        self.measured_step_flops.clear()
        version = int(checkpoint.get("flop_count_version", 0))
        if version not in {7, 8, 10, 11, 12, 13, COUNT_VERSION}:
            raise ValueError("Unsupported checkpoint FLOP-count format")
        self.calibrated_flops = int(checkpoint.get("calibrated_flops", 0))

    def checkpoint_fields(self) -> dict[str, int | float]:
        return {
            "flop_count_version": COUNT_VERSION,
            "calibrated_flops": self.calibrated_flops,
            "total_flops": self.total_flops,
        }

    def budget_reached(self) -> bool:
        return self.enabled and self.total_flops >= self.budget

    def log_fields(
        self, step_flops: int, measured: bool
    ) -> dict[str, int | float | bool]:
        fields: dict[str, int | float | bool] = {
            "step_flops": step_flops,
            "total_flops": self.total_flops,
            "flop_depth_measured": measured,
        }
        if self.enabled:
            fields.update(
                flop_budget_frac=self.total_flops / self.budget,
                flop_budget_remaining=max(0.0, self.budget - self.total_flops),
            )
        return fields


def build_runtime(cfg) -> FlopRuntime:
    budget_cfg = cfg.training.get("flop_budget") or {}
    enabled = bool(budget_cfg.get("enabled", False))
    budget = float(budget_cfg.get("total_flops", 0.0)) if enabled else 0.0
    if enabled and budget <= 0:
        raise ValueError(
            "training.flop_budget.total_flops must be > 0 when enabled."
        )
    return FlopRuntime(
        enabled=enabled,
        budget=budget,
        first_window_full_backprop=bool((cfg.training.get("recurrence_curriculum") or {}).get("first_window_full_backprop", False)),
    )
