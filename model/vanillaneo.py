from dataclasses import dataclass

import torch
import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch.nn import functional as F

from .attention_strategies import (
    build_attention_strategy,
    execute_packed_window_block,
)
from .config import RecurrentInputInjection
from .transformer import (
    MoESwiGLU,
    RMSNorm,
    TransformerBlockNeo,
    precompute_freqs_cis,
)


_FUSED_LINEAR_CROSS_ENTROPY = LigerFusedLinearCrossEntropyLoss(
    ignore_index=-100, reduction="sum"
)


def _linear_cross_entropy_output(
    lm_head, hidden, targets, loss_mask=None,
):
    """Return logits for inference or memory-efficient CE for training."""

    if targets is None:
        return lm_head(hidden)
    if loss_mask is None:
        masked_targets = targets
        denominator = (targets != -100).sum().clamp_min(1)
    else:
        loss_mask = loss_mask & (targets != -100)
        masked_targets = targets.masked_fill(~loss_mask, -100)
        denominator = loss_mask.sum().clamp_min(1)

    hidden = hidden.reshape(-1, hidden.size(-1))
    masked_targets = masked_targets.reshape(-1)
    if hidden.device.type == "cuda":
        loss_sum = _FUSED_LINEAR_CROSS_ENTROPY(
            lm_head.weight, hidden, masked_targets
        )
    else:
        loss_sum = F.cross_entropy(
            lm_head(hidden), masked_targets,
            reduction="sum", ignore_index=-100,
        )
    return loss_sum / denominator


@dataclass(frozen=True)
class ForwardContext:
    idx: torch.Tensor
    doc_id: torch.Tensor
    position_ids: torch.Tensor
    block_mask: object
    local_block_mask: object | None = None
    input_valid: torch.Tensor | None = None


@dataclass
class MiddleResult:
    hidden: torch.Tensor
    aux_loss: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None
    router_stats: dict[str, torch.Tensor] | None = None


class VanillaGPTNeo(nn.Module):
    """Concrete Neo backbone whose default middle is a dense block stack."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_strategy = build_attention_strategy(
            config, self._attention_architecture()
        )
        self.uses_ffn_moe = bool(config.moe_ffn)
        self.moe_ffn_scope = str(config.moe_ffn_scope)
        if (
            self.moe_ffn_scope != "all"
            and self._attention_architecture()
            != "lol_dynamic_window"
        ):
            raise ValueError(
                "moe_ffn_scope='prelude_coda' requires a LoL model."
            )
        self.eot_token_id = int(getattr(config, "eot_token_id", 0))
        injection = config.recurrent_input_injection
        if (
            injection is RecurrentInputInjection.DIAGONAL
            and type(self) is VanillaGPTNeo
        ):
            raise ValueError(
                "diagonal recurrent input injection requires a routed model."
            )
        if (
            injection is RecurrentInputInjection.ADAPTER
            and type(self) is VanillaGPTNeo
        ):
            raise ValueError(
                "adapter recurrent input injection requires "
                "arch=huginn."
            )

        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        head_dim = config.n_embd // config.n_head
        if head_dim % 2 != 0:
            raise ValueError("Per-head dimension must be even for RoPE.")

        span = getattr(config, "hidden_key_span", None)
        self.hidden_key_span = (
            None if span is None else (int(span[0]), int(span[1]))
        )
        embedding = nn.Embedding(config.vocab_size, config.n_embd)
        middle = self._build_middle(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": embedding,
                **middle,
                "ln_f": RMSNorm(config.n_embd),
            }
        )
        self._build_middle_boundaries(config)
        self._ffn_moe_module_cache = tuple(
            module for module in self.modules()
            if isinstance(module, MoESwiGLU)
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        freqs_cis = precompute_freqs_cis(head_dim, config.block_size)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.supports_chunked_linear_cross_entropy = True

    def _attention_architecture(self):
        return "vanilla_neo"

    def _build_middle(self, config):
        return {
            "h": nn.ModuleList(
                [TransformerBlockNeo(config) for _ in range(config.n_layer)]
            )
        }

    def _build_middle_boundaries(self, config):
        del config

    @staticmethod
    def _compute_doc_meta(idx, eot_token_id, input_valid=None):
        """Infer document and per-document position ids from packed tokens."""
        _, sequence_length = idx.shape
        is_eot = idx == eot_token_id
        if input_valid is not None:
            is_eot = is_eot & input_valid
        doc_id = is_eot.long().cumsum(dim=1)
        positions = torch.arange(
            sequence_length, device=idx.device
        ).unsqueeze(0).expand_as(idx)
        eot_positions = torch.where(
            is_eot, positions, torch.full_like(positions, -1)
        )
        last_eot, _ = eot_positions.cummax(dim=1)
        position_ids = positions - last_eot.clamp(min=0)
        return doc_id, position_ids

    def _prepare_context(self, idx, doc_id, position_ids, input_valid=None):
        if doc_id is None or position_ids is None:
            doc_id, position_ids = self._compute_doc_meta(
                idx, self.eot_token_id, input_valid
            )
        hidden_key = None
        if self.hidden_key_span is not None:
            start, stop = self.hidden_key_span
            hidden_key = (position_ids >= start) & (position_ids < stop)
        block_mask, local_block_mask = (
            self.attention_strategy.prepare_masks(
                doc_id, hidden_key=hidden_key
            )
        )
        return ForwardContext(
            idx=idx,
            doc_id=doc_id,
            position_ids=position_ids,
            block_mask=block_mask,
            local_block_mask=local_block_mask,
            input_valid=input_valid,
        )

    def _block_mask_for_layer(self, context):
        if self.attention_strategy.uses_windows:
            return context.block_mask
        return self.attention_strategy.block_mask(
            context.block_mask,
            context.local_block_mask,
        )

    def _packed_windows(self, hidden, context, depth):
        metadata = self.attention_strategy.metadata(
            context.doc_id, context.position_ids, depth,
            input_valid=context.input_valid,
        )
        return self.attention_strategy.pack(
            hidden,
            context.position_ids,
            metadata,
        )

    @staticmethod
    def _restore_packed_ffn_routes(block, packed):
        module = block.mlp
        routes = (
            module.last_routes if isinstance(module, MoESwiGLU) else None
        )
        if routes:
            routes[-1] = routes[-1].view(
                *packed.valid.shape, module.top_k
            )[
                packed.token_windows, packed.token_positions
            ]

    def _apply_window_block(self, block, hidden, context, depth):
        packed = self._packed_windows(hidden, context, depth)
        output = execute_packed_window_block(
            block,
            packed.values,
            packed.position_ids,
            packed.valid,
            self.freqs_cis,
        )
        self._restore_packed_ffn_routes(block, packed)
        return packed.restore(output)

    def _apply_block(self, block, hidden, context, block_mask=None):
        if block_mask is None:
            block_mask = context.block_mask
        return block(
            hidden,
            self.freqs_cis,
            position_ids=context.position_ids,
            block_mask=block_mask,
        )

    def _ffn_moe_modules(self):
        return self._ffn_moe_module_cache

    def _begin_ffn_stats(self, *, capture_routes=False):
        for module in self._ffn_moe_modules():
            module.begin_stats(capture_routes=capture_routes)

    def _finish_ffn_stats(self, hidden):
        modules = self._ffn_moe_modules()
        zero = hidden.new_zeros(
            modules[0].n_experts, dtype=torch.float32
        )
        rows = []
        for module in modules:
            loads, importance = module.end_stats()
            rows.append(
                (zero, zero) if loads is None else (loads, importance)
            )
        loads = torch.stack([row[0] for row in rows])
        importance = torch.stack([row[1] for row in rows])
        load_fraction = loads / loads.sum(-1, keepdim=True).clamp_min(1.0)
        importance_fraction = importance / importance.sum(
            -1, keepdim=True
        ).clamp_min(1e-12)
        active = loads.sum(-1) > 0
        per_site_aux = loads.size(-1) * (
            load_fraction * importance_fraction
        ).sum(-1)
        active = active.to(per_site_aux.dtype)
        aux_loss = (
            (per_site_aux * active).sum() / active.sum().clamp_min(1.0)
            + hidden.sum() * 0.0
        )
        stats = {
            "ffn_load_sum": loads,
            "ffn_importance_sum": importance,
            "ffn_aux_loss": aux_loss,
        }
        token_count = hidden.size(0) * hidden.size(1)
        captured = [module.last_routes for module in modules]
        if all(
            routes is not None
            and len(routes) == 1
            and routes[0].size(0) == token_count
            for routes in captured
        ):
            stats["ffn_routing_token_paths"] = torch.stack(
                [routes[0] for routes in captured], dim=1
            ).view(
                hidden.size(0),
                hidden.size(1),
                len(modules),
                modules[0].top_k,
            )
        return stats

    def _forward_middle(
        self,
        hidden,
        context,
        *,
        return_router_stats=False,
        return_router_diagnostics=True,
        return_router_trace=False,
        num_steps=None,
        backprop_steps=None,
        first_window_full_backprop=False,
    ):
        del (
            num_steps,
            return_router_stats,
            return_router_diagnostics,
            return_router_trace,
            backprop_steps,
            first_window_full_backprop,
        )
        block_count = len(self.transformer.h)
        blocks = self.transformer.h
        for index, block in enumerate(blocks):
            packed_depth = self.attention_strategy.packed_depth(
                index, block_count
            )
            if packed_depth is not None:
                hidden = self._apply_window_block(
                    block, hidden, context, packed_depth
                )
            else:
                hidden = self._apply_block(
                    block, hidden, context,
                    block_mask=self._block_mask_for_layer(context),
                )
        return MiddleResult(hidden)

    def _format_output(self, lm_output, middle, return_router_stats):
        if return_router_stats and self.uses_ffn_moe:
            return (
                lm_output,
                middle.hidden.sum() * 0.0,
                middle.loss_mask,
                middle.router_stats,
            )
        return lm_output

    def forward(
        self,
        idx,
        doc_id=None,
        position_ids=None,
        targets=None,
        return_router_stats=False,
        return_router_diagnostics=True,
        backprop_steps=None,
        first_window_full_backprop=False,
        return_router_trace=False,
        num_steps=None,
        input_valid=None,
    ):
        return_router_stats = return_router_stats or return_router_trace
        collect_ffn_stats = self.uses_ffn_moe and return_router_stats
        if collect_ffn_stats:
            self._begin_ffn_stats(capture_routes=return_router_trace)
        try:
            context = self._prepare_context(
                idx, doc_id, position_ids, input_valid
            )
            middle_kwargs = {
                "return_router_stats": return_router_stats,
                "return_router_diagnostics": return_router_diagnostics,
                "return_router_trace": return_router_trace,
                "num_steps": num_steps,
            }
            if backprop_steps is not None:
                middle_kwargs["backprop_steps"] = backprop_steps
            if first_window_full_backprop:
                middle_kwargs["first_window_full_backprop"] = True
            middle = self._forward_middle(
                self.transformer.wte(idx), context, **middle_kwargs
            )
            ffn_stats = (
                self._finish_ffn_stats(middle.hidden)
                if collect_ffn_stats else None
            )
            if ffn_stats is not None:
                middle.router_stats = {
                    **(middle.router_stats or {}),
                    **ffn_stats,
                }
            if targets is not None and middle.loss_mask is not None:
                middle.loss_mask = middle.loss_mask & (targets != -100)
            hidden = self.transformer.ln_f(middle.hidden)
            lm_output = _linear_cross_entropy_output(
                self.lm_head, hidden, targets, middle.loss_mask
            )
            return self._format_output(
                lm_output, middle, return_router_stats
            )
        finally:
            if collect_ffn_stats:
                for module in self._ffn_moe_modules():
                    module.end_stats()

    def _compile_middle(self):
        for index in range(len(self.transformer.h)):
            block = self.transformer.h[index]
            dynamic = (
                self.attention_strategy.packed_depth(
                    index, len(self.transformer.h)
                ) is not None
            )
            self.transformer.h[index] = (
                torch.compile(block, dynamic=True)
                if dynamic else torch.compile(block)
            )

    def compile_blocks(self):
        """Compile the architecture-specific middle blocks."""
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 256
        self._compile_middle()
