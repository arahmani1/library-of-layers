import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention_strategies import (
    execute_packed_window_block,
)
from .transformer import (
    RMSNorm,
    TransformerBlockNeo,
    _grouped_mm,
    apply_rotary_emb,
)
from .routers import OuterRouter, _biased_topk
from .config import RecurrentInputInjection
from .vanillaneo import MiddleResult, VanillaGPTNeo


@dataclass
class RouteDecision:
    indices: torch.Tensor
    weights: torch.Tensor
    aux_loss: torch.Tensor
    scores: torch.Tensor | None = None
    stats: dict[str, torch.Tensor] | None = None


class DiagonalRecurrentInputInjection(nn.Module):
    """Stable diagonal prelude injection from Parcae."""

    def __init__(self, dim):
        super().__init__()
        self.prelude_norm = RMSNorm(dim)
        self.A_log = nn.Parameter(torch.zeros(dim))
        target_decay = math.sqrt(1.0 / 5.0)
        initial_dt = -math.log(target_decay)
        inverse_softplus_dt = math.log(math.expm1(initial_dt))
        self.dt_bias = nn.Parameter(
            torch.full((dim,), inverse_softplus_dt)
        )
        self.B = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.B.weight)
        for parameter in self.parameters():
            parameter._force_adamw = True
            parameter._no_weight_decay = True

    def prepare(self, prelude_output):
        return self.B(self.prelude_norm(prelude_output))

    def forward(self, state, projected_prelude):
        dt = F.softplus(self.dt_bias)
        decay = torch.exp(-dt * torch.exp(self.A_log))
        return state * decay + dt * projected_prelude


class LoLWindow(VanillaGPTNeo):

    def __init__(self, config):
        if type(self) is LoLWindow:
            raise TypeError("LoLWindow is an internal base; construct LoLDynamicWindow.")
        if config.recurrent_input_injection is RecurrentInputInjection.ADAPTER:
            raise ValueError(
                "adapter recurrent input injection requires "
                "arch=huginn."
            )
        if config.moe_ffn and config.moe_ffn_scope != "prelude_coda":
            raise ValueError("Paper LoL inner MoE requires moe_ffn_scope='prelude_coda'.")
        super().__init__(config)
        self.n_experts = int(config.n_experts)
        self.recurrent_depth = int(config.recurrent_depth)
        self.uses_depth_expert_film = bool(
            getattr(config, "depth_expert_film", False)
        )
        if self.uses_depth_expert_film:
            self.depth_expert_film_gamma = nn.Parameter(torch.ones(
                self.recurrent_depth, 2, config.n_embd
            ))
            self.depth_expert_film_beta = nn.Parameter(torch.zeros(
                self.recurrent_depth, 2, config.n_embd
            ))
            self.depth_expert_film_gamma._no_weight_decay = True
            self.depth_expert_film_beta._no_weight_decay = True

        self.weight_tie_router = bool(config.weight_tie_router)
        router_count = 1 if self.weight_tie_router else self.recurrent_depth
        self.outer_routers = nn.ModuleList([
            OuterRouter(config)
            for _ in range(router_count)
        ])
        # learned_token_depth indexes one query table by
        # absolute depth. Untied routers each build their own table, so tie
        # them to a single one: depth d owns row d, rather than keeping private
        # copies whose other rows nothing trains. The table therefore costs
        # recurrent_depth x n_embd whether or not the router is tied.
        first_encoder = self.outer_routers[0].inputs
        if not self.weight_tie_router:
            for router in self.outer_routers[1:]:
                router.inputs.first_window_query = (
                    first_encoder.first_window_query
                )

        self.recurrent_input_injection = None
        if (
            config.recurrent_input_injection
            is RecurrentInputInjection.DIAGONAL
        ):
            self.recurrent_input_injection = DiagonalRecurrentInputInjection(config.n_embd)
        self.grouped_window_dispatch = bool(
            getattr(config, "grouped_window_dispatch", False)
        )
        self._head_dim = config.n_embd // config.n_head

    def _depth_expert_film(self, depth):
        if not self.uses_depth_expert_film:
            return None
        if depth is None:
            raise ValueError("depth is required when depth_expert_film is enabled.")
        return (
            self.depth_expert_film_gamma[int(depth)],
            self.depth_expert_film_beta[int(depth)],
        )

    def _build_middle(self, config):
        middle_config = (
            config
            if not config.moe_ffn
            else replace(config, moe_ffn=False)
        )
        return {
            "experts": nn.ModuleList(
                TransformerBlockNeo(middle_config)
                for _ in range(config.n_experts)
            )
        }

    def _build_middle_boundaries(self, config):
        self.prelude_blocks = nn.ModuleList(
            TransformerBlockNeo(config)
            for _ in range(config.n_prelude_layers)
        )
        self.coda_blocks = nn.ModuleList(
            TransformerBlockNeo(config)
            for _ in range(config.n_coda_layers)
        )
    def _prepare_recurrent_input(self, prelude_output):
        if self.recurrent_input_injection is None:
            return None
        return self.recurrent_input_injection.prepare(prelude_output)

    def _inject_recurrent_input(self, state, prepared_input):
        if self.recurrent_input_injection is None:
            return state
        return self.recurrent_input_injection(state, prepared_input)

    def _apply_prelude(self, hidden, context):
        """Run every full-sequence block before the routed middle."""
        for block in self.prelude_blocks:
            hidden = self._apply_block(block, hidden, context)
        return hidden

    def _apply_coda(self, hidden, context):
        """Run every full-sequence block after the routed middle."""
        for block in self.coda_blocks:
            hidden = self._apply_block(block, hidden, context)
        return hidden

    # ------------------------------------------------------------------
    # torch.compile entry point
    # ------------------------------------------------------------------
    def _compile_middle(self):
        for blocks in (
            self.prelude_blocks, self.coda_blocks,
        ):
            for index in range(len(blocks)):
                blocks[index] = torch.compile(blocks[index])
        if self.grouped_window_dispatch:
            # Window counts vary with document boundaries and recurrent depth,
            # while batch, sequence, width, heads and window size stay fixed.
            # Let automatic dynamism specialize those stable dimensions and
            # promote only dimensions that actually change.
            self._dispatch_windows_grouped = torch.compile(
                self._dispatch_windows_grouped
            )
        else:
            for i in range(self.n_experts):
                self.transformer.experts[i] = torch.compile(
                    self.transformer.experts[i]
                )

    def _router_for_depth(self, depth):
        return self.outer_routers[0 if self.weight_tie_router else int(depth)]


    def _select_router_scores(self, scores, depth):
        weights, indices = _biased_topk(
            scores, 1, self._router_bias(depth)
        )
        return indices.squeeze(-1), weights.squeeze(-1)


    def _stack_expert_weights(self, dtype, active_experts=None):
        experts = self.transformer.experts

        def selected(getter):
            values = [getter(expert) for expert in experts]
            if active_experts is not None:
                values = [
                    value if active else value.detach()
                    for value, active in zip(values, active_experts)
                ]
            return values

        def stack_linear(getter):
            return torch.stack(selected(getter)).transpose(
                1, 2
            ).contiguous().to(dtype)

        def stack_norm(getter):
            return torch.stack(selected(getter))

        packed = {
            "qkv": stack_linear(lambda e: e.attn.c_attn.weight),
            "proj": stack_linear(lambda e: e.attn.c_proj.weight),
            "ln1": stack_norm(lambda e: e.ln_1.weight),
            "ln2": stack_norm(lambda e: e.ln_2.weight),
        }
        packed["gate_up"] = stack_linear(
            lambda e: e.mlp.gate_up_proj.weight
        )
        packed["down"] = stack_linear(
            lambda e: e.mlp.down_proj.weight
        )
        packed["qn"] = stack_norm(lambda e: e.attn.q_norm.weight)
        packed["kn"] = stack_norm(lambda e: e.attn.k_norm.weight)
        return packed

    @staticmethod
    def _rms_grouped(x, weight, eps, dtype):
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        output = xf * rms
        output = output * weight.float()
        return output.to(dtype)

    def _dispatch_windows_grouped(
        self, x, global_window_id, expert_per_window,
        position_ids, doc_of_window, window_start_pos, n_windows_total,
        *, packed=None, restore_output=True,
        depth=None,
    ):
        del doc_of_window  # Shared dispatch signature with the reference path.
        D = x.size(-1)
        W, H = self.attention_strategy.pack_width, self.config.n_head
        Dh, G = self._head_dim, self.n_experts
        if packed is None:
            packed = self.attention_strategy.pack_layout(
                x, position_ids, global_window_id, window_start_pos,
                n_windows_total, expert_per_window,
            )
        packed_experts = expert_per_window[packed.window_permutation]
        counts = torch.bincount(packed_experts, minlength=G)

        token_expert = packed_experts.repeat_interleave(W)
        offsets = torch.cumsum(counts * W, 0).to(torch.int32)
        N = n_windows_total * W
        xs = packed.values.reshape(N, D)
        cdtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda") else xs.dtype
        )
        weights = getattr(self, "_grouped_weight_cache", None)
        if weights is None and not torch.is_grad_enabled():
            weights = getattr(self, "_grouped_no_grad_weight_cache", None)
        if weights is None:
            weights = self._stack_expert_weights(
                cdtype, counts.ne(0).tolist()
            )

        eps1 = self.transformer.experts[0].ln_1.eps
        h1 = self._rms_grouped(
            xs,
            weights["ln1"][token_expert],
            eps1, cdtype,
        )
        film = self._depth_expert_film(depth)
        if film is not None:
            gamma, beta = film
            h1 = h1 * gamma[0].to(h1.dtype) + beta[0].to(h1.dtype)
        qkv = _grouped_mm(h1, weights["qkv"], offsets=offsets)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(n_windows_total, W, H, Dh)
        k = k.view(n_windows_total, W, H, Dh)
        v = v.view(n_windows_total, W, H, Dh)
        epsq = self.transformer.experts[0].attn.q_norm.eps
        q = self._rms_grouped(
            q.reshape(N, H, Dh),
            weights["qn"][token_expert].unsqueeze(1),
            epsq, q.dtype,
        ).view(n_windows_total, W, H, Dh)
        k = self._rms_grouped(
            k.reshape(N, H, Dh),
            weights["kn"][token_expert].unsqueeze(1),
            epsq, k.dtype,
        ).view(n_windows_total, W, H, Dh)
        q, k = apply_rotary_emb(
            q, k, self.freqs_cis, position_ids=packed.position_ids
        )
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2), is_causal=True,
        ).transpose(1, 2).contiguous().view(N, D)
        x1 = xs + _grouped_mm(
            y, weights["proj"], offsets=offsets
        )
        eps2 = self.transformer.experts[0].ln_2.eps
        h2 = self._rms_grouped(
            x1,
            weights["ln2"][token_expert],
            eps2, cdtype,
        )
        if film is not None:
            gamma, beta = film
            h2 = h2 * gamma[1].to(h2.dtype) + beta[1].to(h2.dtype)

        gate, up = _grouped_mm(
            h2, weights["gate_up"], offsets=offsets
        ).chunk(2, dim=-1)
        x2 = x1 + _grouped_mm(
            F.silu(gate) * up,
            weights["down"], offsets=offsets,
        )

        output = x2.view(n_windows_total, W, D)
        return packed.restore(output) if restore_output else output

    def _combine_expert_output(
        self, x, selected_output, global_window_id, weight_per_window,
    ):
        """Scale the selected expert residual by its raw router probability."""
        B, T, D = x.shape
        flat_window_id = global_window_id.reshape(B * T)
        residual = selected_output.reshape(B * T, D) - x.reshape(B * T, D)
        selected_weight = weight_per_window[flat_window_id].to(residual.dtype)
        combined = residual * selected_weight.unsqueeze(-1)
        combined = x.reshape(B * T, D) + combined
        return combined.view(B, T, D)

    def _dispatch_windows(
        self, x, global_window_id, expert_per_window, weight_per_window,
        position_ids, doc_of_window, window_start_pos, n_windows_total,
        *, gradient_window_mask=None, depth=None,
    ):
        if gradient_window_mask is not None:
            # Pack once in (gradient policy, expert) order. Each partition is
            # contiguous and already expert-sorted, including partial windows.
            sort_key = (
                gradient_window_mask.long() * self.n_experts + expert_per_window
            )
            packed = self.attention_strategy.pack_layout(
                x, position_ids, global_window_id, window_start_pos,
                n_windows_total, sort_key,
            )
            no_grad_count = int((~gradient_window_mask).sum())
            outputs = []
            for start, stop, gradients in (
                (0, no_grad_count, False),
                (no_grad_count, n_windows_total, True),
            ):
                if start == stop:
                    continue
                with torch.set_grad_enabled(
                    torch.is_grad_enabled() and gradients
                ):
                    values = packed.values[start:stop]
                    if gradients and no_grad_count:
                        # Autograd saves expert inputs. A view would retain
                        # storage for every later window through backward.
                        values = values.clone()
                    # Keep the full restore map here; only the outer dispatch
                    # restores tokens after concatenating packed kernel outputs.
                    subset = replace(
                        packed,
                        values=values,
                        position_ids=packed.position_ids[start:stop],
                        valid=packed.valid[start:stop],
                        window_permutation=packed.window_permutation[start:stop],
                    )
                    arguments = (
                        x, global_window_id, expert_per_window, position_ids,
                        doc_of_window, window_start_pos, stop - start,
                    )
                    if self.grouped_window_dispatch:
                        output = self._dispatch_windows_grouped(
                            *arguments, packed=subset, restore_output=False,
                            depth=depth,
                        )
                    else:
                        output = self._dispatch_windows_padded(
                            *arguments, packed=subset, restore_output=False,
                            depth=depth,
                        )
                outputs.append(output)
            selected_output = packed.restore(torch.cat(outputs))
        elif self.grouped_window_dispatch:
            selected_output = self._dispatch_windows_grouped(
                x, global_window_id, expert_per_window,
                position_ids, doc_of_window, window_start_pos,
                n_windows_total, depth=depth,
            )
        else:
            selected_output = self._dispatch_windows_padded(
                x, global_window_id, expert_per_window,
                position_ids, doc_of_window, window_start_pos,
                n_windows_total, depth=depth,
            )
        return self._combine_expert_output(
            x, selected_output, global_window_id, weight_per_window,
        )

    # ------------------------------------------------------------------
    # Expert dispatch (padded (K_e, W, C) packing — one call per expert)
    # ------------------------------------------------------------------
    def _dispatch_windows_padded(
        self, x, global_window_id, expert_per_window,
        position_ids, doc_of_window, window_start_pos, n_windows_total,
        *, packed=None, restore_output=True, depth=None,
    ):
        B, T, D = x.shape
        W = self.attention_strategy.pack_width

        if packed is None:
            n_docs_upper = B * (T + 1)  # strict upper bound on doc_of_window+1
            sort_key = expert_per_window.long() * n_docs_upper + doc_of_window
            packed = self.attention_strategy.pack_layout(
                x, position_ids, global_window_id, window_start_pos,
                n_windows_total, sort_key,
            )
        counts = torch.bincount(
            expert_per_window[packed.window_permutation], minlength=self.n_experts
        )
        offsets = torch.cumsum(counts, 0) - counts
        packed_out = torch.empty_like(packed.values)

        counts_cpu = counts.tolist()
        offsets_cpu = offsets.tolist()
        film = self._depth_expert_film(depth)

        for expert, block in enumerate(self.transformer.experts):
            count = counts_cpu[expert]
            if count == 0:
                continue
            start = offsets_cpu[expert]
            end = start + count

            values = packed.values[start:end]
            positions = packed.position_ids[start:end]
            valid = packed.valid[start:end]

            output = execute_packed_window_block(
                block,
                values,
                positions,
                valid,
                self.freqs_cis,
                block_mask=None,
                depth_film=film,
            ).view(count, W, D)

            packed_out[start:end] = output

        return packed.restore(packed_out) if restore_output else packed_out

    # ------------------------------------------------------------------
    # Shared recurrent forward template
    # ------------------------------------------------------------------
    def _route_at_depth(
        self, hidden, metadata, depth,
        return_router_diagnostics=True,
        gradient_window_mask=None, aux_gradient_scale=None,
    ):
        scores = self._router_for_depth(depth)(
            hidden,
            global_window_id=metadata.global_window_id,
            is_first_window_of_doc=metadata.is_first_window_of_doc,
            window_start_token_idx=metadata.window_start_token_idx,
            depth=depth,
        )
        if gradient_window_mask is not None:
            scores = torch.where(gradient_window_mask[:, None], scores, scores.detach())
        unbiased_indices = scores.argmax(dim=-1)
        indices, weights = self._select_router_scores(
            scores, depth
        )
        auxiliary_scores = scores
        if aux_gradient_scale is not None:
            auxiliary_scores = scores.detach() + (scores - scores.detach()) * aux_gradient_scale[:, None]
        aux_loss, stats = self._router_statistics(
            auxiliary_scores, unbiased_indices, indices, metadata,
            return_router_diagnostics,
        )
        return RouteDecision(
            indices=indices,
            weights=weights,
            aux_loss=aux_loss,
            scores=scores,
            stats=stats,
        )


    def _dispatch_decision(
        self,
        hidden,
        context,
        metadata,
        decision,
        *,
        gradient_window_mask=None, depth=None,
    ):
        return self._dispatch_windows(
            hidden,
            metadata.global_window_id,
            decision.indices,
            decision.weights,
            context.position_ids,
            metadata.doc_of_window,
            metadata.window_start_pos,
            metadata.n_windows_total,
            gradient_window_mask=gradient_window_mask,
            depth=depth,
        )

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
        depth_limit = (
            self.recurrent_depth
            if num_steps is None
            else max(1, min(int(num_steps), self.recurrent_depth))
        )
        gradient_depth = (
            depth_limit if backprop_steps is None
            else max(1, min(int(backprop_steps), depth_limit))
        )
        no_grad_depth = depth_limit - gradient_depth
        if first_window_full_backprop and (
            self.config.recurrent_input_injection is not RecurrentInputInjection.DIAGONAL
        ):
            raise ValueError("first_window_full_backprop requires causal diagonal block routing")
        mixed_backprop = first_window_full_backprop and no_grad_depth > 0
        auxiliary_gradient_depth = depth_limit if mixed_backprop else gradient_depth
        hidden = self._apply_prelude(hidden, context)
        prepared_input = self._prepare_recurrent_input(hidden)
        aux_loss = torch.zeros(
            (), device=hidden.device, dtype=torch.float32
        )
        sequence_aux_loss = torch.zeros_like(aux_loss)
        gradient_aux_loss = torch.zeros_like(aux_loss)
        gradient_sequence_aux_loss = torch.zeros_like(aux_loss)
        collected: dict[str, list[torch.Tensor]] = {}
        gradient_collected: dict[str, list[torch.Tensor]] = {}
        if self.grouped_window_dispatch:
            compute_dtype = (
                torch.get_autocast_dtype("cuda")
                if torch.is_autocast_enabled("cuda") else hidden.dtype
            )
            self._grouped_no_grad_weight_cache = None
            if no_grad_depth == 0:
                self._grouped_weight_cache = self._stack_expert_weights(compute_dtype)
            else:
                # Detached weights are reusable across the entire prefix.
                # Differentiable stacks still include only routed experts, so
                # prefix-only later-window experts keep grad=None.
                self._grouped_weight_cache = None
                with torch.no_grad():
                    self._grouped_no_grad_weight_cache = self._stack_expert_weights(
                        compute_dtype
                    )

        first_metadata = None
        routing_token_paths = [] if return_router_trace else None
        forward_grad_enabled = torch.is_grad_enabled()
        try:
            for depth in range(depth_limit):
                gradient_bearing = depth >= no_grad_depth
                prefix_first_only = mixed_backprop and not gradient_bearing
                track_gradients = gradient_bearing or mixed_backprop
                if depth == no_grad_depth and self.grouped_window_dispatch:
                    self._grouped_no_grad_weight_cache = None
                metadata = self.attention_strategy.metadata(
                    context.doc_id, context.position_ids, depth,
                    input_valid=context.input_valid,
                )
                first_tokens = (
                    metadata.is_first_window_of_doc[
                        metadata.global_window_id
                    ].unsqueeze(-1)
                    if mixed_backprop else None
                )
                if mixed_backprop and depth <= no_grad_depth:
                    hidden = torch.where(first_tokens, hidden, hidden.detach())
                elif no_grad_depth and depth == no_grad_depth:
                    hidden = hidden.detach()
                with torch.set_grad_enabled(
                    forward_grad_enabled and track_gradients
                ):
                    hidden = self._inject_recurrent_input(
                        hidden, prepared_input
                    )
                    if prefix_first_only:
                        hidden = torch.where(first_tokens, hidden, hidden.detach())
                    if first_metadata is None:
                        first_metadata = metadata
                    gradient_window_mask = (
                        metadata.is_first_window_of_doc
                        if prefix_first_only else None
                    )
                    route_gradient_kwargs = {}
                    if mixed_backprop:
                        route_gradient_kwargs = {
                            "gradient_window_mask": gradient_window_mask,
                            "aux_gradient_scale": torch.where(
                                metadata.is_first_window_of_doc,
                                1.0, depth_limit / gradient_depth,
                            ),
                        }
                    decision = self._route_at_depth(
                        hidden,
                        metadata,
                        depth,
                        **route_gradient_kwargs,
                        return_router_diagnostics=return_router_diagnostics,
                    )
                    if routing_token_paths is not None:
                        routing_token_paths.append(decision.indices[metadata.global_window_id].detach())
                    aux_loss = aux_loss + decision.aux_loss
                    if track_gradients:
                        gradient_aux_loss = (
                            gradient_aux_loss + decision.aux_loss
                        )
                    if decision.stats is not None:
                        sequence_aux_loss = (
                            sequence_aux_loss
                            + decision.stats["sequence_aux_loss"]
                        )
                        if track_gradients:
                            gradient_sequence_aux_loss = (
                                gradient_sequence_aux_loss
                                + decision.stats["sequence_aux_loss"]
                            )
                        for name, value in decision.stats.items():
                            if name != "sequence_aux_loss":
                                collected.setdefault(name, []).append(value)
                                if (
                                    track_gradients
                                    and name in {"p_sum", "count"}
                                ):
                                    gradient_collected.setdefault(
                                        name, []
                                    ).append(value)
                    hidden = self._dispatch_decision(
                        hidden, context, metadata, decision,
                        depth=depth,
                        **({"gradient_window_mask": gradient_window_mask}
                           if gradient_window_mask is not None and forward_grad_enabled else {}),
                    )
                    if prefix_first_only:
                        hidden = torch.where(first_tokens, hidden, hidden.detach())
        finally:
            if self.grouped_window_dispatch:
                self._grouped_weight_cache = None
                self._grouped_no_grad_weight_cache = None

        executed_aux_loss = aux_loss / depth_limit
        gradient_aux_loss = gradient_aux_loss / auxiliary_gradient_depth
        aux_loss = (
            executed_aux_loss.detach()
            + gradient_aux_loss
            - gradient_aux_loss.detach()
        )
        hidden = self._apply_coda(hidden, context)
        loss_mask = torch.ones_like(context.idx, dtype=torch.bool)
        router_stats = None
        if return_router_stats and collected:
            router_stats = {
                name: torch.stack(values, dim=0)
                for name, values in collected.items()
            }
            if "p_sum" in router_stats:
                router_stats["p_sum"] = router_stats["p_sum"].detach()
            router_stats["gradient_p_sum"] = torch.stack(
                gradient_collected["p_sum"], dim=0
            )
            router_stats["gradient_count"] = torch.stack(
                gradient_collected["count"], dim=0
            )
            executed_sequence_aux_loss = sequence_aux_loss / depth_limit
            gradient_sequence_aux_loss = (
                gradient_sequence_aux_loss / auxiliary_gradient_depth
            )
            router_stats["sequence_aux_loss"] = (
                executed_sequence_aux_loss.detach()
                + gradient_sequence_aux_loss
                - gradient_sequence_aux_loss.detach()
            )
        if routing_token_paths is not None:
            if router_stats is None:
                router_stats = {}
            router_stats.update({
                "routing_token_paths": torch.stack(
                    routing_token_paths, dim=2
                ),
                "routing_depth0_window_ids": (
                    first_metadata.global_window_id.detach()
                ),
            })
        return MiddleResult(
            hidden=hidden,
            aux_loss=aux_loss,
            loss_mask=loss_mask,
            router_stats=router_stats,
        )

    def _format_output(self, lm_output, middle, return_router_stats):
        output = (lm_output, middle.aux_loss, middle.loss_mask)
        if return_router_stats and middle.router_stats is not None:
            return (*output, middle.router_stats)
        return output

    # ------------------------------------------------------------------
    # Expert tracing  (inference utility)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def trace_experts(
        self,
        idx,
        doc_id=None,
        position_ids=None,
        input_valid=None,
        return_scores=False,
    ):
        context = self._prepare_context(idx, doc_id, position_ids, input_valid)
        embedding = self.transformer.wte(idx)
        x = self._apply_prelude(embedding, context)
        prepared_input = self._prepare_recurrent_input(x)
        traces = []
        for depth in range(self.recurrent_depth):
            x = self._inject_recurrent_input(x, prepared_input)
            metadata = self.attention_strategy.metadata(
                context.doc_id, context.position_ids, depth,
                input_valid=context.input_valid,
            )
            decision = self._route_at_depth(
                x,
                metadata,
                depth,
            )
            trace = (
                decision.indices.detach().clone(),
                metadata.global_window_id.clone(),
            )
            if return_scores:
                trace += (decision.scores.detach().float().clone(),)
            traces.append(trace)
            x = self._dispatch_decision(
                x, context, metadata, decision, depth=depth
            )
        return self._format_expert_traces(traces)


    # ------------------------------------------------------------------
    # Inference (prompt-only router input)
    # ------------------------------------------------------------------
    @staticmethod
    def _win_in_doc_idx(is_first_window_of_doc, n_windows_total, device):
        arange_w = torch.arange(n_windows_total, device=device)
        doc_start_marker = torch.where(
            is_first_window_of_doc, arange_w, torch.full_like(arange_w, -1),
        )
        doc_start, _ = doc_start_marker.cummax(0)
        return arange_w - doc_start

    @staticmethod
    def _full_to_prompt_window_map(
        full_dow, full_win_in_doc, prompt_dow, prompt_n,
    ):

        left = torch.searchsorted(prompt_dow, full_dow)
        right = torch.searchsorted(prompt_dow, full_dow + 1)
        count = right - left
        safe_count = count.clamp(min=1)
        offset = full_win_in_doc.clamp(max=safe_count - 1)
        mapped = left + offset

        has_prompt_document = count > 0
        fallback = torch.full_like(mapped, prompt_n - 1)
        return (
            torch.where(has_prompt_document, mapped, fallback),
            has_prompt_document,
        )

    @torch.no_grad()
    def inference(
        self,
        prompt_idx,
        prompt_answer_idx,
        prompt_doc_id=None,
        prompt_position_ids=None,
        doc_id=None,
        position_ids=None,
        recurrent_depth=None,
    ):
        self._validate_inference_inputs(prompt_idx, prompt_answer_idx)
        depth_limit = self._resolve_inference_depth(recurrent_depth)
        prompt_idx = prompt_idx.to(prompt_answer_idx.device)
        full_context = self._prepare_context(
            prompt_answer_idx, doc_id, position_ids
        )
        prompt_context = self._prepare_context(
            prompt_idx, prompt_doc_id, prompt_position_ids
        )
        shared_k = max(
            prompt_answer_idx.size(1), prompt_idx.size(1)
        ) + 1
        full_embedding = self.transformer.wte(prompt_answer_idx)
        prompt_embedding = self.transformer.wte(prompt_idx)
        x_full = self._apply_prelude(full_embedding, full_context)
        x_prompt = self._apply_prelude(prompt_embedding, prompt_context)
        full_prepared = self._prepare_recurrent_input(x_full)
        prompt_prepared = self._prepare_recurrent_input(x_prompt)
        aux_loss = torch.zeros(
            (), device=prompt_answer_idx.device, dtype=torch.float32
        )
        for depth in range(depth_limit):
            x_full = self._inject_recurrent_input(
                x_full, full_prepared
            )
            x_prompt = self._inject_recurrent_input(
                x_prompt, prompt_prepared
            )
            full_metadata = self.attention_strategy.metadata(
                full_context.doc_id,
                full_context.position_ids,
                depth,
                document_key_space=shared_k,
            )
            prompt_metadata = self.attention_strategy.metadata(
                prompt_context.doc_id,
                prompt_context.position_ids,
                depth,
                document_key_space=shared_k,
            )
            prompt_decision = self._route_at_depth(
                x_prompt,
                prompt_metadata,
                depth,
            )
            aux_loss = aux_loss + prompt_decision.aux_loss
            full_decision = self._map_prompt_decision(
                prompt_decision,
                prompt_metadata,
                full_metadata,
            )
            x_prompt = self._dispatch_decision(
                x_prompt, prompt_context, prompt_metadata,
                prompt_decision, depth=depth,
            )
            x_full = self._dispatch_decision(
                x_full, full_context, full_metadata, full_decision,
                depth=depth,
            )
        if depth_limit:
            aux_loss = aux_loss / depth_limit
        x_full = self._apply_coda(x_full, full_context)
        return self.lm_head(self.transformer.ln_f(x_full)), aux_loss

    @staticmethod
    def _validate_inference_inputs(prompt_idx, prompt_answer_idx):
        if prompt_idx.dim() != 2 or prompt_answer_idx.dim() != 2:
            raise ValueError(
                "Expected prompt_idx and prompt_answer_idx to be rank-2 "
                "tensors (B, T)."
            )
        if prompt_idx.size(0) != prompt_answer_idx.size(0):
            raise ValueError(
                "Batch mismatch: prompt_idx and prompt_answer_idx must "
                "share dim 0."
            )
        if prompt_idx.size(1) == 0:
            raise ValueError(
                "prompt_idx must contain at least one token for routing."
            )
        if prompt_answer_idx.size(1) == 0:
            raise ValueError("prompt_answer_idx must contain at least one token.")

    def _resolve_inference_depth(self, recurrent_depth):
        if recurrent_depth is None:
            return self.recurrent_depth
        depth_limit = int(recurrent_depth)
        if depth_limit < 0:
            raise ValueError("recurrent_depth must be >= 0.")
        return min(depth_limit, self.recurrent_depth)

    def _map_prompt_decision(
        self, prompt_decision, prompt_metadata, full_metadata,
    ):
        full_win_in_doc = self._win_in_doc_idx(
            full_metadata.is_first_window_of_doc,
            full_metadata.n_windows_total,
            full_metadata.global_window_id.device,
        )
        mapping, has_prompt = self._full_to_prompt_window_map(
            full_metadata.doc_of_window,
            full_win_in_doc,
            prompt_metadata.doc_of_window,
            prompt_metadata.n_windows_total,
        )
        indices = prompt_decision.indices[mapping]
        weights = prompt_decision.weights[mapping]
        scores = (
            None if prompt_decision.scores is None
            else prompt_decision.scores[mapping]
        )
        fallback_indices = torch.zeros_like(indices)
        fallback_weights = torch.ones_like(weights)
        return RouteDecision(
            indices=torch.where(
                has_prompt, indices, fallback_indices
            ),
            weights=torch.where(
                has_prompt, weights, fallback_weights
            ),
            aux_loss=prompt_decision.aux_loss,
            scores=scores,
        )





    @torch.no_grad()
    def inference_fixed_path(
        self, prompt_idx, prompt_answer_idx, fixed_path,
        recurrent_depth=None, doc_id=None, position_ids=None, *, fixed_weights,
    ):
        """Force validation-derived experts and residual multipliers causally."""
        self._validate_inference_inputs(prompt_idx, prompt_answer_idx)
        depth_limit = self._resolve_inference_depth(recurrent_depth)
        # One expert per depth, or a [windows, depths] table indexed by
        # the window's position in its document (last row repeats).
        path = torch.as_tensor(
            fixed_path, dtype=torch.long,
            device=prompt_answer_idx.device,
        )
        if path.dim() == 1:
            path = path.unsqueeze(0)
        if path.dim() != 2 or path.size(0) == 0 or path.size(1) != depth_limit:
            raise ValueError(
                f"fixed_path has shape {tuple(path.shape)}; expected "
                f"[depth_limit={depth_limit}] or "
                f"[windows, depth_limit={depth_limit}]."
            )
        if path.numel() and (
            int(path.min()) < 0 or int(path.max()) >= self.n_experts
        ):
            raise ValueError(
                f"fixed_path entries must be in [0, {self.n_experts})."
            )
        weight_table = torch.as_tensor(
            fixed_weights, dtype=torch.float32, device=prompt_answer_idx.device,
        )
        if weight_table.dim() == 1:
            weight_table = weight_table.unsqueeze(0)
        if weight_table.shape != path.shape or not bool(torch.isfinite(weight_table).all())\
                or bool(((weight_table < 0) | (weight_table > 1)).any()):
            raise ValueError("fixed_weights must match fixed_path and contain finite probabilities in [0, 1].")
        context = self._prepare_context(
            prompt_answer_idx, doc_id, position_ids
        )
        embedding = self.transformer.wte(prompt_answer_idx)
        hidden = self._apply_prelude(embedding, context)
        prepared = self._prepare_recurrent_input(hidden)
        zero = torch.zeros(
            (), device=hidden.device, dtype=torch.float32
        )
        for depth in range(depth_limit):
            hidden = self._inject_recurrent_input(hidden, prepared)
            metadata = self.attention_strategy.metadata(
                context.doc_id, context.position_ids, depth,
                input_valid=context.input_valid,
            )
            window_in_doc = self._win_in_doc_idx(
                metadata.is_first_window_of_doc,
                metadata.n_windows_total,
                hidden.device,
            )
            indices = path[
                window_in_doc.clamp(max=path.size(0) - 1), depth
            ]
            weights = weight_table[
                window_in_doc.clamp(max=path.size(0) - 1), depth
            ].to(hidden.dtype)
            decision = RouteDecision(
                indices=indices,
                weights=weights,
                aux_loss=zero,
            )
            hidden = self._dispatch_decision(
                hidden, context, metadata, decision, depth=depth
            )
        hidden = self._apply_coda(hidden, context)
        return self.lm_head(self.transformer.ln_f(hidden)), zero
