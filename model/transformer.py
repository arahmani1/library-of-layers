import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention

from .routers import _biased_topk
from .config import validate_paper_normalization


# Eager FlexAttention materializes the score matrix. One dynamic compiled
# kernel handles the variable packed lengths used by routed experts.
flex_attention = torch.compile(flex_attention, dynamic=True)


def _grouped_mm(inputs, weights, offsets):
    """Use the native PyTorch grouped GEMM for paper GPU execution."""
    return torch._grouped_mm(inputs, weights, offs=offsets)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        inverse_rms = torch.rsqrt(
            x_float.pow(2).mean(-1, keepdim=True) + self.eps
        )
        output = x_float * inverse_rms
        output = output * self.weight.float()
        return output.to(input_dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    frequencies = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)
    )
    positions = torch.arange(end, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    return torch.view_as_real(
        torch.polar(torch.ones_like(angles), angles)
    )


def _rotary_frequencies(freqs_cis, values, position_ids):
    if position_ids is None:
        sequence_length = values.shape[1]
        return freqs_cis[:sequence_length].view(
            1, sequence_length, 1, values.shape[-1] // 2, 2
        )
    return freqs_cis[position_ids].unsqueeze(2)


def apply_rotary_emb_reference(xq, xk, freqs_cis, position_ids=None):
    """Readable complex reference for the optimized real-pair rotation."""
    frequencies = torch.view_as_complex(
        _rotary_frequencies(freqs_cis, xq, position_ids)
    )

    def rotate(values):
        values_complex = torch.view_as_complex(
            values.float().reshape(*values.shape[:-1], -1, 2)
        )
        return torch.view_as_real(
            values_complex * frequencies
        ).flatten(3).type_as(values)

    return rotate(xq), rotate(xk)


def apply_rotary_emb(xq, xk, freqs_cis, position_ids=None):
    """Apply adjacent-pair RoPE using compiler-friendly real arithmetic."""
    frequencies = _rotary_frequencies(freqs_cis, xq, position_ids)
    cosine = frequencies[..., 0]
    sine = frequencies[..., 1]

    def rotate(values):
        pairs = values.float().reshape(*values.shape[:-1], -1, 2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        return torch.stack((
            even * cosine - odd * sine,
            even * sine + odd * cosine,
        ), dim=-1).flatten(3).type_as(values)

    return rotate(xq), rotate(xk)


class CausalSelfAttentionNeo(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_paper_normalization(config)
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=False
        )
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self, x, freqs_cis, position_ids=None, block_mask=None
    ):
        batch_size, sequence_length, embedding_dim = x.size()
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        shape = (
            batch_size,
            sequence_length,
            self.n_head,
            self.head_dim,
        )
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_emb(
            query, key, freqs_cis, position_ids=position_ids
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if block_mask is None:
            output = F.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
        else:
            output = flex_attention(
                query, key, value, block_mask=block_mask
            )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_dim
        )
        return self.c_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * (4 * config.n_embd) / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)
        self.gate_up_proj = nn.Linear(
            config.n_embd, 2 * hidden_dim, bias=False
        )
        self.down_proj = nn.Linear(
            hidden_dim, config.n_embd, bias=False
        )

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def _moe_routing_stats(indices, weights, n_experts):
    loads = torch.bincount(
        indices.reshape(-1), minlength=n_experts
    ).float()
    importance = torch.zeros(
        n_experts, device=indices.device, dtype=torch.float32
    )
    importance.scatter_add_(
        0, indices.reshape(-1), weights.reshape(-1)
    )
    return loads, importance


def _dispatch_moe_swiglu(
    flat, indices, weights, gate_up_proj, down_proj,
):
    top_k = indices.size(-1)
    token_ids = torch.arange(
        flat.size(0), device=flat.device
    ).repeat_interleave(top_k)
    expert_ids = indices.reshape(-1)
    gate_weights = weights.reshape(-1).to(flat.dtype)
    order = torch.argsort(expert_ids, stable=True)
    sorted_tokens = token_ids[order]
    sorted_experts = expert_ids[order]
    expert_input = flat[sorted_tokens]

    if flat.device.type == "cuda":
        offsets = torch.bincount(
            sorted_experts, minlength=gate_up_proj.size(0)
        ).cumsum(0).to(torch.int32)
        compute_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda") else flat.dtype
        )
        gate_up = _grouped_mm(
            expert_input.to(compute_dtype),
            gate_up_proj.to(compute_dtype),
            offsets,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = _grouped_mm(
            F.silu(gate) * up,
            down_proj.to(compute_dtype),
            offsets,
        ).to(flat.dtype)
    else:
        pieces = []
        for expert in range(gate_up_proj.size(0)):
            current = expert_input[sorted_experts == expert]
            gate, up = (
                current @ gate_up_proj[expert]
            ).chunk(2, dim=-1)
            pieces.append(
                (F.silu(gate) * up) @ down_proj[expert]
            )
        expert_output = torch.cat(pieces, dim=0)

    output = torch.zeros_like(flat)
    output.index_add_(
        0, sorted_tokens,
        expert_output * gate_weights[order].unsqueeze(-1),
    )
    return output


class MoESwiGLU(nn.Module):
    """Dropless token-routed SwiGLU with stacked expert weights."""

    def __init__(self, config):
        super().__init__()
        self.n_experts = int(config.moe_ffn_n_experts)
        self.top_k = int(config.moe_ffn_top_k)
        dense_hidden = int(2 * (4 * config.n_embd) / 3)
        dense_hidden = 256 * ((dense_hidden + 255) // 256)
        divided_hidden = (dense_hidden + self.top_k - 1) // self.top_k
        self.hidden_dim = 256 * ((divided_hidden + 255) // 256)
        if config.moe_ffn_hidden_dim is not None:
            self.hidden_dim = config.moe_ffn_hidden_dim
        self.router = nn.Linear(config.n_embd, self.n_experts, bias=False)
        self.gate_up_proj = nn.Parameter(torch.empty(
            self.n_experts, config.n_embd, 2 * self.hidden_dim,
        ))
        self.down_proj = nn.Parameter(torch.empty(
            self.n_experts, self.hidden_dim, config.n_embd,
        ))
        self.gate_up_proj._muon_matrix_batch = True
        self.down_proj._muon_matrix_batch = True
        self.register_buffer(
            "expert_bias", torch.zeros(self.n_experts, dtype=torch.float32)
        )
        self.last_loads = None
        self.last_importance = None
        self.last_routes = None
        self._collect_stats = False
        self._collect_routes = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        for expert in range(self.n_experts):
            nn.init.kaiming_uniform_(
                self.gate_up_proj[expert].transpose(0, 1),
                a=math.sqrt(5),
            )
            nn.init.kaiming_uniform_(
                self.down_proj[expert].transpose(0, 1),
                a=math.sqrt(5),
            )

    def begin_stats(self, *, capture_routes=False):
        self.last_loads = None
        self.last_importance = None
        self.last_routes = [] if capture_routes else None
        self._collect_stats = True
        self._collect_routes = bool(capture_routes)

    def end_stats(self):
        self._collect_stats = False
        self._collect_routes = False
        return self.last_loads, self.last_importance

    def record_stats(self, loads, importance):
        if not self._collect_stats:
            return
        if self.last_loads is None:
            self.last_loads = loads
            self.last_importance = importance
        else:
            self.last_loads = self.last_loads + loads
            self.last_importance = self.last_importance + importance

    def record_routes(self, active, indices, total_tokens):
        if not self._collect_routes:
            return
        routes = torch.full(
            (total_tokens, self.top_k),
            -1,
            device=indices.device,
            dtype=torch.long,
        )
        routes.index_copy_(0, active, indices.detach())
        self.last_routes.append(routes)

    def _route(self, flat):
        with torch.autocast(device_type=flat.device.type, enabled=False):
            scores = torch.sigmoid(
                F.linear(flat.float(), self.router.weight.float())
            )
            weights, indices = _biased_topk(
                scores, self.top_k, self.expert_bias
            )
            weights = weights / weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
        return indices, weights

    def _experts(self, flat, indices, weights):
        return _dispatch_moe_swiglu(
            flat, indices, weights,
            self.gate_up_proj, self.down_proj,
        )

    def forward(self, x, token_mask=None):
        flat = x.reshape(-1, x.size(-1))
        active = (
            torch.arange(flat.size(0), device=x.device)
            if token_mask is None
            else token_mask.reshape(-1).nonzero(as_tuple=True)[0]
        )
        active_input = flat[active]
        if active_input.size(0) == 0:
            if self._collect_stats:
                zero = torch.zeros(
                    self.n_experts, device=x.device, dtype=torch.float32
                )
                self.record_stats(zero, zero)
            self.record_routes(
                active,
                torch.empty(
                    (0, self.top_k), device=x.device, dtype=torch.long
                ),
                flat.size(0),
            )
            return torch.zeros_like(x)

        indices, weights = self._route(active_input)
        self.record_routes(active, indices, flat.size(0))
        if self._collect_stats:
            self.record_stats(*_moe_routing_stats(
                indices, weights, self.n_experts
            ))
        active_output = self._experts(active_input, indices, weights)
        output = torch.zeros_like(flat)
        output.index_copy_(0, active, active_output)
        return output.view_as(x)


class TransformerBlockNeo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttentionNeo(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = (
            MoESwiGLU(config) if config.moe_ffn else SwiGLU(config)
        )

    def attention_residual(
        self, x, freqs_cis, position_ids=None, block_mask=None,
        depth_film=None,
    ):
        normalized = self.ln_1(x)
        if depth_film is not None:
            gamma, beta = depth_film
            normalized = (
                normalized * gamma[0].to(normalized.dtype)
                + beta[0].to(normalized.dtype)
            )
        return self.attn(
            normalized, freqs_cis,
            position_ids=position_ids, block_mask=block_mask,
        )

    def ffn_residual(self, x, ffn_token_mask=None, depth_film=None):
        normalized = self.ln_2(x)
        if depth_film is not None:
            gamma, beta = depth_film
            normalized = (
                normalized * gamma[1].to(normalized.dtype)
                + beta[1].to(normalized.dtype)
            )
        if isinstance(self.mlp, MoESwiGLU):
            return self.mlp(normalized, token_mask=ffn_token_mask)
        return self.mlp(normalized)

    def forward(
        self, x, freqs_cis, position_ids=None, block_mask=None,
        ffn_token_mask=None, depth_film=None,
    ):
        x = x + self.attention_residual(
            x, freqs_cis,
            position_ids=position_ids, block_mask=block_mask,
            depth_film=depth_film,
        )
        return x + self.ffn_residual(
            x, ffn_token_mask, depth_film=depth_film
        )


class SandwichBlockNeo(TransformerBlockNeo):
    """Pre- and post-normalized block from the recurrent-depth paper.

    ``x = n2(x + attn(n1(x)))`` then ``x = n4(x + mlp(n3(x)))``: the residual
    stream is re-normalized after every sublayer, which keeps a weight-shared
    block stable when it is unrolled to arbitrary depth.
    """

    def __init__(self, config):
        super().__init__(config)
        self.ln_1_post = RMSNorm(config.n_embd)
        self.ln_2_post = RMSNorm(config.n_embd)

    def forward(
        self, x, freqs_cis, position_ids=None, block_mask=None,
        ffn_token_mask=None,
    ):
        x = self.ln_1_post(x + self.attention_residual(
            x, freqs_cis,
            position_ids=position_ids, block_mask=block_mask,
        ))
        return self.ln_2_post(x + self.ffn_residual(x, ffn_token_mask))
