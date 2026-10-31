import torch
import torch.nn as nn
import torch.nn.functional as F


def _biased_topk(routing_scores, k, routing_bias=None):
    """Select with an optional bias while preserving unbiased gate values."""
    selection_scores = routing_scores
    if routing_bias is not None:
        selection_scores = selection_scores + routing_bias.to(
            device=routing_scores.device,
            dtype=routing_scores.dtype,
        )
    indices = torch.topk(selection_scores, k, dim=-1).indices
    return routing_scores.gather(-1, indices), indices


class RouterInputEncoder(nn.Module):
    """Build the document-bounded prefix mean for window routing."""

    def __init__(self, config):
        super().__init__()
        self.width = int(config.n_embd)
        first_window_routing = str(getattr(config, "first_window_routing", "")).lower()
        if first_window_routing != "learned_token_depth":
            raise ValueError("first_window_routing must be 'learned_token_depth', "
                             f"got {first_window_routing!r}.")
        # Depth d pools row d with its document's first-token state. Untied
        # routers share this table through LoLWindow after construction.
        self.first_window_query = nn.Parameter(torch.empty(
            int(config.recurrent_depth), self.width,
        ))
        nn.init.normal_(self.first_window_query, mean=0.0, std=0.02)

    def build_features(
        self, hidden, *, global_window_id, is_first_window_of_doc,
        window_start_token_idx,
    ):
        flat_hidden = hidden.reshape(-1, hidden.size(-1)).float()
        group = global_window_id.reshape(-1)
        windows = int(is_first_window_of_doc.numel())
        # Canonical window ids are consecutive runs in flattened token order.
        # CUDA index_add_ uses unordered floating atomics: identical inputs can
        # produce different router probabilities before any training update.
        window_count = torch.bincount(group, minlength=windows)
        window_sum = torch.segment_reduce(
            flat_hidden, "sum", lengths=window_count,
        )
        window_features = (
            window_sum / window_count.clamp_min(1).float().unsqueeze(-1)
        )
        count = window_features.size(0)
        idx = torch.arange(count, device=window_features.device)
        marker = torch.where(
            is_first_window_of_doc,
            idx,
            torch.full_like(idx, -1),
        )
        doc_start, _ = marker.cummax(0)
        cumulative = torch.cat((
            window_features.new_zeros(1, window_features.size(-1)),
            window_features.cumsum(0),
        ))
        prefix_count = idx - doc_start
        prefix_sum = cumulative[idx] - cumulative[doc_start]
        denominator = prefix_count.clamp_min(1).unsqueeze(-1).float()
        mean = prefix_sum / denominator
        # First windows have no completed history; use only their first token.
        no_history = flat_hidden[window_start_token_idx]
        return torch.where(
            (prefix_count > 0).unsqueeze(-1), mean, no_history,
        )

    def first_window_vector(self, depth):
        """Return the trained query for this absolute recurrent depth."""
        index = int(depth)
        if not 0 <= index < self.first_window_query.size(0):
            raise ValueError(
                f"Depth {index} has no first-window query; the table holds "
                f"{self.first_window_query.size(0)} recurrent depths."
            )
        return self.first_window_query[index]

    def forward(self, hidden, *, depth, **kwargs):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            encoded = self.build_features(hidden, **kwargs)
            query = self.first_window_vector(depth).float()
            # First-window prefix_mean falls back only to the first token.
            query = 0.5 * (encoded + query)
            encoded = torch.where(
                kwargs["is_first_window_of_doc"].unsqueeze(-1), query, encoded,
            )
        return encoded


class NoisyRouter(nn.Module):
    """Top-k router with learned, input-dependent training noise.

    Noise encourages exploration during training and is disabled during
    evaluation. Projection, noise generation, and softmax run in float32 because
    ``softplus`` can overflow under lower-precision autocast. The predicted
    standard deviation is bounded to keep noise magnitudes stable.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_experts)
        self.noise = nn.Linear(config.n_experts, config.n_experts)
        self.softplus = nn.Softplus()
        self.max_noise_std = 10.0  # Upper bound on noise magnitude.

        # Start with zero predicted noise and learn how much to add.
        nn.init.zeros_(self.noise.weight)
        nn.init.zeros_(self.noise.bias)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.c_fc(x.float())
            if self.training:
                noise_std = self.softplus(self.noise(logits)).clamp(
                    max=self.max_noise_std
                )
                logits = logits + noise_std * torch.randn_like(logits)
            return F.softmax(logits, dim=-1)


class OuterRouter(nn.Module):
    """Shared causal feature encoder plus one expert-affinity head."""

    def __init__(self, config):
        super().__init__()
        self.inputs = RouterInputEncoder(config)
        self.head = NoisyRouter(config)

    def forward(self, hidden, **kwargs):
        return self.head(self.inputs(hidden, **kwargs))
