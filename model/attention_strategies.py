"""Central attention policy, compatibility, and window execution."""

from dataclasses import dataclass
from enum import Enum

import torch
from torch.nn.attention.flex_attention import create_block_mask


class AttentionStrategyName(str, Enum):
    FULL_CAUSAL = "full_causal"
    SLIDING_WINDOW = "sliding_window"
    SHIFTED_WINDOW = "shifted_window"


def parse_attention_strategy_name(value):
    """Return the canonical strategy name accepted by every subsystem."""
    return AttentionStrategyName(
        str(getattr(value, "value", value)).lower()
    )


_ALLOWED_STRATEGIES = {
    "vanilla_neo": {
        AttentionStrategyName.FULL_CAUSAL,
        AttentionStrategyName.SLIDING_WINDOW,
        AttentionStrategyName.SHIFTED_WINDOW,
    },
    "lol_dynamic_window": {AttentionStrategyName.SHIFTED_WINDOW},
    "huginn": {AttentionStrategyName.FULL_CAUSAL},
}

@dataclass(frozen=True)
class WindowMetadata:
    global_window_id: torch.Tensor
    is_first_window_of_doc: torch.Tensor
    global_doc_id: torch.Tensor
    doc_of_window: torch.Tensor
    window_start_pos: torch.Tensor
    window_start_token_idx: torch.Tensor
    n_windows_total: int
    active_window_mask: torch.Tensor

@dataclass(frozen=True)
class PackedWindows:
    values: torch.Tensor
    position_ids: torch.Tensor
    valid: torch.Tensor
    window_permutation: torch.Tensor
    token_windows: torch.Tensor
    token_positions: torch.Tensor
    output_shape: tuple[int, int, int]

    def restore(self, values):
        return values[
            self.token_windows, self.token_positions
        ].view(self.output_shape)


def build_doc_causal_block_mask(doc_id, window=None, hidden_key=None):
    """Build a full or trailing-window causal mask within packed documents.

    ``hidden_key`` is an optional ``[B, T]`` boolean marking tokens that only
    hidden tokens may attend to. A hidden token still attends causally to
    everything (including other hidden tokens), so its own hidden state is
    computed normally; every other token simply cannot read it.
    """
    batch_size, sequence_length = doc_id.shape

    # Separate closures: the CPU FlexAttention kernel is sensitive to the
    # structure of the traced mask expression.
    if hidden_key is None and window is None:
        def mask_mod(batch, head, query, key):
            del head
            return (query >= key) & (
                doc_id[batch, query] == doc_id[batch, key]
            )
    elif hidden_key is None:
        def mask_mod(batch, head, query, key):
            del head
            return (
                (query >= key)
                & (query < key + window)
                & (doc_id[batch, query] == doc_id[batch, key])
            )
    elif window is None:
        def mask_mod(batch, head, query, key):
            del head
            return (query >= key) & (
                (doc_id[batch, query] == doc_id[batch, key])
                & (~hidden_key[batch, key] | hidden_key[batch, query])
            )
    else:
        def mask_mod(batch, head, query, key):
            del head
            return (
                (query >= key)
                & (query < key + window)
                & (
                    (doc_id[batch, query] == doc_id[batch, key])
                    & (~hidden_key[batch, key] | hidden_key[batch, query])
                )
            )

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=doc_id.device,
    )


def _window_start_mask(position_ids, *, window, shift, first_window=None):
    is_start = (position_ids == 0) | (
        ((position_ids + int(shift)) % int(window)) == 0
    )
    if first_window is None:
        return is_start
    # ``first_window`` pins positions [0, first_window) of every document into
    # one fixed-length window at every depth. The shifted stride schedule is
    # untouched from that boundary onwards, so only the second window absorbs
    # the shift instead of the first.
    length = int(first_window)
    return (
        (position_ids == 0)
        | (position_ids == length)
        | (is_start & (position_ids > length))
    )


def build_window_metadata(
    doc_id, position_ids, *, window, shift=0, first_window=None,
    document_key_space=None, input_valid=None,
):
    """Return the canonical disjoint window layout for packed documents."""
    batch_size, sequence_length = doc_id.shape
    if document_key_space is None:
        document_key_space = sequence_length + 1
    batch = torch.arange(batch_size, device=doc_id.device).unsqueeze(-1)
    global_doc_id = batch * int(document_key_space) + doc_id

    is_window_start = _window_start_mask(
        position_ids, window=window, shift=shift, first_window=first_window
    )
    flat_starts = is_window_start.reshape(-1).long()
    global_window_id = (flat_starts.cumsum(0) - 1).view_as(doc_id)
    start_indices = is_window_start.reshape(-1).nonzero(as_tuple=True)[0]
    start_positions = position_ids.reshape(-1)[start_indices]
    doc_of_window = global_doc_id.reshape(-1)[start_indices]
    active_window_mask = (
        torch.ones(start_indices.numel(), device=doc_id.device, dtype=torch.bool)
        if input_valid is None else
        torch.bincount(
            global_window_id.reshape(-1)[input_valid.reshape(-1)],
            minlength=start_indices.numel(),
        ) > 0
    )
    return WindowMetadata(
        global_window_id=global_window_id,
        is_first_window_of_doc=start_positions == 0,
        global_doc_id=global_doc_id,
        doc_of_window=doc_of_window,
        window_start_pos=start_positions,
        window_start_token_idx=start_indices,
        n_windows_total=int(start_indices.numel()),
        active_window_mask=active_window_mask,
    )


def execute_packed_window_block(
    block, values, position_ids, valid, freqs_cis, *,
    block_mask=None, depth_film=None,
):
    """Canonical causal full-block execution for already-packed windows."""
    kwargs = {
        "position_ids": position_ids,
        "block_mask": block_mask,
        "ffn_token_mask": valid,
    }
    if depth_film is not None:
        kwargs["depth_film"] = depth_film
    return block(values, freqs_cis, **kwargs)


def canonical_window_counts(
    batch, length, *, window, stride, depth, first_window=None,
):
    """Padded shifted-window counts for full single-document rows."""
    positions = torch.arange(int(length))
    return [
        int(batch) * int(_window_start_mask(
            positions,
            window=window,
            shift=current_depth * int(stride),
            first_window=first_window,
        ).sum())
        for current_depth in range(int(depth))
    ]


@dataclass(frozen=True)
class _AttentionStrategy:
    """Validated attention policy shared by model families."""

    name: AttentionStrategyName
    window: int | None = None
    stride: int | None = None
    first_window: int | None = None

    @property
    def uses_windows(self):
        return self.name is AttentionStrategyName.SHIFTED_WINDOW

    @property
    def pack_width(self):
        """Tokens per packed window slot.

        Every window is packed into one fixed-width slot, so the slot must
        hold the longest window: the pinned first window when it is wider
        than ``window``. The slot width is storage and padding only; window
        boundaries, routes and masks still follow ``window`` and ``stride``.
        """
        return max(int(self.window), int(self.first_window or 0))

    def prepare_masks(self, doc_id, hidden_key=None):
        global_mask = build_doc_causal_block_mask(
            doc_id, hidden_key=hidden_key
        )
        local_mask = (
            build_doc_causal_block_mask(
                doc_id, self.window, hidden_key=hidden_key
            )
            if self.name is AttentionStrategyName.SLIDING_WINDOW
            else None
        )
        return global_mask, local_mask

    def block_mask(self, global_mask, local_mask):
        if self.name is AttentionStrategyName.FULL_CAUSAL:
            return global_mask
        if self.name is AttentionStrategyName.SLIDING_WINDOW:
            return local_mask
        raise ValueError(
            f"{self.name.value} uses packed windows, not a block mask."
        )

    def shift(self, depth):
        if self.name is AttentionStrategyName.SHIFTED_WINDOW:
            return int(depth) * self.stride
        raise ValueError(f"{self.name.value} does not define window shifts.")

    def packed_depth(self, index, block_count):
        """Return a Vanilla block's packed-window depth, or ``None``."""
        if (
            self.name is AttentionStrategyName.SHIFTED_WINDOW
            and 0 < int(index) < int(block_count) - 1
        ):
            return int(index) - 1
        return None

    def metadata(self, doc_id, position_ids, depth=0, document_key_space=None,
                 input_valid=None):
        return build_window_metadata(
            doc_id,
            position_ids,
            window=self.window,
            shift=self.shift(depth),
            first_window=self.first_window,
            document_key_space=document_key_space,
            input_valid=input_valid,
        )

    def pack(
        self,
        values,
        position_ids,
        metadata,
        sort_key=None,
    ):
        """Pack values using one cohesive metadata object."""
        return self.pack_layout(
            values,
            position_ids,
            metadata.global_window_id,
            metadata.window_start_pos,
            metadata.n_windows_total,
            sort_key,
        )

    def pack_layout(
        self,
        values,
        position_ids,
        global_window_id,
        window_start_pos,
        n_windows_total,
        sort_key=None,
    ):
        if not self.uses_windows:
            raise ValueError(f"{self.name.value} does not pack windows.")
        batch_size, sequence_length, width = values.shape
        flat_window = global_window_id.reshape(-1)
        flat_position = position_ids.reshape(-1)
        token_positions = (
            flat_position - window_start_pos[flat_window]
        ).long()
        if sort_key is None:
            sort_key = torch.arange(
                n_windows_total, device=values.device
            )
        permutation = torch.argsort(sort_key, stable=True)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            n_windows_total, device=values.device
        )
        token_windows = inverse[flat_window]

        packed_values = values.new_zeros(
            n_windows_total, self.pack_width, width
        )
        packed_positions = position_ids.new_zeros(
            n_windows_total, self.pack_width
        )
        valid = torch.zeros(
            n_windows_total, self.pack_width,
            device=values.device, dtype=torch.bool,
        )
        packed_values[token_windows, token_positions] = values.reshape(-1, width)
        packed_positions[token_windows, token_positions] = flat_position
        valid[token_windows, token_positions] = True
        return PackedWindows(
            packed_values,
            packed_positions,
            valid,
            permutation,
            token_windows,
            token_positions,
            (batch_size, sequence_length, width),
        )


def build_attention_strategy(config, architecture):
    """Validate and construct the only attention policy for a model middle."""
    if architecture not in _ALLOWED_STRATEGIES:
        raise ValueError(f"Unknown attention architecture {architecture!r}.")
    configured = getattr(config, "attention_strategy", None)
    if configured is None:
        raise ValueError(
            "model.attention_strategy is required."
        )
    name = parse_attention_strategy_name(configured)
    if name not in _ALLOWED_STRATEGIES[architecture]:
        allowed = ", ".join(
            sorted(choice.value for choice in _ALLOWED_STRATEGIES[architecture])
        )
        raise ValueError(
            f"{architecture} cannot use attention strategy {name.value!r}; "
            f"allowed strategies: {allowed}."
        )

    window = getattr(config, "window", None)
    stride = getattr(config, "window_stride", None)
    needs_window = name is not AttentionStrategyName.FULL_CAUSAL
    if needs_window and (window is None or int(window) <= 0):
        raise ValueError(f"{name.value} requires a positive model.window.")
    if not needs_window and window is not None:
        raise ValueError("full_causal requires model.window=null.")
    if name is AttentionStrategyName.SHIFTED_WINDOW:
        if stride is None or int(stride) <= 0:
            raise ValueError(
                "shifted_window requires a positive model.window_stride."
            )
        if architecture == "vanilla_neo" and int(config.n_layer) < 3:
            raise ValueError(
                "shifted_window vanilla requires model.n_layer >= 3."
            )
    elif stride is not None:
        raise ValueError(f"{name.value} requires model.window_stride=null.")
    first_window = getattr(config, "first_window_length", None)
    if first_window is not None:
        first_window = int(first_window)
        if architecture != "lol_dynamic_window":
            raise ValueError(
                "model.first_window_length is only supported by "
                f"lol_dynamic_window, not {architecture}."
            )
        if first_window < 1:
            raise ValueError(
                "model.first_window_length must be positive."
            )
    return _AttentionStrategy(
        name,
        window=None if window is None else int(window),
        stride=None if stride is None else int(stride),
        first_window=first_window,
    )
