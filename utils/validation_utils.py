"""Shared validation metric helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from data import sudoku_sections


MASKED_LOSS_ARCHES = frozenset({"lol_dynamic_window"})


def dataset_section_masks(
    dataset, arm, tag_span, inputs, targets, doc_ids, position_ids,
):
    """Return (loss mask, per-metric masks) for a strategy-corpus loss arm.

    ``dataset`` (``data.dataset``, sudoku or maze) selects the sequence
    layout. Both are ``None`` when ``arm`` is unset, so every other dataset
    keeps training and scoring on every target. Training and standalone
    validation must both apply this mask, or the standalone NLL also scores
    the prompt and the padding the model was never trained on.
    """

    if arm is None:
        return None, None
    layout = sudoku_sections.layout_for(dataset)
    codes = sudoku_sections.target_sections(
        inputs, position_ids, doc_ids, tag_span=tag_span, layout=layout,
    )
    return (
        sudoku_sections.loss_mask(codes, str(arm)),
        sudoku_sections.metric_masks(codes, targets, layout=layout),
    )


def unpack_validation_outputs(outputs, targets, arch: str):
    """Return logits and the architecture's optional token-loss mask."""

    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    loss_mask = None
    if arch in MASKED_LOSS_ARCHES:
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise ValueError(f"{arch} did not return its expected loss mask")
        loss_mask = outputs[2]
        if loss_mask.shape != targets.shape:
            raise ValueError(
                f"{arch} loss mask shape {tuple(loss_mask.shape)} does not "
                f"match targets {tuple(targets.shape)}"
            )
    valid = targets != -100
    loss_mask = valid if loss_mask is None else loss_mask & valid
    return logits, loss_mask


def token_nll_sum(logits, targets, loss_mask=None):
    """Compute a token-weighted NLL numerator and denominator."""

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    mask = targets != -100
    if loss_mask is not None:
        mask = mask & loss_mask.to(device=per_token.device, dtype=torch.bool)
    return per_token.masked_select(mask).sum(), mask.sum()


def target_byte_count(byte_lengths, loss_mask=None):
    """Count UTF-8 bytes for the same targets used by token NLL."""

    if byte_lengths is None:
        return None
    if loss_mask is None:
        return byte_lengths.sum()
    mask = loss_mask.to(device=byte_lengths.device, dtype=torch.bool)
    return byte_lengths.masked_select(mask).sum()


def reset_validation_loader(loader) -> None:
    """Reset a loader to its deterministic rank-specific starting position."""

    loader.reset()


def validation_batch_count(loader, requested: int) -> int:
    """Cap evaluation before a sequential loader wraps and repeats data."""

    available = int(loader.total_batches)
    if available <= 0:
        raise ValueError("Validation split is too small to form one batch")
    if requested <= 0:
        return available
    return min(int(requested), available)
