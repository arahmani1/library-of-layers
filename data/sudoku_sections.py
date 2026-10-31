"""Per-token section labels and content masks for packed strategy corpora.

Everything here is derived on-device from token ids alone -- no sidecar file.
The corpora make that possible: ``Q`` opens every document and occurs nowhere
else, the prompt has a fixed length, ``R`` and ``A`` occur exactly once per
document, and the sudoku and maze vocabularies share the ids of ``<pad>``,
``Q``, ``R`` and ``A`` (``data.build`` verifies every sequence it writes).
A ``Layout`` supplies what differs between the corpora: the prompt length and
the id ranges of the content tokens; ``data.dataset`` selects it.

sudoku (data/sudoku_data/format.py):

    idx   0 .. 82   Q : <81 clues>      PROMPT
    idx  83 .. 90   [ <6 tokens> ]      TAG, by position through tag_span
    idx  91 .. 92   R :                 GLUE
    idx  93 .. a-1  reasoning trace     REASONING
    idx   a .. a+1  A :                 GLUE
    idx a+2..a+82   81 answer digits    ANSWER
    idx a+83        <eos>               GLUE

maze (data/maze_data/format.py):

    idx   0 .. 195  Q : <64 cells with sides> start s goal g   PROMPT
    idx 196 .. 203  [ <6 label cells> ]  TAG, by position through tag_span
    idx 204 .. 205  R :                 GLUE
    idx 206 .. a-1  trace cells         REASONING
    idx   a .. a+1  A :                 GLUE
    idx a+2 .. e-1  path cells          ANSWER
    idx   e         <eos>               GLUE

The maze label is ordinary cell syntax, so only ``data.tag_span=[196,204]``
can mark it (utils/configuration.py requires it with a loss arm).

The "have we passed ``A`` yet" test avoids a segmented scan. ``doc_id`` is the
running count of ``Q`` and ``seen_a`` the running count of ``A``, so inside
document *d* they are equal exactly at and after that document's ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import torch

from .maze_data.format import PROMPT_LENGTH as MAZE_PROMPT_LENGTH
from .maze_data.vocab import VOCAB as MAZE_VOCAB
from .sudoku_data.vocab import VOCAB

PAD_ID = VOCAB["<pad>"]
R_ID = VOCAB["R"]
A_ID = VOCAB["A"]
# The maze vocabulary was laid out to share these ids (data/maze_data/vocab.py).
assert all(MAZE_VOCAB[s] == VOCAB[s] for s in ("<pad>", "Q", "R", "A"))

# Contiguous id ranges, half-open. Checked against vocab.SYMBOLS below.
DIGIT_RANGE = (VOCAB["1"], VOCAB["9"] + 1)          # 18..27
CELL_RANGE = (VOCAB["r1c1"], VOCAB["r9c9"] + 1)     # 32..113
UNIT_RANGE = (VOCAB["row1"], VOCAB["box9"] + 1)     # 113..140
MAZE_CELL_RANGE = (MAZE_VOCAB["r1c1"], MAZE_VOCAB["r8c8"] + 1)   # 21..85

PAD = 0
PROMPT = 1
GLUE = 2
TAG = 3
REASONING = 4
ANSWER = 5

# The label is supplied at inference, so supervise only the generated body.
LOSS_SECTIONS = {
    "reasoning_answer_untagged": (GLUE, REASONING, ANSWER),
}


@dataclass(frozen=True)
class Layout:
    """What one corpus's sequences look like to the section labeller."""

    name: str
    # Per-document positions before this are the prompt; the pinned first
    # window starts here and may extend over the label (first_window_length).
    prompt_length: int
    # Ids that are answer content after "A :"; the rest of the answer is glue.
    answer_range: tuple[int, int]
    # Reasoning content classes by id range, scored as ``reasoning_<name>``.
    content_ranges: tuple[tuple[str, tuple[int, int]], ...]
    answer_metric: str

    @property
    def metric_names(self):
        """Fixed order: evaluation accumulates these into tensor columns."""
        return tuple(
            [f"reasoning_{name}" for name, _bounds in self.content_ranges]
            + ["reasoning_content", "reasoning_filler", self.answer_metric,
               "strategy_tag"]
        )


SUDOKU = Layout(
    "sudoku", 83, DIGIT_RANGE,
    (("cell", CELL_RANGE), ("digit", DIGIT_RANGE), ("unit", UNIT_RANGE)),
    "answer_digit",
)
MAZE = Layout(
    "maze", MAZE_PROMPT_LENGTH, MAZE_CELL_RANGE,
    (("cell", MAZE_CELL_RANGE),), "answer_cell",
)
LAYOUTS = {SUDOKU.name: SUDOKU, MAZE.name: MAZE}


def layout_for(dataset):
    """The layout ``data.dataset`` selects; only these corpora have sections."""
    try:
        return LAYOUTS[dataset]
    except KeyError:
        raise ValueError(
            "data.loss_sections labels the sequences of "
            f"data.dataset in {sorted(LAYOUTS)}, not {dataset!r}."
        ) from None


def _in_range(idx, bounds):
    low, high = bounds
    return (idx >= low) & (idx < high)


def section_codes(idx, position_ids, doc_id, tag_span=None, layout=SUDOKU):
    """Label every token of ``idx`` with the section it belongs to.

    ``position_ids`` and ``doc_id`` come from ``compute_doc_meta(idx, VOCAB["Q"])``,
    so ``doc_id`` counts ``Q`` and is 1 for the first document of a row.
    ``tag_span`` is the per-document ``[start, stop)`` of the strategy label
    for a Sudoku description or maze label: every token in it, brackets
    included, is ``TAG``.
    """

    is_pad = idx == PAD_ID
    # Padding trails the final document, so it inherits that document's
    # "past A" and "past R" state and has to be excluded explicitly.
    past_a = (idx == A_ID).cumsum(dim=1) == doc_id
    past_r = (idx == R_ID).cumsum(dim=1) == doc_id
    # "R :" opens the trace, so it starts two tokens after R. Reading the
    # boundary off the stream accommodates both corpus layouts.
    in_trace = torch.zeros_like(past_r)
    in_trace[:, 2:] = past_r[:, :-2]
    is_answer = _in_range(idx, layout.answer_range)

    def filled(code):
        return torch.full_like(idx, code)

    codes = filled(GLUE)
    codes = torch.where(in_trace, filled(REASONING), codes)
    codes = torch.where(past_a & is_answer, filled(ANSWER), codes)
    codes = torch.where(past_a & ~is_answer, filled(GLUE), codes)
    if tag_span is not None:
        start, stop = (int(bound) for bound in tag_span)
        in_span = (position_ids >= start) & (position_ids < stop)
        codes = torch.where(in_span, filled(TAG), codes)
    codes = torch.where(position_ids < layout.prompt_length, filled(PROMPT), codes)
    return torch.where(is_pad, filled(PAD), codes)


def target_sections(idx, position_ids, doc_id, tag_span=None, layout=SUDOKU):
    """Section codes aligned to ``targets``, i.e. shifted one position left.

    ``targets[b, t] == idx[b, t + 1]`` within a loader window, so the target's
    section is the next token's section. The final column's target lies outside
    the window and is labelled ``PAD`` (never supervised, never scored) -- one
    token per row in 2048.
    """

    codes = section_codes(idx, position_ids, doc_id, tag_span=tag_span, layout=layout)
    shifted = torch.full_like(codes, PAD)
    shifted[:, :-1] = codes[:, 1:]
    return shifted


def loss_mask(sections, arm="reasoning_answer_untagged"):
    """Boolean mask over targets selecting the sections one arm supervises."""

    try:
        wanted = LOSS_SECTIONS[arm]
    except KeyError:
        raise ValueError(
            f"Unknown loss arm {arm!r}; expected one of "
            f"{sorted(LOSS_SECTIONS)}."
        ) from None
    mask = torch.zeros_like(sections, dtype=torch.bool)
    for code in wanted:
        mask |= sections == code
    return mask


def metric_masks(sections, targets, layout=SUDOKU):
    """Named boolean masks for teacher-forced per-token accuracy.

    Split by *what the token is*, because the halves measure different
    abilities: a cell token is the solver's choice of where to act next, a
    digit token is the constraint that follows from it, and filler is
    (for S1-S4; in S5 the ``x``/``back`` filler tokens are the decisions)
    syntax the model gets for free. A maze trace is cells only, so its
    filler class is empty. Keys are ``layout.metric_names``.
    """

    reasoning = sections == REASONING
    classes = {
        name: reasoning & _in_range(targets, bounds)
        for name, bounds in layout.content_ranges
    }
    content = reduce(torch.logical_or, classes.values())
    masks = {f"reasoning_{name}": mask for name, mask in classes.items()}
    masks["reasoning_content"] = content
    masks["reasoning_filler"] = reasoning & ~content
    masks[layout.answer_metric] = sections == ANSWER
    masks["strategy_tag"] = sections == TAG
    return masks
