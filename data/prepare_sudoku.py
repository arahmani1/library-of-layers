"""Pack sudoku-strategies sequences into fixed-width rows for the trainer.

``DataLoaderLit_MultiProc`` memory-maps a flat token stream and always cuts it
at exact multiples of ``block_size``, so a loader row and a written record
coincide as long as every row is exactly ``block_size`` tokens. This script
fills rows with *whole* sequences only -- a puzzle is never split across a row
boundary -- and pads the remainder with ``<pad>``.

``Q`` (id 3) is the document delimiter rather than a new symbol: it is already
the first token of every sequence and occurs exactly once, so ``compute_doc_meta``
recovers per-document ids and positions with no vocabulary remapping. Packed
documents never see each other -- attention masks are doc-causal and the
router's prefix features reset at each document start.

Rows are filled from a *shuffled* stream with best-fit placement. Sorting by
length packs marginally tighter but leaves about half of the rows holding a
single strategy, which would correlate the router's load-balancing statistics
with strategy inside every micro-batch. Shuffled best-fit reaches ~90.6%
occupancy at block 2048 with ~3% single-strategy rows (v2 manifests).

Usage:
    python -m data.prepare_sudoku \\
        --source datasets/sudoku_strategies/v2 \\
        --out datasets/sudoku_packed/v2 --block-size 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.preparation_utils import atomic_write_json
from data.strategy_corpus import (
    MODEL_VOCAB_SIZE, SHUFFLE_BUFFER, TOKEN_DTYPE, iter_split, pack,
)
from data.sudoku_data.vocab import PAD, SYMBOLS, VOCAB

# The document delimiter the trainer segments on. ``Q`` opens every sequence
# and appears nowhere else, so it needs no dedicated token.
EOT_TOKEN_ID = VOCAB["Q"]
SOURCE_SPLITS = ("train", "val", "test", "test_heldout_pairs")
STRATEGIES = tuple(f"S{k}" for k in range(1, 6))
# Source records contain "Q : <81 clues> [ S<k> ] R : <trace>".
# Packing expands each strategy token into its description.
PROMPT_LENGTH = 83
TAG_SPAN = slice(83, 86)
TAG_RANGE = (VOCAB["S1"], VOCAB["S5"] + 1)
# Replace the tag token with a fixed schematic
# step of the strategy, padded with "." to DESCRIPTION_LENGTH tokens:
#   Q : <81 clues> [ <description> ] R : ...
DESCRIPTION_LENGTH = 6
STRATEGY_DESCRIPTIONS = {
    "S1": "r5c5 = { 5 }",
    "S2": "row3 5 -> r3c7",
    "S3": "3 box4 -> r6c3",
    "S4": "set r5c5 = 5 :",
    "S5": "r1c3 = 1 x back",
}
# "[ <description> ]", per-document positions.
DESCRIPTION_SPAN = (PROMPT_LENGTH, PROMPT_LENGTH + DESCRIPTION_LENGTH + 2)


def description_tokens(strategy):
    """Fixed-length description ids for ``strategy`` ("S1" .. "S5")."""
    ids = [VOCAB[symbol] for symbol in STRATEGY_DESCRIPTIONS[strategy].split()]
    if len(ids) > DESCRIPTION_LENGTH:
        raise ValueError(
            f"{strategy} description has {len(ids)} tokens, more than "
            f"DESCRIPTION_LENGTH={DESCRIPTION_LENGTH}"
        )
    ids += [VOCAB["."]] * (DESCRIPTION_LENGTH - len(ids))
    return np.asarray(ids, dtype=TOKEN_DTYPE)


def strategy_of_tag(tokens):
    """Read "S<k>" off the tag token of a tagged sequence."""
    tag = int(tokens[TAG_SPAN.start + 1])
    if not TAG_RANGE[0] <= tag < TAG_RANGE[1]:
        raise ValueError(
            "Expected a strategy tag at position "
            f"{TAG_SPAN.start + 1}, found id {tag}"
        )
    return f"S{tag - TAG_RANGE[0] + 1}"


def with_description(tokens):
    """Replace the tag token of one sequence by its strategy description."""
    strategy = strategy_of_tag(tokens)
    return np.concatenate((
        tokens[:TAG_SPAN.start + 1],
        description_tokens(strategy),
        tokens[TAG_SPAN.start + 2:],
    ))


def source_records(source, split, strategy=None):
    """Yield description-labelled uint16 sequences for the selected strategy."""
    for meta, tokens in iter_split(source, split):
        if strategy is not None and meta["strategy"] != strategy:
            continue
        yield with_description(tokens.astype(TOKEN_DTYPE))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-length", type=int, default=340,
        help="Close a row once it cannot hold a sequence this short.",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=SHUFFLE_BUFFER)
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # Per-strategy validation splits let the routing diagnostics read expert
    # usage per strategy; the pooled ``val`` carries the headline NLL.
    targets = [(name, name, None) for name in SOURCE_SPLITS]
    targets += [(f"val_{s}", "val", s) for s in STRATEGIES]
    splits, packing = pack(
        [
            (name, split, source_records(source, split, strategy))
            for name, split, strategy in targets
        ],
        out, args.block_size, args.min_length, args.seed, args.shuffle_buffer,
        PAD,
    )

    atomic_write_json(out / "manifest.json", {
        "format_version": 1,
        "dataset": {"id": "sudoku-strategies", "source": str(source)},
        "packing": {
            "block_size": args.block_size,
            "policy": "shuffled best-fit, whole sequences only",
            "pad_token_id": PAD,
            "seed": args.seed,
            "splits": packing,
        },
        "layout": {
            "prompt_length": PROMPT_LENGTH,
            "strategy_tag": True,
            "tag_span": list(DESCRIPTION_SPAN),
            "strategy_descriptions": STRATEGY_DESCRIPTIONS,
            "reasoning_start": DESCRIPTION_SPAN[1] + 2,
        },
        "splits": splits,
        "token_dtype": "uint16",
        "tokenizer": {
            "id": "sudoku-strategies",
            "eot_token_id": EOT_TOKEN_ID,
            "vocab_size": len(SYMBOLS),
            "model_vocab_size": MODEL_VOCAB_SIZE,
        },
    })


if __name__ == "__main__":
    main()
