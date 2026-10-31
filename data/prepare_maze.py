"""Pack maze-strategies sequences into fixed-width rows for the trainer.

Same policy as ``data.prepare_sudoku``: whole sequences only, rows of
``block_size`` tokens filled best-fit from a shuffled stream and padded with
``<pad>``, ``Q`` as the document delimiter, and per-strategy validation
splits. The maze label is part of the question (``[ <six cells> ]`` at
positions 196..203), so there is no tag to drop or describe; the manifest
publishes its span as ``layout.tag_span`` for ``data.tag_span`` and
``model.hidden_key_span``.

Usage:
    python -m data.prepare_maze \\
        --source datasets/maze_strategies/v1 \\
        --out datasets/maze_packed/v1 --block-size 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.maze_data.format import (
    LABEL_SPAN, LABELS, PROMPT_LENGTH, Q_LEN, REASONING_START,
)
from data.maze_data.vocab import PAD, SYMBOLS, VOCAB
from data.preparation_utils import atomic_write_json
from data.strategy_corpus import (
    MODEL_VOCAB_SIZE, SHUFFLE_BUFFER, TOKEN_DTYPE, iter_split, open_corpus,
    pack,
)

EOT_TOKEN_ID = VOCAB["Q"]
SOURCE_SPLITS = ("train", "val", "test", "test_heldout_pairs", "test_swap")
STRATEGIES = tuple(f"S{k}" for k in range(1, 6))


def shortest_sequence(config):
    """No sequence is shorter: ``min_path`` trace cells and as many path
    cells around the question, "R :", "A :" and <eos>."""
    return Q_LEN + 2 + config["min_path"] + 2 + config["min_path"] + 1


def source_records(source, split, strategy=None):
    """Yield uint16 token arrays, optionally restricted to one strategy."""
    for meta, tokens in iter_split(source, split):
        if strategy is None or meta["strategy"] == strategy:
            yield tokens.astype(TOKEN_DTYPE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-length", type=int, default=None,
        help="Close a row once it cannot hold a sequence this short "
             "(default: the corpus's shortest possible sequence).",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=SHUFFLE_BUFFER)
    args = parser.parse_args(argv)

    source = Path(args.source).expanduser()
    out = Path(args.out).expanduser()
    config, corpus = open_corpus(source)
    if corpus.NAME != "maze":
        parser.error(f"{source} holds a {corpus.NAME} corpus")
    min_length = args.min_length or shortest_sequence(config)
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
        out, args.block_size, min_length, args.seed, args.shuffle_buffer, PAD,
    )

    atomic_write_json(out / "manifest.json", {
        "format_version": 1,
        "dataset": {"id": "maze-strategies", "source": str(source)},
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
            # Per-document positions of "[ <six label cells> ]".
            "tag_span": list(LABEL_SPAN),
            "strategy_labels": {f"S{k}": label for k, label in LABELS.items()},
            "reasoning_start": REASONING_START,
        },
        "splits": splits,
        "token_dtype": "uint16",
        "tokenizer": {
            "id": "maze-strategies",
            "eot_token_id": EOT_TOKEN_ID,
            "vocab_size": len(SYMBOLS),
            "model_vocab_size": MODEL_VOCAB_SIZE,
        },
    })


if __name__ == "__main__":
    main()
