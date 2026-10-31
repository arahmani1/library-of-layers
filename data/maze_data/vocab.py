"""Fixed 85-symbol vocabulary; ids are list positions.

The structural symbols share their ids with the sudoku vocabulary
(``<pad>`` 0, ``<eos>`` 1, ``<unk>`` 2, ``Q`` 3, ``R`` 4, ``A`` 5, ``:`` 6),
so the trainer segments and labels both corpora with the same ids.
"""

SYMBOLS = (
    "<pad> <eos> <unk> Q R A : [ ] . start goal N E S W".split()
    + [f"S{k}" for k in range(1, 6)]
    + [f"r{r}c{c}" for r in range(1, 9) for c in range(1, 9)]
)
VOCAB = {s: i for i, s in enumerate(SYMBOLS)}
PAD, EOS, UNK = VOCAB["<pad>"], VOCAB["<eos>"], VOCAB["<unk>"]


def encode(text):
    return [VOCAB.get(t, UNK) for t in text.split(" ")]


def decode(ids):
    return " ".join(SYMBOLS[i] for i in ids)
