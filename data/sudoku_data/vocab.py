"""Fixed 140-symbol vocabulary; ids are list positions."""

SYMBOLS = (
    "<pad> <eos> <unk> Q R A : ; = -> { } [ ] x back set .".split()
    + [str(d) for d in range(1, 10)]
    + [f"S{k}" for k in range(1, 6)]
    + [f"r{r}c{c}" for r in range(1, 10) for c in range(1, 10)]
    + [f"row{u}" for u in range(1, 10)]
    + [f"col{u}" for u in range(1, 10)]
    + [f"box{u}" for u in range(1, 10)]
)
VOCAB = {s: i for i, s in enumerate(SYMBOLS)}
PAD, EOS, UNK = VOCAB["<pad>"], VOCAB["<eos>"], VOCAB["<unk>"]


def encode(text):
    return [VOCAB.get(t, UNK) for t in text.split(" ")]


def decode(ids):
    return " ".join(SYMBOLS[i] for i in ids)
