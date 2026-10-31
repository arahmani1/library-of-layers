"""Shared mechanics of the five-strategy corpora (sudoku and maze).

Both datasets make the same kind of corpus: five sequences per question, one
per strategy, over a fixed small vocabulary with ``uint8`` ids; shards of
concatenated token streams with ``int64`` offsets and one JSON line of
metadata per sequence; hash splits by question; a held-out-pairs split. This
module owns those mechanics. ``data/<name>_data/corpus.py`` owns what differs
and is the *policy* the builder calls:

    NAME, UNIT, ID_FIELD, SPLITS,
    FORMAT_VERSION, SYMBOLS, UNK, decode, VERIFICATION
    add_arguments(parser), params(args)   dataset options -> config.json
    generate(rng, params)                 one attempt: a record, or the
                                          reason it was rejected (a string)
    sample_entry(meta, ids)               one entry of sample.txt

A record is ``{"key": str, "sequences": [(strategy, ids: bytes, fields)]}``:
``key`` identifies the question for deduplication and the hash splits, and
``fields`` are the per-sequence metadata columns in the order they are
written, between the id and the split.

Worker ``w`` draws from ``random.Random(seed * 1000 + w)`` and ids are
``"{seed}-{w}-{n}"``, so the same configuration reproduces the same shards.
"""

import hashlib
import json
import multiprocessing
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

DATASETS = ("sudoku", "maze")
SHARD_SIZE = 100_000
SAMPLE_SIZE = 200
# Packed trainer rows: ids are uint8 in the shards and uint16 once packed.
TOKEN_DTYPE = np.uint16
# Padded past the real symbols purely for tensor-core friendly matmuls.
MODEL_VOCAB_SIZE = 256
# Rows whose remaining capacity cannot hold the shortest sequence are closed.
# Bounding the open pool keeps the packer's memory flat over 36M sequences.
MAX_OPEN_ROWS = 65536
WRITE_CHUNK_ROWS = 4096
SHUFFLE_BUFFER = 1 << 20


def dataset_module(name):
    """The policy module of a dataset: ``data.<name>_data.corpus``."""
    if name == "sudoku":
        from data.sudoku_data import corpus
    elif name == "maze":
        from data.maze_data import corpus
    else:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    return corpus


def open_corpus(data_dir):
    """``(config, policy)`` of a built corpus, from its ``config.json``."""
    config = json.loads((Path(data_dir).expanduser() / "config.json").read_text())
    # Sudoku corpora built before the maze existed carry no dataset name.
    return config, dataset_module(config.get("dataset", "sudoku"))


def split_of(digest):
    bucket = int.from_bytes(digest, "big") % 100
    return "train" if bucket < 98 else "val" if bucket == 98 else "test"


def heldout_strategy(key):
    if int(hashlib.sha1(f"{key}|heldout".encode()).hexdigest(), 16) % 100 >= 2:
        return None
    return int(hashlib.sha1(f"{key}|pair".encode()).hexdigest(), 16) % 5 + 1


class ShardWriter:
    def __init__(self, root, split, shard_size=SHARD_SIZE):
        self.dir = Path(root) / split
        self.shard_size = shard_size
        self.shard = 0
        self.total = 0
        self.tokens_total = 0
        self.tokens = bytearray()
        self.offsets = [0]
        self.meta = []

    def add(self, ids, meta):
        self.tokens += ids
        self.offsets.append(len(self.tokens))
        self.meta.append(meta)
        self.total += 1
        self.tokens_total += len(ids)
        if len(self.meta) == self.shard_size:
            self.flush()

    def flush(self):
        if not self.meta:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        stem = self.dir / f"shard_{self.shard:05d}"
        stem.with_suffix(".bin").write_bytes(bytes(self.tokens))
        np.asarray(self.offsets, dtype=np.int64).tofile(stem.with_suffix(".idx"))
        with open(stem.with_suffix(".jsonl"), "w") as f:
            for m in self.meta:
                f.write(json.dumps(m) + "\n")
        self.shard += 1
        self.tokens = bytearray()
        self.offsets = [0]
        self.meta = []


def iter_split(data_dir, split):
    """Yield ``(metadata, token_ids)`` for every sequence of a split."""
    d = Path(data_dir).expanduser() / split
    if not d.is_dir():
        return
    for stem in sorted(p.stem for p in d.glob("shard_*.bin")):
        tokens = np.fromfile(d / f"{stem}.bin", dtype=np.uint8)
        offsets = np.fromfile(d / f"{stem}.idx", dtype=np.int64)
        with open(d / f"{stem}.jsonl") as f:
            for n, line in enumerate(f):
                yield json.loads(line), tokens[offsets[n] : offsets[n + 1]]


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _worker(task):
    dataset, w, quota, seed, params = task
    corpus = dataset_module(dataset)
    rng = random.Random(seed * 1000 + w)
    out = []
    rejected = Counter()
    n = 0
    while len(out) < quota:
        record = corpus.generate(rng, params)
        if isinstance(record, str):
            rejected[record] += 1
            continue
        record["id"] = f"{seed}-{w}-{n}"
        out.append(record)
        n += 1
    return out, rejected


def build(dataset, out, n, workers, seed, params):
    """Generate ``n`` questions with ``workers`` processes into ``out``.

    Questions are deduplicated on their key and split 98/1/1 by its hash;
    2% of train questions lose one strategy's sequence to
    ``test_heldout_pairs``, and the first ``params["n_swap"]`` test questions
    go to ``test_swap`` when the policy has that split. Returns the
    ``config.json`` contents.
    """
    corpus = dataset_module(dataset)
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    n_swap = params.get("n_swap", 0)
    if n_swap and "test_swap" not in corpus.SPLITS:
        raise ValueError(f"{dataset} has no test_swap split")

    tasks = [
        (dataset, w, n // workers + (1 if w < n % workers else 0), seed, params)
        for w in range(workers)
    ]
    writers = {split: ShardWriter(out, split) for split in corpus.SPLITS}
    seen = set()
    duplicates = 0
    rejected = Counter()
    swapped = 0
    sample = []
    t0 = time.time()

    with multiprocessing.Pool(workers) as pool:
        for batch, batch_rejected in pool.imap(_worker, tasks):
            rejected.update(batch_rejected)
            for rec in batch:
                digest = hashlib.sha1(rec["key"].encode()).digest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                split = split_of(digest)
                if split == "test" and swapped < n_swap:
                    split = "test_swap"
                    swapped += 1
                held = heldout_strategy(rec["key"]) if split == "train" else None
                for k, ids, fields in rec["sequences"]:
                    dest = "test_heldout_pairs" if k == held else split
                    meta = {corpus.ID_FIELD: rec["id"], **fields, "split": dest}
                    writers[dest].add(ids, meta)
                    if dest == "train" and len(sample) < SAMPLE_SIZE:
                        sample.append(corpus.sample_entry(meta, ids))

    for writer in writers.values():
        writer.flush()

    elapsed = time.time() - t0
    accepted = len(seen)
    unit = corpus.UNIT
    config = {
        "dataset": dataset,
        "format_version": corpus.FORMAT_VERSION,
        f"n_{unit}": n,
        **params,
        "workers": workers,
        "seed": seed,
        "shard_size": SHARD_SIZE,
        "git_commit": _git_commit(),
        "run": {
            f"{unit}_accepted": accepted,
            f"{unit}_rejected": sum(rejected.values()),
            "rejection_rate": sum(rejected.values())
            / max(accepted + sum(rejected.values()), 1),
            "rejections": dict(sorted(rejected.items())),
            "duplicates_removed": duplicates,
            "elapsed_s": round(elapsed, 1),
            f"{unit}_per_s": round(n / max(elapsed, 1e-9), 1),
            "sequences": {split: writers[split].total for split in corpus.SPLITS},
            "tokens": {
                split: writers[split].tokens_total for split in corpus.SPLITS
            },
            "verification": corpus.VERIFICATION,
        },
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out / "vocab.json", "w") as f:
        json.dump(corpus.SYMBOLS, f)
    with open(out / "sample.txt", "w") as f:
        f.write("\n".join(sample) + "\n")
    return config


def pack(targets, out, block_size, min_length, seed, shuffle_buffer, pad_token_id):
    """Pack whole sequences into ``block_size`` rows, one ``<name>.bin`` each.

    ``targets`` is ``[(name, split, records)]`` where ``records`` yields
    ``uint16`` token arrays. Rows are filled from a shuffled stream with
    best-fit placement (data/prepare_sudoku.py explains why not by length)
    and padded with ``pad_token_id``. Returns the manifest's ``splits`` and
    ``packing`` sections.
    """
    # Trainer-side dependencies stay out of the raw builders' import graph.
    from data.preparation_utils import RowPacker, shuffled

    out = Path(out)
    splits = {}
    packing = {}
    started = time.monotonic()
    for name, split, records in targets:
        rng = random.Random(f"{seed}:{name}")
        buffer_size = shuffle_buffer if split == "train" else 1 << 16
        packer = RowPacker(
            block_size, min_length, pad_token_id=pad_token_id,
            token_dtype=TOKEN_DTYPE, max_open_rows=MAX_OPEN_ROWS,
        )
        pending = []
        with open(out / f"{name}.bin", "wb") as handle:
            def drain(rows):
                pending.extend(rows)
                if len(pending) >= WRITE_CHUNK_ROWS:
                    np.concatenate(pending).tofile(handle)
                    pending.clear()

            for tokens in shuffled(records, buffer_size, rng):
                drain(packer.add(tokens))
            drain(packer.flush())
            if pending:
                np.concatenate(pending).tofile(handle)
        tokens_count = packer.n_rows * block_size
        splits[name] = {"tokens": f"{name}.bin", "tokens_count": tokens_count}
        packing[name] = {
            "sequences": packer.n_sequences,
            "rows": packer.n_rows,
            "content_tokens": packer.used_tokens,
            "occupancy": round(packer.occupancy, 5),
            "sequences_per_row": round(
                packer.n_sequences / max(packer.n_rows, 1), 3
            ),
        }
        print(
            f"{name:20s} {packer.n_sequences:>10,} seqs -> "
            f"{packer.n_rows:>9,} rows  occupancy {packer.occupancy:6.2%}  "
            f"{tokens_count:>14,} tokens"
        )
    print(f"Done in {(time.monotonic() - started) / 60:.1f} min -> {out}")
    return splits, packing
