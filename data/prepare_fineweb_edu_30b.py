"""Prepare the 30B-token FineWeb-Edu binary from the ``sample/100BT`` shards.

FineWeb-Edu publishes 10BT, 100BT, and 350BT samples but no 30BT sample, so the
30B corpus is defined as an ordered prefix of ``sample/100BT``. Shards are
downloaded concurrently and tokenized as they land, then consumed in sorted
order so the output does not depend on download timing.

Documents are packed as ``[EOT, tokens]`` in ``uint16``. The validation split is
keyed on an md5 of the document text, so it is independent of worker count and
completion order and the resulting binaries are byte-reproducible.
"""

import argparse
import os
import time
import multiprocessing as mp
from pathlib import Path
from functools import partial
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from tqdm import tqdm

from data.preparation_utils import (
    OrderedProcessMapper,
    atomic_write_json,
    configure_worker_process,
    download_hf_file,
    load_json,
    ordered_thread_map,
    stable_hash_bucket,
)

REPO_ID = "HuggingFaceFW/fineweb-edu"
SHARD_PREFIX = "sample/100BT"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
TARGET_TOKENS = 30_000_000_000
# Shards hold ~752M tokens each, so 30B needs ~40; 44 covers the target with margin.
MAX_SHARDS = 44
# Documents per batch handed to a tokenizer worker.
CHUNK_SIZE = 2000
# The Hub rate-limits unauthenticated Xet reads; keep concurrency modest and
# retry, since one 429 would otherwise abort the whole job.
DOWNLOAD_WORKERS = 8
DOWNLOAD_MAX_RETRIES = 8
DOWNLOAD_BACKOFF_BASE = 5
DOWNLOAD_BACKOFF_CAP = 120
# Leave cores for the download threads and the writer.
TOKENIZE_WORKERS = max(1, mp.cpu_count() - 16)

_tokenizer = None
_eot_token = None
_val_fraction = None


def _load_tokenizer(val_fraction):
    global _tokenizer, _eot_token, _val_fraction

    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    _eot_token = _tokenizer.eos_token_id
    _val_fraction = val_fraction


def worker_init(val_fraction):
    """Initialise the tokenizer in each worker process."""

    configure_worker_process()
    _load_tokenizer(val_fraction)


def process_batch(batch_texts):
    """Tokenize one batch of documents and return raw train/validation bytes."""

    encoded_docs = _tokenizer(batch_texts, add_special_tokens=False)["input_ids"]

    train_bytes = bytearray()
    val_bytes = bytearray()
    train_count = 0
    val_count = 0

    for text, tokens in zip(batch_texts, encoded_docs):
        full_tokens = [_eot_token] + tokens
        tokens_bytes = np.asarray(full_tokens, dtype=np.uint16).tobytes()

        # The explicit MD5 mode preserves the existing FineWeb split exactly.
        bucket = stable_hash_bucket(text, 10000, algorithm="md5")
        if bucket < int(_val_fraction * 10000):
            val_bytes.extend(tokens_bytes)
            val_count += len(full_tokens)
        else:
            train_bytes.extend(tokens_bytes)
            train_count += len(full_tokens)

    return bytes(train_bytes), train_count, bytes(val_bytes), val_count


def _save_checkpoint(path, shards_done, train_tokens, val_tokens):
    atomic_write_json(path, {
        "shards_done": shards_done,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    })


def prepare_fineweb_edu_30b(output_dir, val_fraction=0.01, workers=None):
    workers = TOKENIZE_WORKERS if workers is None else workers
    if workers < 1:
        raise ValueError("workers must be positive")
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_bin_path = save_dir / "train.bin"
    val_bin_path = save_dir / "val.bin"
    checkpoint_path = save_dir / ".prep30b_checkpoint.json"

    shards = sorted(
        f for f in HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith(SHARD_PREFIX) and f.endswith(".parquet")
    )[:MAX_SHARDS]

    ckpt = load_json(checkpoint_path)
    if ckpt is None:
        existing = [
            path.name
            for path in (train_bin_path, val_bin_path)
            if path.exists() and path.stat().st_size
        ]
        if existing:
            names = ", ".join(existing)
            raise FileExistsError(
                f"Refusing to overwrite existing dataset files in "
                f"{save_dir}: {names}. Choose an empty --output-dir."
            )
        ckpt = {
            "shards_done": 0, "train_tokens": 0, "val_tokens": 0,
        }
        _save_checkpoint(checkpoint_path, 0, 0, 0)
    shards_done = int(ckpt["shards_done"])
    train_tokens = int(ckpt["train_tokens"])
    val_tokens = int(ckpt["val_tokens"])

    # A checkpoint covers complete shards only. Drop bytes from an interrupted
    # next shard before appending again.
    for path, tokens in (
        (train_bin_path, train_tokens),
        (val_bin_path, val_tokens),
    ):
        size = tokens * np.dtype(np.uint16).itemsize
        if not path.exists():
            if size:
                raise RuntimeError(f"Checkpoint expects missing file {path}")
            path.touch()
        if path.stat().st_size < size:
            raise RuntimeError(f"{path} is shorter than its checkpoint")
        with path.open("r+b") as stream:
            stream.truncate(size)

    if train_tokens + val_tokens >= TARGET_TOKENS:
        print(f"Already prepared {train_tokens + val_tokens:,} tokens.")
        return

    if shards_done:
        print(
            f"Resuming after {shards_done} shards: {train_tokens:,} train + "
            f"{val_tokens:,} validation tokens"
        )
    print(f"{len(shards)} candidate shards | target {TARGET_TOKENS:,} tokens")
    print(
        f"{DOWNLOAD_WORKERS} download threads | "
        f"{workers} tokenizer workers"
    )
    print(f"Writing to {save_dir.resolve()}")

    # Fetch every remaining shard concurrently; consume them strictly in order
    # so the output stays byte-reproducible regardless of completion order.
    download = partial(
        download_hf_file,
        REPO_ID,
        repo_type="dataset",
        max_retries=DOWNLOAD_MAX_RETRIES,
        backoff_base=DOWNLOAD_BACKOFF_BASE,
        backoff_cap=DOWNLOAD_BACKOFF_CAP,
    )
    if workers == 1:
        _load_tokenizer(val_fraction)
    started = time.monotonic()
    remaining_shards = shards[shards_done:]
    downloads = ordered_thread_map(
        remaining_shards, download, DOWNLOAD_WORKERS
    )
    with (
        OrderedProcessMapper(
            workers,
            initializer=worker_init,
            initargs=(val_fraction,),
        ) as tokenizer,
        train_bin_path.open("ab") as f_train,
        val_bin_path.open("ab") as f_val,
    ):
        pbar = tqdm(total=TARGET_TOKENS, initial=train_tokens + val_tokens,
                    unit="tok", unit_scale=True, desc="Tokenizing",
                    mininterval=5.0, smoothing=0.05)
        try:
            for i, path in enumerate(downloads, start=shards_done):

                parquet = pq.ParquetFile(path)
                batches = parquet.iter_batches(batch_size=CHUNK_SIZE,
                                               columns=["text"])
                texts = (b.column("text").to_pylist() for b in batches)

                for train_b, t_count, val_b, v_count in tokenizer.map(
                    process_batch, texts, chunksize=1
                ):
                    if train_b:
                        f_train.write(train_b)
                    if val_b:
                        f_val.write(val_b)
                    train_tokens += t_count
                    val_tokens += v_count
                    pbar.update(t_count + v_count)

                shards_done = i + 1
                f_train.flush()
                f_val.flush()
                os.fsync(f_train.fileno())
                os.fsync(f_val.fileno())
                _save_checkpoint(
                    checkpoint_path, shards_done, train_tokens, val_tokens
                )

                if train_tokens + val_tokens >= TARGET_TOKENS:
                    break
        finally:
            pbar.close()

    total = train_tokens + val_tokens
    if total < TARGET_TOKENS:
        raise RuntimeError(
            f"Candidate shards supplied {total:,} tokens, below the "
            f"{TARGET_TOKENS:,}-token target."
        )
    elapsed = time.monotonic() - started
    print(
        f"Done in {elapsed / 3600:.2f} h: {total:,} tokens "
        f"({train_tokens:,} train, {val_tokens:,} validation) from "
        f"{shards_done} shards"
    )
    print(f"train.bin: {train_bin_path.stat().st_size / 1e9:.1f} GB")
    print(f"val.bin: {val_bin_path.stat().st_size / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for the prepared train and validation binaries.",
    )
    parser.add_argument("--workers", type=int, help="Tokenizer processes; defaults to host CPUs minus 16.")
    args = parser.parse_args()
    if args.workers is None:
        prepare_fineweb_edu_30b(args.output_dir)
    else:
        prepare_fineweb_edu_30b(args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
