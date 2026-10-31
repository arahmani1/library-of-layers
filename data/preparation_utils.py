"""Small primitives shared by binary dataset preparation scripts."""

from __future__ import annotations

import bisect
import hashlib
import json
import multiprocessing as mp
import os
import signal
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np

from huggingface_hub import hf_hub_download

_SENTINEL = object()


def configure_worker_process() -> None:
    """Configure a tokenizer worker without changing parent process state."""

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def atomic_write_json(path, payload) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n"
    )
    os.replace(temporary, path)


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def stable_hash_bucket(value, modulus, *, algorithm="sha256") -> int:
    """Map text to a deterministic bucket using an explicit hash."""

    digest = hashlib.new(algorithm, str(value).encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % int(modulus)


def download_hf_file(
    repo_id,
    filename,
    *,
    repo_type="dataset",
    revision=None,
    max_retries=8,
    backoff_base=5,
    backoff_cap=120,
):
    """Download through the Hub cache, retrying transient failures."""

    for attempt in range(max_retries):
        try:
            return hf_hub_download(
                repo_id,
                filename,
                repo_type=repo_type,
                revision=revision,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(backoff_base * 2**attempt, backoff_cap))


def _map_chunk(function, items):
    return [function(item) for item in items]


class OrderedProcessMapper:
    """One reusable process pool whose results preserve input order."""

    def __init__(self, workers, *, initializer=None, initargs=()):
        self.workers = int(workers)
        self.initializer = initializer
        self.initargs = initargs
        self.pool = None
        self.active_map = None

    def _start_pool(self):
        self.pool = mp.get_context("spawn").Pool(
            self.workers,
            initializer=self.initializer,
            initargs=self.initargs,
        )

    def __enter__(self):
        return self

    def _discard_map(self, active_map):
        source = active_map["source"]
        close = getattr(source, "close", None)
        if close is not None:
            close()
        for result in active_map["pending"]:
            try:
                result.get()
            except Exception:
                pass
        active_map["pending"].clear()
        if self.active_map is active_map:
            self.active_map = None

    def map(self, function, items, *, chunksize=1):
        if self.workers <= 1:
            yield from map(function, items)
            return
        if self.active_map is not None:
            raise RuntimeError("Previous ordered map is still active")
        if self.pool is None:
            self._start_pool()
        source = iter(items)
        pending = deque()
        active_map = {"source": source, "pending": pending}
        self.active_map = active_map
        exhausted = False
        pending_limit = 2 * self.workers
        chunk_size = max(1, int(chunksize))

        def fill_pending():
            nonlocal exhausted
            while not exhausted and len(pending) < pending_limit:
                chunk = tuple(islice(source, chunk_size))
                if not chunk:
                    exhausted = True
                    break
                pending.append(
                    self.pool.apply_async(_map_chunk, (function, chunk))
                )

        try:
            fill_pending()
            while pending:
                results = pending.popleft().get()
                fill_pending()
                yield from results
            self.active_map = None
        finally:
            if self.active_map is active_map:
                self._discard_map(active_map)

    def discard_pending(self):
        """Discard bounded speculative work without restarting workers."""

        if self.active_map is None:
            return
        self._discard_map(self.active_map)

    def __exit__(self, error_type, _error, _traceback):
        if self.pool is None:
            return
        if self.active_map is not None:
            self._discard_map(self.active_map)
        if error_type is None:
            self.pool.close()
        else:
            self.pool.terminate()
        self.pool.join()
        self.pool = None


def ordered_thread_map(items, function, workers, lookahead=None):
    """Run independent I/O calls concurrently and yield in input order.

    At most ``lookahead`` calls stay outstanding, so a fast producer cannot
    run arbitrarily far ahead of a slow consumer and stage the entire input.
    """

    source = iter(items)
    limit = max(1, int(workers if lookahead is None else lookahead))
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = deque()
    exhausted = False

    def submit_one():
        nonlocal exhausted
        if exhausted:
            return
        item = next(source, _SENTINEL)
        if item is _SENTINEL:
            exhausted = True
            return
        futures.append(executor.submit(function, item))

    try:
        while len(futures) < limit and not exhausted:
            submit_one()
        while futures:
            result = futures.popleft().result()
            submit_one()
            yield result
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


class RowPacker:
    """Best-fit packer emitting rows of exactly ``block_size`` tokens.

    ``open_capacity`` is a sorted ``(remaining, row_id)`` list, so placing a
    sequence is a bisect rather than a scan over every open row.
    """

    def __init__(
        self, block_size, min_length, *, pad_token_id=0,
        token_dtype=np.uint16, max_open_rows=65536, with_validity=False,
    ):
        self.block_size = int(block_size)
        self.min_length = int(min_length)
        self.pad_token_id = int(pad_token_id)
        self.token_dtype = np.dtype(token_dtype)
        self.max_open_rows = int(max_open_rows)
        self.with_validity = bool(with_validity)
        if not 1 <= self.min_length <= self.block_size:
            raise ValueError("Require 1 <= min_length <= block_size")
        if self.max_open_rows < 1:
            raise ValueError("max_open_rows must be positive")
        self.rows = {}
        self.fills = {}
        self.open_capacity = []
        self.next_row_id = 0
        self.n_sequences = 0
        self.n_rows = 0
        self.used_tokens = 0

    def _new_row(self):
        row_id = self.next_row_id
        self.next_row_id += 1
        self.rows[row_id] = np.full(
            self.block_size, self.pad_token_id, dtype=self.token_dtype
        )
        self.fills[row_id] = 0
        return row_id

    def _close(self, row_id):
        row = self.rows.pop(row_id)
        filled = self.fills.pop(row_id)
        self.used_tokens += filled
        self.n_rows += 1
        if self.with_validity:
            validity = np.zeros(self.block_size, dtype=np.uint8)
            validity[:filled] = 1
            return row, validity
        return row

    def add(self, tokens):
        """Place one sequence, yielding any row this placement completes."""
        length = len(tokens)
        if length > self.block_size:
            raise ValueError(
                f"Sequence of {length} tokens exceeds block_size "
                f"{self.block_size}."
            )
        index = bisect.bisect_left(self.open_capacity, (length, -1))
        if index == len(self.open_capacity):
            if len(self.rows) >= self.max_open_rows:
                # Evict the fullest open row rather than growing without
                # bound; it is the one least likely to take another sequence.
                _, victim = self.open_capacity.pop(0)
                yield self._close(victim)
            row_id = self._new_row()
            remaining = self.block_size
        else:
            remaining, row_id = self.open_capacity.pop(index)

        start = self.fills[row_id]
        self.rows[row_id][start:start + length] = tokens
        self.fills[row_id] = start + length
        self.n_sequences += 1
        remaining -= length
        if remaining >= self.min_length:
            bisect.insort(self.open_capacity, (remaining, row_id))
        else:
            yield self._close(row_id)

    def flush(self):
        """Emit every still-open row, padded to ``block_size``."""
        self.open_capacity.clear()
        for row_id in sorted(self.rows):
            yield self._close(row_id)

    @property
    def occupancy(self):
        total = self.n_rows * self.block_size
        return self.used_tokens / total if total else 0.0


def shuffled(records, buffer_size, rng):
    """Stream records through a fixed-size reservoir to decorrelate order.

    Record order depends only on the input stream, buffer size, and RNG.
    """
    if buffer_size < 1:
        raise ValueError("buffer_size must be positive")
    buffer = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue
        index = rng.randrange(buffer_size)
        buffer[index], record = record, buffer[index]
        yield record
    rng.shuffle(buffer)
    yield from buffer
