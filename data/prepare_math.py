"""Prepare a unique, document-packed math mixture (never starts training).

The default scans all eligible records before quota selection. --early-quota
instead scans complete source prefixes until unique train quotas are filled;
validation and duplicate propagation cover the scanned prefixes. That mode
prefetches downloads, writes durable per-shard token/metadata streams, and
rebuilds its disposable node-local SQLite index after interruption.
"""
from __future__ import annotations

import argparse
import fnmatch
import fcntl
import hashlib
import json
import math
import os
import random
import sqlite3
import unicodedata
from itertools import chain, islice
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi

from data.preparation_utils import (
    OrderedProcessMapper, RowPacker, atomic_write_json, configure_worker_process,
    download_hf_file, load_json, ordered_thread_map, shuffled, stable_hash_bucket,
)

SOURCES = (
    ("4plus", "nvidia/Nemotron-CC-Math-v1", "4plus"),
    ("4plus_MIND", "nvidia/Nemotron-CC-Math-v1", "4plus_MIND"),
    ("openmath", "nvidia/OpenMathInstruct-2", "default"),
)
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
_TOKENIZER = None
_SETTINGS = None


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def identity(text):
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()


def encode_record(item, tokenizer, settings):
    """Return identity, emitted-content identity, and complete retained tokens."""
    source, record = item
    if source == "openmath":
        problem, solution = record["problem"], record["generated_solution"]
        if not isinstance(problem, str) or not isinstance(solution, str):
            return None
        if not normalized(problem) or not normalized(solution):
            return None
        group = identity(problem)
        text = f"Question:\n{problem}\n\nSolution:\n{solution}"
    else:
        text = record["text"]
        if not isinstance(text, str) or not normalized(text):
            return None
        group = identity(text)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    eot = settings["eot_token_id"]
    if not tokens or eot in tokens:
        return None
    if min(tokens) < 0 or max(tokens) >= settings["vocab_size"]:
        raise ValueError("Tokenizer emitted an out-of-vocabulary token")
    limit = settings["block_size"] - 1
    if len(tokens) > limit:
        if source == "openmath":
            return None
        tokens = tokens[:limit]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
    return group, identity(text), [eot, *tokens]


def initialize_worker(settings, worker=True):
    global _TOKENIZER, _SETTINGS
    from transformers import AutoTokenizer
    if worker:
        configure_worker_process()
    _SETTINGS = settings
    _TOKENIZER = AutoTokenizer.from_pretrained(
        settings["tokenizer_id"], revision=settings["tokenizer_revision"],
        use_fast=True,
    )


def encode_worker(item):
    return encode_record(item, _TOKENIZER, _SETTINGS)


def resolve_sources(api):
    """Resolve published train-file patterns against immutable Hub revisions."""
    result = []
    resolved = {}
    for name, repo_id, config in SOURCES:
        if repo_id not in resolved:
            resolved[repo_id] = api.dataset_info(repo_id)
        info = resolved[repo_id]
        card = info.card_data.to_dict() if info.card_data else {}
        configs = card.get("configs", [])
        selected = next((c for c in configs if c["config_name"] == config), None)
        if selected is None:
            raise ValueError(f"Missing published data_files for {repo_id}/{config}")
        patterns = []
        for entry in selected["data_files"]:
            if entry["split"] == "train":
                paths = entry["path"]
                patterns.extend([paths] if isinstance(paths, str) else paths)
        files = sorted(f for f in api.list_repo_files(repo_id, repo_type="dataset", revision=info.sha)
                       if f.endswith(".parquet") and any(fnmatch.fnmatch(f, p) for p in patterns))
        if not files:
            raise ValueError(f"No published train parquet files for {repo_id}/{config}")
        result.append(dict(name=name, id=repo_id, config=config, split="train",
                           revision=info.sha, files=files))
    return result


def source_records(source):
    import pyarrow.parquet as pq
    columns = ["problem", "generated_solution"] if source["name"] == "openmath" else ["text"]
    for filename in source["files"]:
        path = download_hf_file(source["id"], filename, revision=source["revision"])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns):
            yield from batch.to_pylist()


def stage_records(db, sources, settings, mapper, records=None, encode=None):
    """Each transaction checkpoints both accepted records and consumed offsets."""
    records = source_records if records is None else records
    encode = encode_worker if encode is None else encode
    db.executescript("""
        CREATE TABLE IF NOT EXISTS groups (id TEXT PRIMARY KEY, val INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, source TEXT NOT NULL, group_id TEXT NOT NULL,
            content TEXT NOT NULL, tokens BLOB NOT NULL, length INTEGER NOT NULL,
            UNIQUE(content, group_id));
        CREATE INDEX IF NOT EXISTS content_index ON documents(content);
        CREATE INDEX IF NOT EXISTS group_index ON documents(group_id);
        CREATE TABLE IF NOT EXISTS progress (source TEXT PRIMARY KEY, offset INTEGER, done INTEGER);
    """)
    dtype = np.dtype(settings["token_dtype"])
    for source in sources:
        name = source["name"]
        progress = db.execute("SELECT offset, done FROM progress WHERE source=?", (name,)).fetchone()
        offset, done = progress or (0, False)
        if done:
            continue
        stream = ((name, r) for r in islice(records(source), offset, None))
        encoded = mapper.map(encode, stream, chunksize=32)
        while True:
            batch = list(islice(encoded, settings["checkpoint_records"]))
            if not batch:
                break
            with db:
                for value in batch:
                    if value is None:
                        continue
                    group, content, tokens = value
                    val = stable_hash_bucket(f'{settings["seed"]}:{group}', 10**9) < int(settings["validation_fraction"] * 10**9)
                    db.execute("INSERT OR IGNORE INTO groups VALUES (?, ?)", (group, int(val)))
                    db.execute("INSERT OR IGNORE INTO documents(source, group_id, content, tokens, length) VALUES (?, ?, ?, ?, ?)",
                               (name, group, content, np.asarray(tokens, dtype=dtype).tobytes(), len(tokens)))
                offset += len(batch)
                db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, 0)", (name, offset))
        with db:
            db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, 1)", (name, offset))
    propagate_validation(db)


def propagate_validation(db):
    """Promote every identity linked to validation by emitted content."""
    with db:
        while True:
            changed = db.execute("""
                UPDATE groups SET val=1 WHERE val=0 AND id IN (
                    SELECT d.group_id FROM documents d JOIN documents other USING(content)
                    JOIN groups g ON g.id=other.group_id WHERE g.val=1)
            """).rowcount
            if not changed:
                break


def select_records(db, settings, *, allow_incomplete=False):
    """Choose canonical unique records before selecting source token prefixes."""
    with db:
        db.executescript("""
            DROP TABLE IF EXISTS selected;
            CREATE TABLE selected (id INTEGER PRIMARY KEY, split TEXT NOT NULL);
            DROP TABLE IF EXISTS canonical;
            CREATE TABLE canonical AS SELECT MIN(id) AS id FROM documents GROUP BY content;
            CREATE UNIQUE INDEX canonical_id ON canonical(id);
        """)
        db.execute("""INSERT INTO selected SELECT d.id, 'val' FROM documents d
                      JOIN canonical c USING(id) JOIN groups g ON g.id=d.group_id WHERE g.val=1""")
        counts = {}
        incomplete = []
        for name, quota in zip((s[0] for s in SOURCES), settings["quotas"]):
            count = 0
            reached = quota == 0
            cursor = db.execute("""SELECT d.id, d.length FROM documents d JOIN canonical c USING(id)
                JOIN groups g ON g.id=d.group_id WHERE d.source=? AND g.val=0 ORDER BY d.id""", (name,))
            for doc_id, length in cursor:
                if reached:
                    break
                if count + length > quota and not allow_incomplete:
                    reached = True
                    break
                db.execute("INSERT INTO selected VALUES (?, 'train')", (doc_id,))
                count += length
                if count >= quota:
                    reached = True
                    break
            if not reached:
                if allow_incomplete:
                    incomplete.append(name)
                else:
                    raise ValueError(f"Source {name} exhausted: {count:,} eligible unique tokens for quota {quota:,}; examples are never repeated")
            counts[name] = {"quota": quota, "content_tokens": count, "shortfall": max(0, quota - count)}
    return (counts, incomplete) if allow_incomplete else counts


def mixed_records(db, split, settings, out=None):
    if out is None:
        query = """SELECT d.tokens, d.length FROM documents d JOIN selected s USING(id)
            WHERE s.split=? AND d.source=? ORDER BY d.id"""
    else:
        query = """SELECT d.token_file, d.token_offset, d.length FROM documents d
            JOIN selected s USING(id) WHERE s.split=? AND d.source=? ORDER BY d.id"""
    cursors = [iter(db.execute(query, (split, source[0]))) for source in SOURCES]
    consumed = [0, 0, 0]
    active = set(range(len(cursors)))
    handles = {}
    try:
        while active:
            index = min(active, key=lambda i: (consumed[i] / settings["shares"][i], i))
            value = next(cursors[index], None)
            if value is None:
                active.remove(index)
                continue
            if out is None:
                blob, length = value
            else:
                filename, offset, length = value
                if filename not in handles:
                    handles[filename] = (out / filename).open("rb")
                handle = handles[filename]
                handle.seek(offset)
                blob = handle.read(length * np.dtype(settings["token_dtype"]).itemsize)
                if len(blob) != length * np.dtype(settings["token_dtype"]).itemsize:
                    raise ValueError(f"Truncated staged tokens in {filename}")
            consumed[index] += length
            yield np.frombuffer(blob, dtype=settings["token_dtype"])
    finally:
        for handle in handles.values():
            handle.close()


def write_packed(db, out, settings, *, spooled=False):
    splits = {}
    for split in ("train", "val"):
        packer = RowPacker(settings["block_size"], 2,
                           pad_token_id=settings["eot_token_id"], token_dtype=np.dtype(settings["token_dtype"]),
                           max_open_rows=settings["max_open_rows"], with_validity=True)
        token_path, valid_path = out / f"{split}.bin", out / f"{split}.validity.bin"
        hashes = [hashlib.sha256(), hashlib.sha256()]
        records = shuffled(mixed_records(db, split, settings, out if spooled else None), settings["shuffle_buffer"], random.Random(settings["seed"]))
        rows = chain.from_iterable(packer.add(record) for record in records)
        with token_path.open("wb") as tokens, valid_path.open("wb") as validity:
            for row, valid in chain(rows, packer.flush()):
                for handle, digest, array in zip((tokens, validity), hashes, (row, valid)):
                    blob = array.tobytes()
                    handle.write(blob)
                    digest.update(blob)
            for handle, digest, array in zip((tokens, validity), hashes, (
                np.asarray([settings["eot_token_id"]], dtype=settings["token_dtype"]),
                np.asarray([0], dtype=np.uint8),
            )):
                blob = array.tobytes()
                handle.write(blob)
                digest.update(blob)
        count = packer.n_rows * settings["block_size"] + 1
        if token_path.stat().st_size != count * np.dtype(settings["token_dtype"]).itemsize or valid_path.stat().st_size != count:
            raise ValueError(f"Invalid artifact sizes for {split}")
        splits[split] = dict(tokens=token_path.name, validity=valid_path.name,
                             tokens_count=count, content_tokens=packer.used_tokens,
                             documents=packer.n_sequences, rows=packer.n_rows,
                             sha256={"tokens": hashes[0].hexdigest(), "validity": hashes[1].hexdigest()})
    if not splits["train"]["rows"] or not splits["val"]["rows"]:
        raise ValueError("Preparation requires nonempty training and validation splits")
    return splits


def _iter_shard_rows(path, source):
    import pyarrow.parquet as pq

    columns = ["problem", "generated_solution"] if source["name"] == "openmath" else ["text"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=columns):
        yield from batch.to_pylist()


def _stage_paths(out, source_name, index):
    stem = out / ".stage" / f"{source_name}-{index:04d}"
    return (stem.with_suffix(".tokens.bin"), stem.with_suffix(".records.parquet"),
            stem.with_suffix(".done.json"))


def _digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _init_stream_db(db, cache_mib):
    # This index is disposable: committed shard artifacts are the durable source.
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
    db.executescript("""
        CREATE TABLE groups (id TEXT PRIMARY KEY, val INTEGER NOT NULL);
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY, source TEXT NOT NULL, group_id TEXT NOT NULL,
            content TEXT NOT NULL, token_file TEXT NOT NULL, token_offset INTEGER NOT NULL,
            length INTEGER NOT NULL, UNIQUE(content, group_id));
        CREATE INDEX group_index ON documents(group_id);
    """)


def _insert_stream_rows(db, source_name, token_file, rows, settings):
    threshold = int(settings["validation_fraction"] * 10**9)
    with db:
        db.executemany("INSERT OR IGNORE INTO groups VALUES (?, ?)",
                       ((group, int(stable_hash_bucket(f'{settings["seed"]}:{group}', 10**9) < threshold))
                        for _, group, _, _, _ in rows))
        db.executemany("""INSERT OR IGNORE INTO documents
            (id, source, group_id, content, token_file, token_offset, length)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ((doc_id, source_name, group, content, token_file, offset, length)
             for doc_id, group, content, offset, length in rows))


def _write_stream_shard(db, out, source, rank, index, ordinal, path, settings, mapper):
    import pyarrow as pa
    import pyarrow.parquet as pq

    token_path, metadata_path, marker_path = _stage_paths(out, source["name"], index)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_tmp = token_path.with_suffix(token_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    token_tmp.unlink(missing_ok=True)
    metadata_tmp.unlink(missing_ok=True)
    schema = pa.schema([("id", pa.int64()), ("group_id", pa.string()),
                        ("content", pa.string()), ("token_offset", pa.int64()),
                        ("length", pa.int32())])
    rows = []
    count = 0
    documents = 0
    raw_tokens = 0
    token_hash = hashlib.sha256()
    token_file = token_path.relative_to(out).as_posix()

    def flush(writer):
        if not rows:
            return
        writer.write_table(pa.Table.from_pylist([
            dict(id=doc_id, group_id=group, content=content,
                 token_offset=offset, length=length)
            for doc_id, group, content, offset, length in rows
        ], schema=schema))
        _insert_stream_rows(db, source["name"], token_file, rows, settings)
        rows.clear()

    with token_tmp.open("wb") as tokens:
        writer = pq.ParquetWriter(metadata_tmp, schema, compression=None)
        try:
            stream = ((source["name"], record) for record in _iter_shard_rows(path, source))
            for count, value in enumerate(mapper.map(encode_worker, stream, chunksize=32), 1):
                if value is None:
                    continue
                group, content, encoded = value
                blob = np.asarray(encoded, dtype=settings["token_dtype"]).tobytes()
                offset = tokens.tell()
                tokens.write(blob)
                token_hash.update(blob)
                rows.append(((rank + 1) * 10**12 + ordinal + count,
                             group, content, offset, len(encoded)))
                documents += 1
                raw_tokens += len(encoded)
                if len(rows) >= settings["checkpoint_records"]:
                    flush(writer)
            flush(writer)
        finally:
            writer.close()
        tokens.flush()
        os.fsync(tokens.fileno())
    with metadata_tmp.open("rb") as metadata:
        os.fsync(metadata.fileno())
    os.replace(token_tmp, token_path)
    os.replace(metadata_tmp, metadata_path)
    marker = dict(version=1, source=source["name"], filename=source["files"][index],
                  rows=count, documents=documents, raw_tokens=raw_tokens,
                  token_file=token_file, metadata_file=metadata_path.relative_to(out).as_posix(),
                  token_bytes=token_path.stat().st_size, metadata_bytes=metadata_path.stat().st_size,
                  token_sha256=token_hash.hexdigest(), metadata_sha256=_digest_file(metadata_path))
    atomic_write_json(marker_path, marker)
    print(f'{source["name"]} shard {index + 1}/{len(source["files"])}: '
          f'{count:,} rows, {documents:,} retained, {raw_tokens:,} raw tokens', flush=True)
    return marker


def _load_stream_markers(db, out, sources, settings):
    import pyarrow.parquet as pq

    totals = [0] * len(sources)
    ordinals = [0] * len(sources)
    next_indices = [0] * len(sources)
    for rank, source in enumerate(sources):
        for index, filename in enumerate(source["files"]):
            token_path, metadata_path, marker_path = _stage_paths(out, source["name"], index)
            marker = load_json(marker_path)
            if marker is None:
                if any(_stage_paths(out, source["name"], later)[2].exists()
                       for later in range(index + 1, len(source["files"]))):
                    raise ValueError(f'Noncontiguous shard checkpoints for {source["name"]}')
                break
            if (marker.get("version") != 1 or marker.get("source") != source["name"]
                    or marker.get("filename") != filename
                    or marker.get("token_file") != token_path.relative_to(out).as_posix()
                    or marker.get("metadata_file") != metadata_path.relative_to(out).as_posix()):
                raise ValueError(f'Incompatible shard checkpoint {marker_path}')
            for artifact, size_key, hash_key in ((token_path, "token_bytes", "token_sha256"),
                                                 (metadata_path, "metadata_bytes", "metadata_sha256")):
                if artifact.stat().st_size != marker[size_key] or _digest_file(artifact) != marker[hash_key]:
                    raise ValueError(f'Invalid staged shard artifact {artifact}')
            token_file = marker["token_file"]
            seen = 0
            for batch in pq.ParquetFile(metadata_path).iter_batches(batch_size=settings["checkpoint_records"]):
                columns = batch.to_pydict()
                rows = list(zip(columns["id"], columns["group_id"], columns["content"],
                                columns["token_offset"], columns["length"]))
                _insert_stream_rows(db, source["name"], token_file, rows, settings)
                seen += len(rows)
            if seen != marker["documents"]:
                raise ValueError(f'Invalid staged shard metadata {metadata_path}')
            totals[rank] += marker["raw_tokens"]
            ordinals[rank] += marker["rows"]
            next_indices[rank] += 1
    return totals, ordinals, next_indices


def _prepare_quota(args, out, settings, sources):
    scratch = Path(args.scratch_dir)
    if scratch.resolve() == out.resolve() or out.resolve() in scratch.resolve().parents:
        raise ValueError('Local scratch must be outside the output directory')
    scratch.mkdir(parents=True, exist_ok=True)
    db_path = scratch / "staging.sqlite"
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as db:
        _init_stream_db(db, args.sqlite_cache_mib)
        raw_tokens, ordinals, next_indices = _load_stream_markers(db, out, sources, settings)
        downloads = {}
        with OrderedProcessMapper(args.workers, initializer=initialize_worker, initargs=(settings,)) as mapper:
            if args.workers == 1:
                initialize_worker(settings, worker=False)

            def process_next(rank):
                source = sources[rank]
                index = next_indices[rank]
                if index >= len(source["files"]):
                    return False
                if rank not in downloads:
                    downloads[rank] = ordered_thread_map(
                        source["files"][index:],
                        lambda filename, src=source: download_hf_file(
                            src["id"], filename, revision=src["revision"]),
                        args.download_workers, lookahead=args.download_workers)
                path = next(downloads[rank])
                marker = _write_stream_shard(db, out, source, rank, index,
                                             ordinals[rank], path, settings, mapper)
                next_indices[rank] += 1
                ordinals[rank] += marker["rows"]
                raw_tokens[rank] += marker["raw_tokens"]
                return True

            try:
                for rank, quota in enumerate(settings["quotas"]):
                    while raw_tokens[rank] < quota:
                        if not process_next(rank):
                            raise ValueError(f'Source {sources[rank]["name"]} exhausted before raw quota')
                while True:
                    propagate_validation(db)
                    counts, incomplete = select_records(db, settings, allow_incomplete=True)
                    if not incomplete:
                        break
                    for name in incomplete:
                        rank = next(i for i, source in enumerate(sources) if source["name"] == name)
                        if not process_next(rank):
                            raise ValueError(f'Source {name} exhausted before unique train quota')
            finally:
                for iterator in downloads.values():
                    iterator.close()
        splits = write_packed(db, out, settings, spooled=True)
    return counts, splits


def _prepare(args):
    out = Path(args.output_dir)
    early_quota = bool(getattr(args, "early_quota", False))
    requested = {key: value for key, value in vars(args).items()
                 if key not in ("output_dir", "workers", "download_workers", "scratch_dir", "sqlite_cache_mib")}
    if not early_quota:
        requested.pop("early_quota", None)
    if early_quota and (args.download_workers < 1 or args.sqlite_cache_mib < 1):
        raise ValueError("Download workers and SQLite cache must be positive")
    if args.token_budget <= 0 or args.block_size < 2 or args.workers < 1:
        raise ValueError("Budget and workers must be positive; block size must be at least 2")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("Validation fraction must lie strictly between 0 and 1")
    if any(not math.isfinite(s) or s <= 0 for s in args.shares) or not math.isclose(sum(args.shares), 1.0, abs_tol=1e-9):
        raise ValueError("Three positive source shares must sum to 1")
    if min(args.shuffle_buffer, args.max_open_rows, args.checkpoint_records) < 1:
        raise ValueError("Packing buffers and checkpoint interval must be positive")
    checkpoint = load_json(out / "preparation.json")
    if checkpoint:
        if checkpoint.get("version") != (2 if early_quota else 1) or checkpoint["requested"] != requested:
            raise ValueError("Incompatible preparation resume settings")
        settings, sources = checkpoint["settings"], checkpoint["sources"]
        if (out / "manifest.json").exists():
            manifest = load_json(out / "manifest.json")
            for split in manifest["splits"].values():
                if ((out / split["tokens"]).stat().st_size != split["tokens_count"] * np.dtype(settings["token_dtype"]).itemsize
                        or (out / split["validity"]).stat().st_size != split["tokens_count"]):
                    raise ValueError("Published artifacts have invalid sizes")
                for key in ("tokens", "validity"):
                    digest = hashlib.sha256()
                    with (out / split[key]).open("rb") as handle:
                        for chunk in iter(lambda: handle.read(8 << 20), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != split["sha256"][key]:
                        raise ValueError("Published artifact checksum mismatch")
            return manifest
    else:
        if any(path.name != ".prepare.lock" for path in out.iterdir()):
            raise FileExistsError("Refusing to overwrite existing output without preparation checkpoint")
        from transformers import AutoTokenizer
        api = HfApi()
        revision = api.model_info(args.tokenizer, revision=args.tokenizer_revision).sha
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=revision, use_fast=True)
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS/document delimiter")
        sources = resolve_sources(api)
        quotas = [int(args.token_budget * s) for s in args.shares]
        quotas[-1] = args.token_budget - sum(quotas[:-1])
        settings = dict(requested, tokenizer_id=args.tokenizer, tokenizer_revision=revision,
                        vocab_size=len(tokenizer), eot_token_id=tokenizer.eos_token_id,
                        token_dtype="uint16" if len(tokenizer) <= 65536 else "uint32", quotas=quotas)
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / "preparation.json", dict(version=2 if early_quota else 1,
                          requested=requested, settings=settings, sources=sources))
    if early_quota:
        counts, splits = _prepare_quota(args, out, settings, sources)
    else:
        with sqlite3.connect(out / "staging.sqlite") as db:
            with OrderedProcessMapper(args.workers, initializer=initialize_worker, initargs=(settings,)) as mapper:
                if args.workers == 1:
                    initialize_worker(settings, worker=False)
                stage_records(db, sources, settings, mapper)
            counts = select_records(db, settings)
            splits = write_packed(db, out, settings)
    manifest = dict(format_version=1, dataset={"id": "math-mixture", "sources": sources},
                    token_dtype=settings["token_dtype"],
                    tokenizer={"id": settings["tokenizer_id"], "revision": settings["tokenizer_revision"],
                               "vocab_size": settings["vocab_size"], "eot_token_id": settings["eot_token_id"]},
                    packing={"block_size": settings["block_size"], "pad_token_id": settings["eot_token_id"],
                             "policy": "shuffled best-fit, whole retained documents", "seed": settings["seed"]},
                    preparation=settings, source_counts=counts, splits=splits)
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def prepare(args):
    """Hold exclusive output ownership through staging and manifest publication."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".prepare.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another preparation owns {out}") from exc
        return _prepare(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-budget", type=int, default=10_000_000_000, help="Training content + delimiters; use 50000000000 for 50B")
    parser.add_argument("--shares", nargs=3, type=float, default=[0.4, 0.4, 0.2], metavar=("4PLUS", "MIND", "OPENMATH"))
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--early-quota", action="store_true",
                        help="Stop after scanned source prefixes satisfy train quotas")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--scratch-dir", default="/tmp/latentreasoning-math-10b")
    parser.add_argument("--sqlite-cache-mib", type=int, default=1024)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--max-open-rows", type=int, default=256)
    parser.add_argument("--checkpoint-records", type=int, default=4096)
    manifest = prepare(parser.parse_args())
    print(json.dumps(manifest["source_counts"], indent=2))


if __name__ == "__main__":
    main()
