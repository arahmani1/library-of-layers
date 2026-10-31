"""English score-0 versus score-4/5 inventory and multilingual-compatible preparation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, closing, contextmanager
import fcntl
import hashlib
import heapq
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np

from data.balanced_corpus import PreparationError, compose_balanced_corpus, prepare_language_artifacts
from data.preparation_utils import (OrderedProcessMapper, atomic_write_json, configure_worker_process,
                                    download_hf_file, load_json, stable_hash_bucket)
from data.prepare_fineweb2_multilingual import (
    HuggingFaceTokenizerAdapter, TOKENIZER_ID, TOKENIZER_REVISION,
    TOKENIZER_VOCAB_SIZE, MODEL_VOCAB_SIZE, EOT_TOKEN_ID,
)

DATASET_ID = "locuslab/fineweb_annotated"
DATASET_REVISION = "d96861caa3a5c926f9f228871fb6e33871e9ea30"
CLASSES = {"unsafe": (5, 4), "safe": (0,)}
EOT_TEXT = "<|endoftext|>"
TEXT_CLEANUP = "strip_literal_eot_v1"
INVENTORY_FIELDS = ("hash", "text", "mask", "tokens", "heldout", "ordering", "url")
_inventory_tokenizer = None


class NativeOffsetTokenizer:
    """Lightweight packing-worker state; reuse the reference byte-accounting adapter."""
    def __init__(self, backend, eos_token_id, tokenizer_path=None):
        self.tokenizer_path = str(tokenizer_path) if tokenizer_path is not None else None
        self.vocab_size = backend.get_vocab_size(with_added_tokens=True)
        self.backend = backend
        self.backend.no_truncation()
        self.backend.no_padding()
        self.eos_token_id = eos_token_id

    def __len__(self):
        return self.vocab_size

    def __getstate__(self):
        return {**self.__dict__, "backend": None if self.tokenizer_path else self.backend}

    def __call__(self, text, **kwargs):
        if self.backend is None:
            from tokenizers import Tokenizer
            self.backend = Tokenizer.from_file(self.tokenizer_path)
            self.backend.no_truncation()
            self.backend.no_padding()
        encoded = self.backend.encode(text, add_special_tokens=False)
        return {"input_ids": encoded.ids, "offset_mapping": encoded.offsets}


@contextmanager
def progress_logging(output_dir, progress, interval=60):
    """One cheap heartbeat per minute, even during downloads or slow I/O."""
    stopped = threading.Event()
    started = time.monotonic()

    def report():
        while not stopped.wait(interval):
            fields = {"event": "progress", "elapsed_seconds": round(time.monotonic() - started), **progress.copy()}
            if fields.get("stage") in ("packing", "composition", "mixing_audit"):
                files = [output_dir / "artifacts" / name / filename
                         for name in CLASSES for filename in ("train.bin", "validation.bin")]
                files += [output_dir / ".staging" / "train.bin.tmp", output_dir / "train.bin"]
                written = {}
                for path in files:
                    try:
                        written[str(path.relative_to(output_dir))] = path.stat().st_size // 4
                    except FileNotFoundError:
                        pass  # A committed file can be renamed while sampled.
                fields["written_tokens"] = written
            print(json.dumps(fields), flush=True)

    thread = threading.Thread(target=report, name="preparation-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def open_inventory(path):
    db = sqlite3.connect(path)
    # Preserve existing assigned splits when opening a legacy local inventory.
    columns = db.execute("PRAGMA table_info(docs)").fetchall()
    if columns and [r[1] for r in columns if r[5]] == ["hash"]:
        db.executescript("""
            ALTER TABLE docs RENAME TO legacy_docs;
            CREATE TABLE docs (hash TEXT NOT NULL, text TEXT NOT NULL, mask INTEGER NOT NULL,
                tokens INTEGER NOT NULL, heldout INTEGER NOT NULL, ordering TEXT NOT NULL,
                PRIMARY KEY(hash,heldout));
            INSERT INTO docs SELECT * FROM legacy_docs;
            DROP TABLE legacy_docs;
        """)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS docs (
            hash TEXT NOT NULL, text TEXT NOT NULL, mask INTEGER NOT NULL,
            tokens INTEGER NOT NULL, heldout INTEGER NOT NULL, ordering TEXT NOT NULL,
            PRIMARY KEY(hash,heldout));
        CREATE INDEX IF NOT EXISTS inventory_counts ON docs(mask,heldout,tokens);
        CREATE TABLE IF NOT EXISTS urls (hash TEXT, url TEXT, PRIMARY KEY(hash,url));
        CREATE TABLE IF NOT EXISTS completed (path TEXT PRIMARY KEY);
    """)
    return db


def prepare_records(rows, score, tokenizer, seed=42, *, native=False):
    """Shared filtering/identity policy for reference and parallel counting."""
    accepted = []
    for row in rows:
        meta = row.get("metadata") or {}
        if meta.get("language") != "en" or not float(meta.get("language_score") or 0) >= .65:
            continue
        if (row.get("safety_metadata") or {}).get("final_harm_score") != score:
            raise PreparationError("Harm-score mismatch in selected shard")
        text = (row.get("text") or "").replace(EOT_TEXT, "")
        normalized = " ".join(unicodedata.normalize("NFC", text).split())
        if not normalized:
            continue
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        url = (meta.get("url") or "").strip() or digest
        heldout = int(stable_hash_bucket(f"{seed}:{url}", 100) == 0)
        order = hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()
        accepted.append((digest, text, url, heldout, order))
    texts = [r[1] for r in accepted]
    if native:
        encoded = (encoding.ids for encoding in tokenizer.encode_batch_fast(texts, add_special_tokens=False))
    else:
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"] if texts else []
    records = []
    for (digest, text, url, heldout, order), tokens in zip(accepted, encoded, strict=True):
        if EOT_TOKEN_ID in tokens:
            raise PreparationError("Embedded EOT in source text")
        records.append((digest, text, 1 << score, len(tokens) + 1, heldout, order, url))
    return records


def insert_records(db, records):
    db.executemany("""INSERT INTO docs VALUES (?,?,?,?,?,?) ON CONFLICT(hash,heldout)
        DO UPDATE SET mask=docs.mask|excluded.mask""",
        (record[:6] for record in records))
    db.executemany("INSERT OR IGNORE INTO urls VALUES (?,?)", ((record[0], record[6]) for record in records))


def ingest(db, rows, score, tokenizer, seed=42):
    """Readable reference path, with the same caller-owned transaction."""
    insert_records(db, prepare_records(rows, score, tokenizer, seed))


def initialize_inventory_worker(tokenizer_path):
    from tokenizers import Tokenizer
    global _inventory_tokenizer
    configure_worker_process()
    os.environ["RAYON_NUM_THREADS"] = "1"
    _inventory_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    _inventory_tokenizer.no_truncation()
    _inventory_tokenizer.no_padding()


def count_inventory_batch(item):
    rows, score, seed = item
    return len(rows), prepare_records(rows, score, _inventory_tokenizer, seed, native=True)


def inventory_shard(db, source_path, source_name, score, seed, checkpoint, progress,
                    *, mapper=None, tokenizer=None):
    """Commit a sequential shard checkpoint; the working SQLite stays disposable."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    metadata = {"format": "safety_inventory_v1", "source": source_name, "seed": str(seed),
                "dataset": DATASET_ID, "revision": DATASET_REVISION,
                "tokenizer": TOKENIZER_ID, "tokenizer_revision": TOKENIZER_REVISION,
                "text_cleanup": TEXT_CLEANUP}
    metadata = {key.encode(): value.encode() for key, value in metadata.items()}
    schema = pa.schema([(name, pa.int64() if name in ("mask", "tokens", "heldout") else pa.string())
                        for name in INVENTORY_FIELDS], metadata=metadata)
    if checkpoint.exists():
        progress.update(stage="restoring_checkpoint", source=source_name)
        parquet = pq.ParquetFile(checkpoint)
        stored = parquet.schema_arrow
        if not stored.remove_metadata().equals(schema.remove_metadata()) or any(
                (stored.metadata or {}).get(key, value if key == b"text_cleanup" else None) != value
                for key, value in metadata.items()):
            raise PreparationError(f"Incompatible inventory checkpoint: {checkpoint}")
        with db:
            for batch in parquet.iter_batches(batch_size=1024):
                insert_records(db, [tuple(row[key] for key in INVENTORY_FIELDS) for row in batch.to_pylist()])
            db.execute("INSERT INTO completed VALUES (?)", (source_name,))
        return {"restored": True, "inventory_seconds": time.monotonic() - started}

    parquet = pq.ParquetFile(source_path)
    progress.update(stage="counting", source=source_name, rows_done=0, rows_total=parquet.metadata.num_rows)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    # Project only required leaves: safety annotation explanations never enter IPC.
    columns = ["text", "metadata.language", "metadata.language_score", "metadata.url", "safety_metadata.final_harm_score"]
    batches = ((batch.to_pylist(), score, seed) for batch in parquet.iter_batches(batch_size=32, columns=columns))
    if mapper is None:
        results = ((len(rows), prepare_records(rows, value, tokenizer, split_seed)) for rows, value, split_seed in batches)
    else:
        results = mapper.map(count_inventory_batch, batches, chunksize=1)
    tokens_counted = eligible_rows = 0
    with db:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            pending = []

            def flush():
                insert_records(db, pending)
                writer.write_table(pa.Table.from_pylist([dict(zip(INVENTORY_FIELDS, record)) for record in pending], schema=schema))
                pending.clear()

            for rows_done, records in results:
                progress["rows_done"] += rows_done
                eligible_rows += len(records)
                tokens_counted += sum(record[3] for record in records)
                pending.extend(records)
                if len(pending) >= 1024:
                    flush()
            if pending:
                flush()
        progress.update(stage="checkpoint", source=source_name)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint)
        directory_fd = os.open(checkpoint.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        db.execute("INSERT INTO completed VALUES (?)", (source_name,))
    elapsed = time.monotonic() - started
    return {"restored": False, "inventory_seconds": elapsed, "tokens_counted": tokens_counted,
            "eligible_rows": eligible_rows, "input_rows": parquet.metadata.num_rows,
            "tokens_per_second": tokens_counted / elapsed, "checkpoint_bytes": checkpoint.stat().st_size}


def inventory_report(db, train_tokens, val_tokens):
    # Split membership is assigned at ingestion; no cross-split overlap cleanup.
    report = {"classes": {}, "cross_class_shared_texts": db.execute(
        """SELECT count(DISTINCT a.hash) FROM docs a JOIN docs b ON a.hash=b.hash
           WHERE (a.mask & 1) != 0 AND (b.mask & 48) != 0""").fetchone()[0]}
    for name, scores in CLASSES.items():
        mask = sum(1 << score for score in scores)
        counts = dict(db.execute("SELECT heldout,sum(tokens) FROM docs WHERE (mask & ?) != 0 GROUP BY heldout", (mask,)))
        train, val = counts.get(0, 0), counts.get(1, 0)
        report["classes"][name] = {"train_tokens": train, "validation_tokens": val,
            "train_shortfall": max(0, train_tokens-train), "validation_shortfall": max(0, val_tokens-val)}
    return report


def documents(db, scores, heldout):
    mask = sum(1 << score for score in scores)
    yield from db.execute("SELECT hash,text FROM docs WHERE (mask & ?) != 0 AND heldout=? ORDER BY ordering,hash", (mask, heldout))


def mixing_report(artifact_dir, classes, *, batch_size=26, block_size=1024, world_sizes=(1, 8), grad_accum_steps=5):
    """Reconstruct the shared composer's spans and audit contiguous loader slices."""
    lengths = [np.fromfile(Path(artifact_dir) / name / "train.lengths.bin", dtype=np.uint64) for name in classes]
    queue = [(0, i) for i in range(len(classes))]
    heapq.heapify(queue)
    indices = [0] * len(classes)
    ends, safe_ends = [0], [0]
    max_run = run = 0
    previous = None
    while queue:
        emitted, i = heapq.heappop(queue)
        count = int(lengths[i][indices[i]])
        indices[i] += 1
        ends.append(ends[-1] + count)
        safe_ends.append(safe_ends[-1] + (count if classes[i] == "safe" else 0))
        run = run + count if previous == i else count
        max_run = max(max_run, run)
        previous = i
        if indices[i] < len(lengths[i]):
            heapq.heappush(queue, (emitted + count, i))
    ends, safe_ends = np.asarray(ends), np.asarray(safe_ends)
    chunk = batch_size * block_size
    # Integral of safe-class membership, evaluated at each loader chunk boundary.
    points = np.arange(0, ((ends[-1] - 1) // chunk + 1) * chunk, chunk)
    segments = np.searchsorted(ends, points, side="right") - 1
    fractions = np.divide(np.diff(safe_ends), np.diff(ends))
    integral = safe_ends[segments] + (points - ends[segments]) * fractions[segments]
    safe_counts = np.diff(integral)
    report = {"batch_size": batch_size, "block_size": block_size,
              "grad_accum_steps": grad_accum_steps, "max_single_class_run_tokens": max_run, "world_sizes": {}}
    for world in world_sizes:
        steps = len(safe_counts) // (world * grad_accum_steps)
        used = safe_counts[:steps * world * grad_accum_steps].reshape(-1, world)
        ranks = []
        for rank in range(world):
            proportions = used[:, rank] / chunk
            ranks.append({"rank": rank, "microbatches": len(proportions),
                          "single_class_microbatches": int(np.count_nonzero((proportions == 0) | (proportions == 1))),
                          "safe_fraction_quantiles": np.quantile(proportions, [0, .05, .5, .95, 1]).tolist() if len(proportions) else []})
        report["world_sizes"][str(world)] = {"max_steps_without_wrap": steps, "ranks": ranks}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fineweb_safety"))
    parser.add_argument("--train-tokens-per-class", type=int, default=5_000_000_000)
    parser.add_argument("--validation-tokens-per-class", type=int, default=25_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scratch-dir", type=Path, default=Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir())))
    parser.add_argument("--reference-packing", action="store_true", help="Use the original HF packing tokenizer instead of native worker state")
    parser.add_argument("--reference-inventory", action="store_true", help="Use the readable serial HF counting path")
    parser.add_argument("--max-new-shards", type=int, help="Timing/resume check: stop after N new checkpoints; requires --inventory-only")
    parser.add_argument("--inventory-only", action="store_true", help="Count the selected class to its quotas without packing")
    parser.add_argument("--inventory-class", choices=tuple(CLASSES), default="unsafe", help="Class to count with --inventory-only")
    parser.add_argument("--checkpoint-source", type=Path, help="Read completed shard checkpoints from another preparation output")
    args = parser.parse_args()
    if args.train_tokens_per_class <= 0 or args.validation_tokens_per_class < 0 or args.workers < 1:
        parser.error("Token quotas and workers must be valid positive values (validation may be zero)")
    if args.max_new_shards is not None and (args.max_new_shards < 1 or not args.inventory_only):
        parser.error("--max-new-shards requires --inventory-only and a positive limit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = {"stage": "initializing"}
        with progress_logging(args.output_dir, progress):
            run(args, progress)


def run(args, progress=None):
    progress = {} if progress is None else progress
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    settings = {"format_version": 1, "dataset": DATASET_ID, "revision": DATASET_REVISION,
        "tokenizer": TOKENIZER_ID, "tokenizer_revision": TOKENIZER_REVISION,
        "seed": args.seed, "english_confidence": .65, "text_cleanup": TEXT_CLEANUP,
        "deduplication": "within_class_and_split",
        "class_scores": {name: list(scores) for name, scores in CLASSES.items()},
        "train_tokens_per_class": args.train_tokens_per_class,
        "validation_tokens_per_class": args.validation_tokens_per_class}
    settings_path = args.output_dir / "preparation.json"
    previous = load_json(settings_path)
    if previous is not None and previous != settings:
        upgraded = dict(previous)
        upgraded.setdefault("class_scores", settings["class_scores"])
        upgraded.setdefault("text_cleanup", TEXT_CLEANUP)
        if upgraded.get("deduplication") in (None, "within_class_with_global_split_isolation"):
            upgraded["deduplication"] = settings["deduplication"]
        if upgraded != settings or (args.output_dir / "artifacts").exists() or (args.output_dir / "manifest.json").exists():
            raise PreparationError("Resume settings changed; use a new output directory")
        # Old committed records passed embedded-EOT rejection, so cleanup is a no-op.
        # Only unpacked inventories may upgrade their source/cleanup policy.
        print("Upgrading compatible unpacked inventory settings", flush=True)
    atomic_write_json(settings_path, settings)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    if (len(tokenizer) != TOKENIZER_VOCAB_SIZE or tokenizer.eos_token_id != EOT_TOKEN_ID
            or tokenizer.eos_token != EOT_TEXT):
        raise PreparationError("Unexpected tokenizer vocabulary/EOT")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    packing_tokenizer = tokenizer if backend is None or getattr(args, "reference_packing", False) else NativeOffsetTokenizer(backend, tokenizer.eos_token_id)
    adapter = HuggingFaceTokenizerAdapter(packing_tokenizer)
    with ExitStack() as resources:
        scratch = getattr(args, "scratch_dir", None) or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
        work_dir = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix="fineweb-safety-", dir=scratch)))
        # Existing shared SQLite inventories are immutable legacy bases; copy their
        # rollback journal too so recovery happens locally rather than modifying the base.
        for filename in ("inventory.sqlite", "inventory.sqlite-journal"):
            if (args.output_dir / filename).exists():
                shutil.copyfile(args.output_dir / filename, work_dir / filename)
        db = resources.enter_context(closing(open_inventory(work_dir / "inventory.sqlite")))
        reference = getattr(args, "reference_inventory", False)
        tokenizer_path = work_dir / "tokenizer.json"
        if not reference or isinstance(packing_tokenizer, NativeOffsetTokenizer):
            tokenizer.backend_tokenizer.save(str(tokenizer_path))
        if isinstance(packing_tokenizer, NativeOffsetTokenizer):
            packing_tokenizer.tokenizer_path = str(tokenizer_path)
        if not reference:
            if args.workers == 1:
                initialize_inventory_worker(tokenizer_path)
        files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset", revision=DATASET_REVISION)
        new_shards = 0
        with OrderedProcessMapper(args.workers, initializer=initialize_inventory_worker,
                                  initargs=(str(tokenizer_path),)) as mapper:
            pending_classes = [getattr(args, "inventory_class", "unsafe")] if args.inventory_only else list(CLASSES)
            while pending_classes:
                name = pending_classes.pop(0)
                scores = CLASSES[name]
                shards = []
                for score in scores:
                    selected = sorted(p for p in files if p.startswith(f"score_{score}/") and p.endswith(".parquet"))
                    if not selected:
                        raise PreparationError(f"No score_{score} shards found")
                    shards.extend((score, path) for path in selected)
                report = inventory_report(db, args.train_tokens_per_class, args.validation_tokens_per_class)
                for i, (score, shard) in enumerate(shards):
                    if not any(report["classes"][name][key] for key in ("train_shortfall", "validation_shortfall")):
                        break
                    if db.execute("SELECT 1 FROM completed WHERE path=?", (shard,)).fetchone():
                        continue
                    timing = {}
                    shard_started = time.monotonic()
                    if not db.execute("SELECT 1 FROM completed WHERE path=?", (shard,)).fetchone():
                        checkpoint = args.output_dir / "inventory_shards" / shard
                        external = getattr(args, "checkpoint_source", None)
                        if not checkpoint.exists() and external is not None:
                            candidate = external / "inventory_shards" / shard
                            if candidate.exists():
                                checkpoint = candidate
                        progress.clear()
                        progress.update(source=shard, shard=i+1, total_shards=len(shards))
                        local = None
                        if not checkpoint.exists():
                            progress.update(stage="download")
                            print(json.dumps(progress), flush=True)
                            download_started = time.monotonic()
                            local = download_hf_file(DATASET_ID, shard, revision=DATASET_REVISION)
                            timing["download_seconds"] = time.monotonic() - download_started
                        timing.update(inventory_shard(db, local, shard, score, args.seed, checkpoint, progress,
                                                      mapper=None if reference else mapper, tokenizer=tokenizer))
                        if not timing["restored"]:
                            new_shards += 1
                    progress.update(stage="deduplication_and_split_counts", source=shard)
                    report = inventory_report(db, args.train_tokens_per_class, args.validation_tokens_per_class)
                    report["completed_shards"] = [r[0] for r in db.execute("SELECT path FROM completed ORDER BY path")]
                    timing["shard_seconds_including_split_checks"] = time.monotonic() - shard_started
                    report["last_shard_timing"] = {"source": shard, **timing}
                    atomic_write_json(args.output_dir / "inventory.json", report)
                    print(json.dumps({"class": name, "shard": i+1, "total_shards": len(shards),
                                      **report["classes"][name], **timing}), flush=True)
                    if getattr(args, "max_new_shards", None) and new_shards >= args.max_new_shards:
                        progress.update(stage="inventory_limit_reached")
                        print(json.dumps(progress), flush=True)
                        return
                if any(report["classes"][name][key] for key in ("train_shortfall", "validation_shortfall")):
                    raise PreparationError(f"{name} unique-token shortfall; see {args.output_dir / 'inventory.json'}")
                if args.inventory_only:
                    return
        artifact_dir = args.output_dir / "artifacts"
        progress.clear()
        progress.update(stage="packing", workers=args.workers)
        print(json.dumps(progress), flush=True)
        prepare_language_artifacts(
            {name: documents(db, scores, 0) for name, scores in CLASSES.items()},
            {name: documents(db, scores, 1) for name, scores in CLASSES.items()},
            artifact_dir, adapter, train_tokens_by_language={name: args.train_tokens_per_class for name in CLASSES},
            validation_tokens_per_language=args.validation_tokens_per_class, workers=args.workers,
            dataset_id=DATASET_ID, dataset_revision=DATASET_REVISION, model_vocab_size=MODEL_VOCAB_SIZE)
        progress.update(stage="composition")
        print(json.dumps(progress), flush=True)
        compose_balanced_corpus(artifact_dir, args.output_dir, tuple(CLASSES),
            target_tokens=2*args.train_tokens_per_class, validation_tokens_per_language=args.validation_tokens_per_class,
            dataset_id=DATASET_ID, dataset_revision=DATASET_REVISION, tokenizer_id=TOKENIZER_ID,
            tokenizer_revision=TOKENIZER_REVISION, tokenizer_vocab_size=TOKENIZER_VOCAB_SIZE,
            model_vocab_size=MODEL_VOCAB_SIZE, eot_token_id=EOT_TOKEN_ID)
        progress.update(stage="mixing_audit")
        print(json.dumps(progress), flush=True)
        atomic_write_json(args.output_dir / "mixing.json", mixing_report(artifact_dir, tuple(CLASSES)))
        progress.update(stage="complete")
        print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()
