"""Prepare an exactly balanced multilingual token corpus."""

from __future__ import annotations

import heapq
import shutil
from pathlib import Path

import numpy as np

from data.preparation_utils import (
    OrderedProcessMapper,
    atomic_write_json,
    configure_worker_process,
    load_json,
)

TOKEN_DTYPE = np.dtype(np.uint32)
LENGTH_DTYPE = np.dtype(np.uint64)


class PreparationError(RuntimeError):
    pass


_worker_tokenizer = None


def _init_worker(tokenizer):
    global _worker_tokenizer
    configure_worker_process()
    _worker_tokenizer = tokenizer


def _tokenize(document):
    document_id, text = document
    tokens, byte_lengths = _worker_tokenizer.encode_with_byte_lengths(text)
    return str(document_id), tokens, byte_lengths


def _write_token_quota(
    mapper,
    source,
    quota,
    tokenizer,
    *,
    language,
    split,
    workers,
    token_file,
    length_file=None,
    byte_file=None,
):
    count = 0
    if quota == 0:
        return
    encoded = (
        mapper.map(_tokenize, source, chunksize=32)
        if workers > 1
        else (
            (
                str(document_id),
                *tokenizer.encode_with_byte_lengths(text),
            )
            for document_id, text in source
        )
    )
    for document_id, tokens, byte_lengths in encoded:
        if len(tokens) != len(byte_lengths):
            raise PreparationError(
                f"{language}/{document_id}: expected one byte length per token"
            )
        if tokenizer.eos_token_id in tokens:
            raise PreparationError(
                f"{language}/{document_id}: document contains EOT token"
            )
        tokens = [tokenizer.eos_token_id, *tokens]
        byte_lengths = [0, *byte_lengths]
        take = min(len(tokens), quota - count)
        token_file.write(np.asarray(tokens[:take], dtype=TOKEN_DTYPE).tobytes())
        if length_file is not None:
            length_file.write(np.asarray([take], dtype=LENGTH_DTYPE).tobytes())
        if byte_file is not None:
            byte_file.write(np.asarray(byte_lengths[:take], dtype=TOKEN_DTYPE).tobytes())
        count += take
        if count == quota:
            break
    else:
        raise PreparationError(
            f"{language}: available {split}={count:,}, "
            f"required {split}={quota:,}"
        )
    if workers > 1:
        mapper.discard_pending()


def prepare_balanced_corpus(
    train_documents_by_language,
    validation_documents_by_language,
    output_dir,
    tokenizer,
    *,
    target_tokens,
    validation_tokens_per_language,
    workers,
    dataset_id,
    dataset_revision,
    model_vocab_size,
):
    """Write exact per-language quotas and return the completed manifest."""

    languages = tuple(train_documents_by_language)
    target_tokens = int(target_tokens)
    validation_quota = int(validation_tokens_per_language)
    if not languages:
        raise ValueError("At least one language is required")
    if set(validation_documents_by_language) != set(languages):
        raise ValueError(
            "Training and validation sources must contain the same languages"
        )
    if target_tokens <= 0 or target_tokens % len(languages):
        raise ValueError(
            "target_tokens must be positive and divisible by language count"
        )
    if validation_quota < 0:
        raise ValueError("validation_tokens_per_language must be nonnegative")
    train_quota = target_tokens // len(languages)
    output_dir = Path(output_dir)
    staging_dir = output_dir / ".staging"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": {"tokens": "train.bin", "tokens_count": target_tokens},
        **{
            language: {
                "tokens": f"val.{language}.bin",
                "byte_lengths": f"val.{language}.bytes.bin",
                "tokens_count": validation_quota,
            }
            for language in languages
            if validation_quota > 0
        },
    }
    expected_manifest = {
        "format_version": 1,
        "dataset": {"id": dataset_id, "revision": dataset_revision},
        "tokenizer": {
            "id": getattr(tokenizer, "tokenizer_id", "test/tokenizer"),
            "revision": getattr(
                tokenizer, "tokenizer_revision", "test-revision"
            ),
            "vocab_size": int(tokenizer.vocab_size),
            "model_vocab_size": int(model_vocab_size),
            "eot_token_id": int(tokenizer.eos_token_id),
        },
        "token_dtype": "uint32",
        "languages": list(languages),
        "language_token_counts": dict.fromkeys(languages, train_quota),
        "splits": splits,
    }
    manifest = load_json(manifest_path)
    if manifest is not None:
        if manifest != expected_manifest:
            raise PreparationError(
                "Existing manifest does not match the requested corpus"
            )
        for split, metadata in manifest["splits"].items():
            expected_size = int(metadata["tokens_count"]) * TOKEN_DTYPE.itemsize
            for key in ("tokens", "byte_lengths"):
                filename = metadata.get(key)
                if filename is None:
                    continue
                path = output_dir / filename
                if not path.is_file():
                    raise PreparationError(
                        f"Manifest split {split!r} is missing committed file "
                        f"{path}"
                    )
                actual_size = path.stat().st_size
                if actual_size != expected_size:
                    raise PreparationError(
                        f"Committed file {path} has {actual_size} bytes; "
                        f"expected {expected_size}"
                    )
        return manifest

    staging_dir.mkdir(exist_ok=True)

    paths = {
        language: {
            "train": staging_dir / f"train.{language}.bin",
            "lengths": staging_dir / f"train.{language}.lengths.bin",
            "validation": output_dir / f"val.{language}.bin",
            "validation_bytes": output_dir / f"val.{language}.bytes.bin",
        }
        for language in languages
    }

    with OrderedProcessMapper(
        workers,
        initializer=_init_worker,
        initargs=(tokenizer,),
    ) as mapper:
        for language in languages:
            language_paths = paths[language]
            with (
                language_paths["train"].open("wb", buffering=8 * 1024 * 1024) as train_file,
                language_paths["lengths"].open("wb", buffering=8 * 1024 * 1024) as lengths_file,
                language_paths["validation"].open("wb", buffering=8 * 1024 * 1024) as validation_file,
                language_paths["validation_bytes"].open("wb", buffering=8 * 1024 * 1024) as byte_file,
            ):
                _write_token_quota(
                    mapper,
                    train_documents_by_language[language],
                    train_quota,
                    tokenizer,
                    language=language,
                    split="train",
                    workers=workers,
                    token_file=train_file,
                    length_file=lengths_file,
                )
                _write_token_quota(
                    mapper,
                    validation_documents_by_language[language],
                    validation_quota,
                    tokenizer,
                    language=language,
                    split="validation",
                    workers=workers,
                    token_file=validation_file,
                    byte_file=byte_file,
                )

    token_streams = [
        np.memmap(paths[language]["train"], dtype=TOKEN_DTYPE, mode="r")
        for language in languages
    ]
    document_lengths = [
        np.memmap(paths[language]["lengths"], dtype=LENGTH_DTYPE, mode="r")
        for language in languages
    ]
    document_indices = [0] * len(languages)
    token_offsets = [0] * len(languages)
    queue = [(0, index) for index in range(len(languages))]
    temporary_path = staging_dir / "train.bin.tmp"
    with temporary_path.open("wb", buffering=8 * 1024 * 1024) as output:
        while queue:
            emitted, index = heapq.heappop(queue)
            count = int(document_lengths[index][document_indices[index]])
            offset = token_offsets[index]
            output.write(token_streams[index][offset:offset + count].tobytes())
            document_indices[index] += 1
            token_offsets[index] += count
            emitted += count
            if emitted < train_quota:
                heapq.heappush(queue, (emitted, index))
    temporary_path.replace(output_dir / "train.bin")

    atomic_write_json(manifest_path, expected_manifest)
    for path in staging_dir.iterdir():
        path.unlink()
    staging_dir.rmdir()
    return expected_manifest


def _language_artifact_manifest(
    language,
    train_tokens,
    validation_tokens,
    tokenizer,
    *,
    dataset_id,
    dataset_revision,
    model_vocab_size,
):
    return {
        "format_version": 1,
        "kind": "balanced_corpus_language",
        "dataset": {"id": dataset_id, "revision": dataset_revision},
        "tokenizer": {
            "id": getattr(tokenizer, "tokenizer_id", "test/tokenizer"),
            "revision": getattr(
                tokenizer, "tokenizer_revision", "test-revision"
            ),
            "vocab_size": int(tokenizer.vocab_size),
            "model_vocab_size": int(model_vocab_size),
            "eot_token_id": int(tokenizer.eos_token_id),
        },
        "token_dtype": "uint32",
        "length_dtype": "uint64",
        "language": language,
        "train": {
            "tokens": "train.bin",
            "document_lengths": "train.lengths.bin",
            "tokens_count": int(train_tokens),
        },
        "validation": {
            "tokens": "validation.bin",
            "byte_lengths": "validation.bytes.bin",
            "tokens_count": int(validation_tokens),
        },
    }


def _validate_language_artifact(path, manifest):
    path = Path(path)
    train = manifest["train"]
    validation = manifest["validation"]
    expected_sizes = {
        train["tokens"]: int(train["tokens_count"]) * TOKEN_DTYPE.itemsize,
        validation["tokens"]: (
            int(validation["tokens_count"]) * TOKEN_DTYPE.itemsize
        ),
        validation["byte_lengths"]: (
            int(validation["tokens_count"]) * TOKEN_DTYPE.itemsize
        ),
    }
    for filename, expected_size in expected_sizes.items():
        artifact = path / filename
        if not artifact.is_file():
            raise PreparationError(
                f"Language artifact is missing committed file {artifact}"
            )
        if artifact.stat().st_size != expected_size:
            raise PreparationError(
                f"Language artifact {artifact} has {artifact.stat().st_size} "
                f"bytes; expected {expected_size}"
            )

    lengths_path = path / train["document_lengths"]
    if not lengths_path.is_file():
        raise PreparationError(
            f"Language artifact is missing committed file {lengths_path}"
        )
    if lengths_path.stat().st_size % LENGTH_DTYPE.itemsize:
        raise PreparationError(
            f"Language artifact has invalid document lengths {lengths_path}"
        )
    lengths = np.memmap(lengths_path, dtype=LENGTH_DTYPE, mode="r")
    if len(lengths) == 0 or int(lengths.sum()) != int(train["tokens_count"]):
        raise PreparationError(
            f"Language artifact document lengths do not sum to "
            f"{int(train['tokens_count']):,}"
        )


def prepare_language_artifacts(
    train_documents_by_language,
    validation_documents_by_language,
    output_dir,
    tokenizer,
    *,
    train_tokens_by_language,
    validation_tokens_per_language,
    workers,
    dataset_id,
    dataset_revision,
    model_vocab_size,
):
    """Prepare committed reusable token streams for individual languages."""

    languages = tuple(train_documents_by_language)
    train_quotas = {
        language: int(tokens)
        for language, tokens in train_tokens_by_language.items()
    }
    validation_quota = int(validation_tokens_per_language)
    if not languages:
        raise ValueError("At least one language is required")
    if set(train_quotas) != set(languages):
        raise ValueError(
            "Training sources and language token quotas must match"
        )
    if set(validation_documents_by_language) != set(languages):
        raise ValueError(
            "Training and validation sources must contain the same languages"
        )
    if any(quota <= 0 for quota in train_quotas.values()):
        raise ValueError("Language training-token quotas must be positive")
    if validation_quota < 0:
        raise ValueError("validation_tokens_per_language must be nonnegative")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        language: _language_artifact_manifest(
            language,
            train_quotas[language],
            validation_quota,
            tokenizer,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            model_vocab_size=model_vocab_size,
        )
        for language in languages
    }
    incomplete = []
    for language in languages:
        language_dir = output_dir / language
        manifest_path = language_dir / "manifest.json"
        existing = load_json(manifest_path)
        if existing is None:
            incomplete.append(language)
            continue
        if existing != manifests[language]:
            raise PreparationError(
                f"Existing language artifact for {language} does not match "
                "the requested corpus"
            )
        _validate_language_artifact(language_dir, existing)

    with OrderedProcessMapper(
        workers,
        initializer=_init_worker,
        initargs=(tokenizer,),
    ) as mapper:
        for language in incomplete:
            language_dir = output_dir / language
            language_dir.mkdir(parents=True, exist_ok=True)
            manifest = manifests[language]
            with (
                (language_dir / manifest["train"]["tokens"]).open("wb", buffering=8 * 1024 * 1024)
                as train_file,
                (language_dir / manifest["train"]["document_lengths"]).open(
                    "wb", buffering=8 * 1024 * 1024
                ) as lengths_file,
                (language_dir / manifest["validation"]["tokens"]).open("wb", buffering=8 * 1024 * 1024)
                as validation_file,
                (language_dir / manifest["validation"]["byte_lengths"]).open(
                    "wb", buffering=8 * 1024 * 1024
                ) as byte_file,
            ):
                _write_token_quota(
                    mapper,
                    train_documents_by_language[language],
                    train_quotas[language],
                    tokenizer,
                    language=language,
                    split="train",
                    workers=workers,
                    token_file=train_file,
                    length_file=lengths_file,
                )
                _write_token_quota(
                    mapper,
                    validation_documents_by_language[language],
                    validation_quota,
                    tokenizer,
                    language=language,
                    split="validation",
                    workers=workers,
                    token_file=validation_file,
                    byte_file=byte_file,
                )
            _validate_language_artifact(language_dir, manifest)
            atomic_write_json(language_dir / "manifest.json", manifest)
    return manifests


def _prefix_document_limit(lengths, quota):
    remaining = int(quota)
    chunk_size = 1_000_000
    for start in range(0, len(lengths), chunk_size):
        chunk = np.asarray(lengths[start:start + chunk_size])
        cumulative = np.cumsum(chunk, dtype=np.uint64)
        chunk_tokens = int(cumulative[-1])
        if chunk_tokens < remaining:
            remaining -= chunk_tokens
            continue
        local_index = int(np.searchsorted(cumulative, remaining, side="left"))
        before = int(cumulative[local_index - 1]) if local_index else 0
        return start + local_index + 1, remaining - before
    raise PreparationError(
        f"Language artifact has fewer than the required {int(quota):,} tokens"
    )


def compose_balanced_corpus(
    artifact_dir,
    output_dir,
    languages,
    *,
    target_tokens,
    validation_tokens_per_language,
    dataset_id,
    dataset_revision,
    tokenizer_id,
    tokenizer_revision,
    tokenizer_vocab_size,
    model_vocab_size,
    eot_token_id,
):
    """Compose an exact balanced corpus from committed language artifacts."""

    languages = tuple(languages)
    target_tokens = int(target_tokens)
    validation_quota = int(validation_tokens_per_language)
    if not languages:
        raise ValueError("At least one language is required")
    if target_tokens <= 0 or target_tokens % len(languages):
        raise ValueError(
            "target_tokens must be positive and divisible by language count"
        )
    if validation_quota < 0:
        raise ValueError("validation_tokens_per_language must be nonnegative")
    train_quota = target_tokens // len(languages)
    splits = {
        "train": {"tokens": "train.bin", "tokens_count": target_tokens},
        **{
            language: {
                "tokens": f"val.{language}.bin",
                "byte_lengths": f"val.{language}.bytes.bin",
                "tokens_count": validation_quota,
            }
            for language in languages
            if validation_quota > 0
        },
    }
    expected_manifest = {
        "format_version": 1,
        "dataset": {"id": dataset_id, "revision": dataset_revision},
        "tokenizer": {
            "id": tokenizer_id,
            "revision": tokenizer_revision,
            "vocab_size": int(tokenizer_vocab_size),
            "model_vocab_size": int(model_vocab_size),
            "eot_token_id": int(eot_token_id),
        },
        "token_dtype": "uint32",
        "languages": list(languages),
        "language_token_counts": dict.fromkeys(languages, train_quota),
        "splits": splits,
    }
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    existing = load_json(manifest_path)
    if existing is not None:
        if existing != expected_manifest:
            raise PreparationError(
                "Existing manifest does not match the requested corpus"
            )
        for metadata in existing["splits"].values():
            expected_size = int(metadata["tokens_count"]) * TOKEN_DTYPE.itemsize
            for key in ("tokens", "byte_lengths"):
                filename = metadata.get(key)
                if filename is None:
                    continue
                path = output_dir / filename
                if not path.is_file() or path.stat().st_size != expected_size:
                    raise PreparationError(
                        f"Committed corpus file {path} is missing or invalid"
                    )
        return existing

    artifact_dir = Path(artifact_dir)
    expected_tokenizer = expected_manifest["tokenizer"]
    token_streams = []
    length_streams = []
    document_limits = []
    artifact_manifests = []
    for language in languages:
        language_dir = artifact_dir / language
        artifact_manifest = load_json(language_dir / "manifest.json")
        if artifact_manifest is None:
            raise PreparationError(
                f"Language artifact for {language} is incomplete"
            )
        try:
            compatible = (
                artifact_manifest.get("format_version") == 1
                and artifact_manifest.get("kind")
                == "balanced_corpus_language"
                and artifact_manifest.get("dataset")
                == expected_manifest["dataset"]
                and artifact_manifest.get("tokenizer") == expected_tokenizer
                and artifact_manifest.get("token_dtype") == "uint32"
                and artifact_manifest.get("length_dtype") == "uint64"
                and artifact_manifest.get("language") == language
                and int(artifact_manifest["train"]["tokens_count"])
                >= train_quota
                and int(artifact_manifest["validation"]["tokens_count"])
                == validation_quota
            )
        except (KeyError, TypeError, ValueError):
            compatible = False
        if not compatible:
            raise PreparationError(
                f"Language artifact for {language} is incompatible"
            )
        _validate_language_artifact(language_dir, artifact_manifest)
        artifact_manifests.append(artifact_manifest)
        tokens = np.memmap(
            language_dir / artifact_manifest["train"]["tokens"],
            dtype=TOKEN_DTYPE,
            mode="r",
        )
        lengths = np.memmap(
            language_dir / artifact_manifest["train"]["document_lengths"],
            dtype=LENGTH_DTYPE,
            mode="r",
        )
        token_streams.append(tokens)
        length_streams.append(lengths)
        document_limits.append(_prefix_document_limit(lengths, train_quota))

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / ".staging"
    if staging_dir.exists():
        for path in staging_dir.iterdir():
            if not path.is_file():
                raise PreparationError(
                    f"Unexpected directory in composition staging: {path}"
                )
            path.unlink()
    else:
        staging_dir.mkdir()

    document_indices = [0] * len(languages)
    token_offsets = [0] * len(languages)
    queue = [(0, index) for index in range(len(languages))]
    temporary_path = staging_dir / "train.bin.tmp"
    with temporary_path.open("wb", buffering=8 * 1024 * 1024) as output:
        while queue:
            emitted, index = heapq.heappop(queue)
            document_count, last_length = document_limits[index]
            document_index = document_indices[index]
            count = (
                last_length
                if document_index == document_count - 1
                else int(length_streams[index][document_index])
            )
            offset = token_offsets[index]
            output.write(token_streams[index][offset:offset + count].tobytes())
            document_indices[index] += 1
            token_offsets[index] += count
            emitted += count
            if emitted < train_quota:
                heapq.heappush(queue, (emitted, index))
    temporary_path.replace(output_dir / "train.bin")

    if validation_quota > 0:
        for language, artifact_manifest in zip(
            languages, artifact_manifests, strict=True
        ):
            language_dir = artifact_dir / language
            validation = artifact_manifest["validation"]
            for source_key, suffix in (
                ("tokens", ".bin"),
                ("byte_lengths", ".bytes.bin"),
            ):
                temporary = staging_dir / f"val.{language}{suffix}.tmp"
                shutil.copyfile(
                    language_dir / validation[source_key], temporary
                )
                temporary.replace(output_dir / f"val.{language}{suffix}")

    atomic_write_json(manifest_path, expected_manifest)
    for path in staging_dir.iterdir():
        path.unlink()
    staging_dir.rmdir()
    return expected_manifest
