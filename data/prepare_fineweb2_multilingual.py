"""Prepare an exactly balanced multilingual FineWeb2 token corpus.

Each language contributes ``target_tokens / len(languages)`` training tokens
without repeating documents. FineWeb2's native test splits provide separate
monolingual validation data. Documents are staged per language, then
interleaved deterministically.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

from data.balanced_corpus import (
    PreparationError,
    compose_balanced_corpus,
    prepare_balanced_corpus,
    prepare_language_artifacts,
)
from data.preparation_utils import download_hf_file, ordered_thread_map

DATASET_ID = "HuggingFaceFW/fineweb-2"
DATASET_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
TOKENIZER_ID = "allenai/Olmo-3-1025-7B"
TOKENIZER_REVISION = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
TOKENIZER_VOCAB_SIZE = 100278
MODEL_VOCAB_SIZE = 100352
EOT_TOKEN_ID = 100257
LANGUAGES = (
    "rus_Cyrl", "cmn_Hani", "spa_Latn", "arb_Arab", "hin_Deva",
    "deu_Latn", "jpn_Jpan", "kor_Hang", "tur_Latn", "fra_Latn",
)


class HuggingFaceTokenizerAdapter:
    """Attach exact UTF-8 byte counts to fast-tokenizer output."""

    def __init__(
        self, tokenizer, tokenizer_id=TOKENIZER_ID,
        tokenizer_revision=TOKENIZER_REVISION,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id
        self.tokenizer_revision = tokenizer_revision
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.vocab_size = int(len(tokenizer))

    def encode_with_byte_lengths(self, text):
        encoded = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
        )
        tokens = [int(token) for token in encoded["input_ids"]]
        if not tokens:
            if text:
                raise PreparationError(
                    "Tokenizer produced no tokens for nonempty text"
                )
            return [], []

        offsets = encoded.get("offset_mapping")
        if offsets is None or len(offsets) != len(tokens):
            raise PreparationError("Tokenizer must return one offset per token")
        byte_offsets = [0]
        for character in text:
            byte_offsets.append(
                byte_offsets[-1] + len(character.encode("utf-8"))
            )
        covered = 0
        byte_lengths = []
        for offset in offsets:
            if not isinstance(offset, (tuple, list)) or len(offset) != 2:
                raise PreparationError("Tokenizer returned an invalid offset")
            start, end = (int(value) for value in offset)
            if not 0 <= start <= end <= len(text) or end < covered:
                raise PreparationError("Tokenizer offsets are out of order")
            byte_lengths.append(byte_offsets[end] - byte_offsets[covered])
            covered = end
        byte_lengths[-1] += byte_offsets[-1] - byte_offsets[covered]
        return tokens, byte_lengths


def _fineweb2_shards(language, split, *, api=None):
    from huggingface_hub import HfApi

    api = HfApi() if api is None else api
    prefix = f"data/{language}/{split}"
    shards = sorted(
        item.path
        for item in api.list_repo_tree(
            DATASET_ID,
            path_in_repo=prefix,
            recursive=True,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )
        if item.path.endswith(".parquet")
    )
    if not shards:
        raise PreparationError(
            f"No FineWeb2 {split} parquet shards found for {language}"
        )
    return shards


def _fineweb2_documents(language, split, download_workers):
    import pyarrow.parquet as pq

    download = partial(
        download_hf_file,
        DATASET_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    shards = _fineweb2_shards(language, split)
    for path in ordered_thread_map(shards, download, download_workers):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=2000, columns=["id", "text"]
        ):
            yield from zip(
                batch.column("id").to_pylist(),
                batch.column("text").to_pylist(),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--direct", action="store_true",
        help="Prepare the balanced ten-language corpus directly.",
    )
    modes.add_argument(
        "--stage-languages",
        nargs="+",
        metavar="LANGUAGE=TOKENS",
        help="Prepare reusable per-language token artifacts.",
    )
    modes.add_argument(
        "--compose", action="store_true",
        help="Compose the ten-language corpus from reusable artifacts.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        help="Total training tokens for direct or composed corpora.",
    )
    parser.add_argument(
        "--validation-tokens-per-language",
        type=int,
        default=25_000_000,
        help="Validation-token quota for each language (default: 25000000).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Committed per-language artifact directory for composition.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Corpus output directory."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 16),
        help="Tokenizer worker processes.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Concurrent shard downloads (default: 8).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved plan and exit."
    )
    args = parser.parse_args()

    if args.validation_tokens_per_language < 0:
        parser.error("--validation-tokens-per-language must be nonnegative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.download_workers < 1:
        parser.error("--download-workers must be at least 1")

    if args.stage_languages is not None:
        if args.target_tokens is not None:
            parser.error("--target-tokens is not used with --stage-languages")
        if args.artifact_dir is not None:
            parser.error("--artifact-dir is not used with --stage-languages")
        language_quotas = {}
        available_languages = set(LANGUAGES)
        for specification in args.stage_languages:
            language, separator, raw_tokens = specification.partition("=")
            try:
                tokens = int(raw_tokens)
            except ValueError:
                tokens = 0
            if (
                separator != "="
                or language not in available_languages
                or tokens <= 0
                or language in language_quotas
            ):
                parser.error(
                    "--stage-languages entries must be unique supported "
                    "LANGUAGE=positive-token-count pairs"
                )
            language_quotas[language] = tokens
        mode = "stage"
        languages = tuple(language_quotas)
        plan = {
            "mode": mode,
            "dataset": f"{DATASET_ID}@{DATASET_REVISION}",
            "tokenizer": f"{TOKENIZER_ID}@{TOKENIZER_REVISION}",
            "language_token_counts": language_quotas,
            "validation_tokens_per_language": (
                args.validation_tokens_per_language
            ),
            "output_dir": str(args.output_dir),
        }
    elif args.compose:
        if args.target_tokens is None or args.target_tokens <= 0:
            parser.error(
                "--target-tokens must be positive with --compose"
            )
        if args.artifact_dir is None:
            parser.error("--artifact-dir is required with --compose")
        if args.target_tokens % len(LANGUAGES):
            parser.error("--target-tokens must be divisible by ten")
        mode = "compose"
        languages = LANGUAGES
        plan = {
            "mode": mode,
            "dataset": f"{DATASET_ID}@{DATASET_REVISION}",
            "tokenizer": f"{TOKENIZER_ID}@{TOKENIZER_REVISION}",
            "languages": languages,
            "target_tokens": args.target_tokens,
            "validation_tokens_per_language": (
                args.validation_tokens_per_language
            ),
            "artifact_dir": str(args.artifact_dir),
            "output_dir": str(args.output_dir),
        }
    else:
        if args.target_tokens is None or args.target_tokens <= 0:
            parser.error(
                "--target-tokens must be positive with --direct"
            )
        if args.artifact_dir is not None:
            parser.error("--artifact-dir is not used with --direct")
        languages = LANGUAGES
        if args.target_tokens % len(languages):
            parser.error(
                "--target-tokens must be divisible by ten"
            )
        mode = "direct"
        plan = {
            "mode": mode,
            "dataset": f"{DATASET_ID}@{DATASET_REVISION}",
            "tokenizer": f"{TOKENIZER_ID}@{TOKENIZER_REVISION}",
            "languages": languages,
            "target_tokens": args.target_tokens,
            "tokens_per_language": args.target_tokens // len(languages),
            "validation_tokens_per_language": (
                args.validation_tokens_per_language
            ),
            "output_dir": str(args.output_dir),
        }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    if mode == "compose":
        compose_balanced_corpus(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            languages=languages,
            target_tokens=args.target_tokens,
            validation_tokens_per_language=args.validation_tokens_per_language,
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            tokenizer_id=TOKENIZER_ID,
            tokenizer_revision=TOKENIZER_REVISION,
            tokenizer_vocab_size=TOKENIZER_VOCAB_SIZE,
            model_vocab_size=MODEL_VOCAB_SIZE,
            eot_token_id=EOT_TOKEN_ID,
        )
        return

    from transformers import AutoTokenizer

    tokenizer = HuggingFaceTokenizerAdapter(
        AutoTokenizer.from_pretrained(
            TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True,
        )
    )
    if (
        tokenizer.vocab_size != TOKENIZER_VOCAB_SIZE
        or tokenizer.eos_token_id != EOT_TOKEN_ID
    ):
        raise PreparationError(
            f"Unexpected tokenizer metadata: vocab={tokenizer.vocab_size}, "
            f"eot={tokenizer.eos_token_id}"
        )
    train_documents = {
        language: _fineweb2_documents(
            language, "train", args.download_workers
        )
        for language in languages
    }
    validation_documents = {
        language: _fineweb2_documents(
            language, "test", args.download_workers
        )
        for language in languages
    }
    if mode == "stage":
        prepare_language_artifacts(
            train_documents_by_language=train_documents,
            validation_documents_by_language=validation_documents,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            train_tokens_by_language=language_quotas,
            validation_tokens_per_language=(
                args.validation_tokens_per_language
            ),
            workers=args.workers,
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            model_vocab_size=MODEL_VOCAB_SIZE,
        )
    else:
        prepare_balanced_corpus(
            train_documents_by_language=train_documents,
            validation_documents_by_language=validation_documents,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            target_tokens=args.target_tokens,
            validation_tokens_per_language=(
                args.validation_tokens_per_language
            ),
            workers=args.workers,
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            model_vocab_size=MODEL_VOCAB_SIZE,
        )


if __name__ == "__main__":
    main()
