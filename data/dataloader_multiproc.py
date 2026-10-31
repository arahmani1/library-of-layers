"""Memory-mapped token batches with packed-document metadata."""

import json
from pathlib import Path

import numpy as np
import torch


def compute_doc_meta(idx: torch.Tensor, eot_token_id: int, input_valid=None):
    """Return document and per-document position IDs for packed tokens.

    EOT is the first token of a document. A truncated document tail before the
    first EOT in a loader window is treated as document zero starting at
    position zero.
    """

    _, length = idx.shape
    is_eot = idx == eot_token_id
    if input_valid is not None:
        is_eot = is_eot & input_valid
    doc_id = is_eot.long().cumsum(dim=1)
    offsets = torch.arange(length, device=idx.device).unsqueeze(0).expand_as(idx)
    eot_offsets = torch.where(is_eot, offsets, torch.full_like(offsets, -1))
    last_eot = eot_offsets.cummax(dim=1).values.clamp(min=0)
    return doc_id, offsets - last_eot


class DataLoaderLit_MultiProc:
    def __init__(
        self,
        data_path,
        batch_size,
        block_size,
        split="train",
        process_rank=0,
        num_processes=1,
        eot_token_id=None,
        model_vocab_size=None,
    ):
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.process_rank = int(process_rank)
        self.num_processes = int(num_processes)

        data_path = Path(data_path)
        manifest_path = data_path / "manifest.json"
        self.manifest = None
        split_metadata = None
        dtype = np.dtype(np.uint16)
        token_path = data_path / f"{split}.bin"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest.get("format_version") != 1:
                raise ValueError(
                    "Data manifest format_version must be 1, got "
                    f"{self.manifest.get('format_version')!r}."
                )
            packing = self.manifest.get("packing")
            if packing is not None:
                try:
                    packed_block_size = int(packing["block_size"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "Data manifest packing is missing a valid block_size"
                    ) from exc
                if packed_block_size <= 0:
                    raise ValueError(
                        "Data manifest packing block_size must be positive, "
                        f"got {packed_block_size}."
                    )
                if self.block_size <= 0 or self.block_size % packed_block_size:
                    raise ValueError(
                        f"Configured block_size {self.block_size} must be a "
                        "positive multiple of data manifest packing "
                        f"block_size {packed_block_size}."
                    )
            try:
                dtype = np.dtype(self.manifest["token_dtype"])
            except (KeyError, TypeError) as exc:
                raise ValueError("Data manifest is missing token_dtype") from exc
            if dtype not in {np.dtype(np.uint16), np.dtype(np.uint32)}:
                raise ValueError(
                    "Data manifest token_dtype must be uint16 or uint32, "
                    f"got {dtype}."
                )
            split_metadata = self.manifest.get("splits", {}).get(split)
            if split_metadata is None:
                available = ", ".join(sorted(self.manifest.get("splits", {})))
                raise ValueError(
                    f"Unknown manifest split {split!r}; available: {available}"
                )
            token_path = data_path / split_metadata["tokens"]
            tokenizer = self.manifest["tokenizer"]
            manifest_eot = int(tokenizer["eot_token_id"])
            if eot_token_id is not None and int(eot_token_id) != manifest_eot:
                raise ValueError(
                    f"Configured EOT {eot_token_id} does not match data "
                    f"manifest EOT {manifest_eot}."
                )
            eot_token_id = manifest_eot
            if model_vocab_size is not None:
                model_vocab_size = int(model_vocab_size)
                if model_vocab_size < int(tokenizer["vocab_size"]):
                    raise ValueError(
                        f"Model vocabulary {model_vocab_size} is smaller than "
                        f"data tokenizer vocabulary {tokenizer['vocab_size']}."
                    )
                manifest_model_vocab = tokenizer.get("model_vocab_size")
                if (
                    manifest_model_vocab is not None
                    and model_vocab_size != int(manifest_model_vocab)
                ):
                    raise ValueError(
                        f"Model padded vocabulary {model_vocab_size} does not "
                        f"match data manifest padded vocabulary "
                        f"{manifest_model_vocab}."
                    )

        self.eot_token_id = int(0 if eot_token_id is None else eot_token_id)
        self.data = np.memmap(token_path, dtype=dtype, mode="r")
        if split_metadata is not None:
            try:
                expected_tokens = int(split_metadata["tokens_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Manifest split {split!r} is missing a valid tokens_count"
                ) from exc
            if len(self.data) != expected_tokens:
                raise ValueError(
                    f"{token_path} contains {len(self.data)} tokens, but the "
                    f"manifest claims {expected_tokens}."
                )
        self.validity = None
        if split_metadata is not None and split_metadata.get("validity"):
            validity_path = data_path / split_metadata["validity"]
            self.validity = np.memmap(validity_path, dtype=np.uint8, mode="r")
            if len(self.validity) != len(self.data):
                raise ValueError(
                    f"Validity sidecar length {len(self.validity)} does not "
                    f"match token length {len(self.data)} for split {split!r}."
                )

        self.byte_lengths = None
        if split_metadata is not None and split_metadata.get("byte_lengths"):
            byte_path = data_path / split_metadata["byte_lengths"]
            self.byte_lengths = np.memmap(byte_path, dtype=np.uint32, mode="r")
            if len(self.byte_lengths) != len(self.data):
                raise ValueError(
                    f"Byte sidecar length {len(self.byte_lengths)} does not "
                    f"match token length {len(self.data)} for split {split!r}."
                )

        global_batch_tokens = (
            self.batch_size * self.block_size * self.num_processes
        )
        self.total_batches = (len(self.data) - 1) // global_batch_tokens
        self.initial_position = (
            self.batch_size * self.block_size * self.process_rank
        )
        self.current_position = self.initial_position
        if self.process_rank == 0:
            print(
                f"Loaded {split} split: {len(self.data):,} tokens. "
                f"(~{self.total_batches} global batches)"
            )

    def _next_window(self, *, include_bytes=False, include_input_valid=False):
        chunk_size = self.batch_size * self.block_size
        if self.current_position + chunk_size + 1 > len(self.data):
            self.current_position = chunk_size * self.process_rank
        start = self.current_position
        end = start + chunk_size
        self.current_position += chunk_size * self.num_processes

        tensors = [
            torch.from_numpy(values.astype(np.int64)).view(
                self.batch_size, self.block_size
            )
            for values in (self.data[start:end], self.data[start + 1:end + 1])
        ]
        input_valid = None
        if self.validity is not None:
            input_valid = torch.from_numpy(
                self.validity[start:end].astype(np.bool_)
            ).view(self.batch_size, self.block_size)
            if bool((input_valid[:, 1:] & ~input_valid[:, :-1]).any()):
                raise ValueError("Validity sidecar must have trailing padding per row")
            target_valid = torch.from_numpy(
                self.validity[start + 1:end + 1].astype(np.bool_)
            ).view(self.batch_size, self.block_size)
            # The first padding EOT also closes the preceding real document.
            supervised = input_valid & (target_valid | (tensors[1] == self.eot_token_id))
            tensors[1].masked_fill_(~supervised, -100)
        if include_bytes:
            if self.byte_lengths is None:
                raise ValueError("This split has no byte-length sidecar")
            tensors.append(
                torch.from_numpy(
                    self.byte_lengths[start + 1:end + 1].astype(np.int64)
                ).view(self.batch_size, self.block_size)
            )
        if include_bytes and self.validity is not None:
            tensors[2].masked_fill_(tensors[1] == -100, 0)
        if include_input_valid:
            tensors.append(input_valid)
        return tuple(tensors)

    @staticmethod
    def _to_device(values, device):
        if torch.device(device).type == "cpu":
            return tuple(None if value is None else value.to(device) for value in values)
        return tuple(
            None if value is None else value.pin_memory().to(device, non_blocking=True)
            for value in values
        )

    def get_batch(self, device):
        return self._to_device(self._next_window(), device)

    def reset(self):
        self.current_position = self.initial_position

    def get_batch_with_meta(self, device, *, include_input_valid=False):
        x, y, input_valid = self._next_window(include_input_valid=True)
        doc_id, position_ids = compute_doc_meta(x, self.eot_token_id, input_valid)
        values = (x, y, doc_id, position_ids)
        if include_input_valid:
            values += (input_valid,)
        return self._to_device(values, device)

    def get_batch_with_byte_meta(self, device, *, include_input_valid=False):
        x, y, byte_lengths, input_valid = self._next_window(
            include_bytes=True, include_input_valid=True
        )
        doc_id, position_ids = compute_doc_meta(x, self.eot_token_id, input_valid)
        values = (x, y, doc_id, position_ids, byte_lengths)
        if include_input_valid:
            values += (input_valid,)
        return self._to_device(values, device)
