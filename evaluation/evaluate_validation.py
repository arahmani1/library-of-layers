#!/usr/bin/env python3
"""Evaluate one training checkpoint's validation NLL and perplexity."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

# Allow running the script from any CWD (e.g., repo root or evaluation/).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataloader_multiproc import DataLoaderLit_MultiProc
from evaluation.model_loading import build_evaluation_model
from utils.validation_utils import (
    reset_validation_loader,
    dataset_section_masks,
    token_nll_sum,
    unpack_validation_outputs,
    validation_batch_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute validation NLL and perplexity for one checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint file.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data.data_dir from checkpoint/config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size from checkpoint/config.",
    )
    parser.add_argument(
        "--validation-set",
        default=None,
        help="Validation split (default: first data.validation_sets entry).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (cpu or cuda).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="If > 0, evaluate only this many validation batches (default: full val set).",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Disable bfloat16 autocast on CUDA.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    return parser.parse_args()


def resolve_input_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    if not p.is_absolute():
        alt = REPO_ROOT / p
        if alt.exists():
            return alt
    return p


def resolve_output_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def evaluate_checkpoint(
    ckpt_path: str,
    data_dir_override: str,
    batch_size_override: int,
    device: str,
    max_val_batches: int,
    use_autocast: bool,
    validation_set_override: str | None = None,
) -> Dict:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    cfg = checkpoint["config"]

    model, arch = build_evaluation_model(state_dict, cfg)
    model.to(device)
    model.eval()

    block_size = cfg["model"]["block_size"]
    data_cfg = cfg.get("data") or {}
    data_dir = data_dir_override or data_cfg["data_dir"]
    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else cfg["training"]["batch_size"]
    )

    validation_sets = cfg["data"].get("validation_sets", ["val"])
    if not validation_sets:
        raise ValueError("data.validation_sets must not be empty")
    validation_set = validation_set_override or str(validation_sets[0])
    val_loader = DataLoaderLit_MultiProc(
        data_dir,
        batch_size,
        block_size,
        split=validation_set,
        eot_token_id=int(cfg["model"].get("eot_token_id", 0)),
        model_vocab_size=int(cfg["model"]["vocab_size"]),
    )
    reset_validation_loader(val_loader)
    n_batches = validation_batch_count(val_loader, max_val_batches)

    autocast_enabled = use_autocast and str(device).startswith("cuda")

    total_nll = 0.0
    total_tokens = 0

    t0 = time.time()
    ckpt_name = Path(ckpt_path).name
    model_bar = tqdm(
        range(n_batches),
        desc=f"model [{ckpt_name}]",
        total=n_batches,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )
    with torch.no_grad():
        for _ in model_bar:
            x, y, doc_id, position_ids, input_valid = (
                val_loader.get_batch_with_meta(device, include_input_valid=True)
                if getattr(val_loader, "validity", None) is not None else
                (*val_loader.get_batch_with_meta(device), None)
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                outputs = model(
                    x, doc_id=doc_id, position_ids=position_ids,
                    **({"input_valid": input_valid} if input_valid is not None else {}),
                )
                logits, loss_mask = unpack_validation_outputs(outputs, y, arch)
                # Score the same targets training validation scores: a
                # sudoku or maze loss arm excludes the prompt, the label and
                # the padding.
                section_mask, _ = dataset_section_masks(
                    data_cfg.get("dataset"), data_cfg.get("loss_sections"),
                    data_cfg.get("tag_span"), x, y, doc_id, position_ids,
                )
                if section_mask is not None:
                    loss_mask = (
                        section_mask if loss_mask is None
                        else loss_mask & section_mask
                    )
                loss_sum, token_count = token_nll_sum(logits, y, loss_mask)
            total_nll += loss_sum.item()
            total_tokens += int(token_count.item())
            if total_tokens > 0:
                model_bar.set_postfix_str(f"tok={total_tokens:,}")

    model_bar.close()

    if total_tokens <= 0:
        raise ValueError(f"No valid validation tokens for architecture {arch!r}")
    elapsed = max(time.time() - t0, 1e-9)
    nll = total_nll / total_tokens
    ppl = math.exp(nll) if nll < 80 else float("inf")


    return {
        "checkpoint": ckpt_path,
        "arch": arch,
        "validation_set": validation_set,
        "step": int(checkpoint.get("step", -1)),
        "batch_size": int(batch_size),
        "block_size": int(block_size),
        "val_batches": int(n_batches),
        "val_tokens": int(total_tokens),
        "nll": float(nll),
        "perplexity": float(ppl),
        "tokens_per_sec": float(total_tokens / elapsed),
    }


def print_table(rows: List[Dict]) -> None:
    headers = [
        "checkpoint",
        "arch",
        "validation_set",
        "step",
        "nll",
        "perplexity",
        "val_batches",
        "val_tokens",
    ]

    printable_rows = []
    for r in rows:
        printable_rows.append(
            {
                "checkpoint": r["checkpoint"],
                "arch": r["arch"],
                "validation_set": r["validation_set"],
                "step": str(r["step"]),
                "nll": f"{r['nll']:.6f}",
                "perplexity": f"{r['perplexity']:.6f}",
                "val_batches": str(r["val_batches"]),
                "val_tokens": str(r["val_tokens"]),
            }
        )

    col_widths = {h: len(h) for h in headers}
    for row in printable_rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(row[h]))

    def fmt_row(row_dict: Dict[str, str]) -> str:
        return " | ".join(row_dict[h].ljust(col_widths[h]) for h in headers)

    print("\nValidation Metrics")
    print(fmt_row({h: h for h in headers}))
    print("-+-".join("-" * col_widths[h] for h in headers))
    for row in printable_rows:
        print(fmt_row(row))


def write_json(path: str, rows: List[Dict]) -> None:
    out = resolve_output_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    checkpoint = resolve_input_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    row = evaluate_checkpoint(
        ckpt_path=str(checkpoint),
        data_dir_override=args.data_dir,
        batch_size_override=args.batch_size,
        device=args.device,
        max_val_batches=args.max_val_batches,
        use_autocast=not args.no_autocast,
        validation_set_override=args.validation_set,
    )
    print_table([row])
    if args.json:
        write_json(args.json, [row])
        print(f"JSON saved to {args.json}")


if __name__ == "__main__":
    main()
