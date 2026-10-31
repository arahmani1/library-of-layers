#!/usr/bin/env python3
"""Save a routed checkpoint's validation routing PCA points as an NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import router  # noqa: E402
from data.dataloader_multiproc import DataLoaderLit_MultiProc  # noqa: E402
from evaluation.model_loading import build_evaluation_model  # noqa: E402
from utils.routing_logger import RoutingAccumulator, active_languages  # noqa: E402
from utils.routing_viz import cluster_space_coordinates  # noqa: E402
from utils.validation_utils import (  # noqa: E402
    reset_validation_loader,
    validation_batch_count,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("lol_dynamic_window",),
        help="Routed architecture; must match the checkpoint's model.arch",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path,
        help="NPZ file for routing_overview.py --multilingual-npz",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        help="Override checkpoint data.data_dir; relative paths use the repo root",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Override checkpoint training.batch_size",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--trace-batches", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=4000)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if args.trace_batches <= 0 or args.max_windows <= 0:
        parser.error("--trace-batches and --max-windows must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or not {
        "config", "model_state_dict",
    } <= checkpoint.keys():
        raise ValueError("Checkpoint must contain config and model_state_dict")
    cfg = checkpoint["config"]
    step = int(checkpoint.get("step", -1))
    arch = str(cfg["model"]["arch"])
    if arch != args.model:
        raise ValueError(
            f"--model {args.model!r} does not match checkpoint model.arch {arch!r}"
        )
    model, _ = build_evaluation_model(checkpoint["model_state_dict"], cfg)
    model.to(device).eval()
    del checkpoint

    model_cfg = cfg["model"]
    languages = active_languages(cfg["data"].get("validation_sets", ["val"]))
    batch_size = (
        int(args.batch_size) if args.batch_size is not None
        else int(cfg["training"]["batch_size"])
    )
    data_dir = args.data_dir or Path(cfg["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    accumulator = RoutingAccumulator(
        languages,
        int(model_cfg["recurrent_depth"]),
        int(model_cfg["n_experts"]),
        max_windows=args.max_windows,
    )

    with torch.no_grad():
        for language_index, language in enumerate(languages):
            loader = DataLoaderLit_MultiProc(
                data_dir,
                batch_size,
                int(model_cfg["block_size"]),
                split=language,
                eot_token_id=int(model_cfg.get("eot_token_id", 0)),
                model_vocab_size=int(model_cfg["vocab_size"]),
            )
            reset_validation_loader(loader)
            batch_count = validation_batch_count(loader, args.trace_batches)
            for _ in range(batch_count):
                inputs, _, doc_ids, position_ids, input_valid = (
                    loader.get_batch_with_meta(device, include_input_valid=True)
                )
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    outputs = router.validation_forward(
                        model, inputs, doc_ids, position_ids, None,
                        return_router_trace=True,
                        input_valid=input_valid,
                    )
                router.accumulate_captured_routing_trace(
                    accumulator, language_index, outputs[3],
                    doc_ids, input_valid=input_valid,
                    validation_name=language,
                )
            if not accumulator.counts_window_sample[language_index].any():
                raise RuntimeError(f"No routing rows captured for {language}")
            print(f"Captured {language}: {batch_count} validation batches")

    accumulator.all_reduce()
    routes, lang_ids, n_tokens = accumulator.reservoir
    ordinals = accumulator.reservoir_ordinals
    selected = ordinals > 0
    if int(selected.sum()) < 2:
        raise RuntimeError("Need at least two routes after the first document window")
    routes = routes[selected].numpy()
    lang_ids = lang_ids[selected].numpy()
    n_tokens = n_tokens[selected].numpy()
    ordinals = ordinals[selected].numpy()
    coordinates = cluster_space_coordinates(routes, accumulator.n_blocks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        coordinates=coordinates,
        routes=routes,
        lang_ids=lang_ids,
        n_tokens=n_tokens,
        ordinals=ordinals,
        languages=np.asarray(languages),
        n_blocks=accumulator.n_blocks,
        n_depths=accumulator.n_depths,
        checkpoint_step=step,
        model=arch,
    )

    print(f"Wrote {args.output.name}")
    print(f"Step {step}, {len(routes)} routes")


if __name__ == "__main__":
    main()
