"""Generate strategy traces from questions only and read off the strategy.

Works for both five-strategy corpora; the task is chosen by the checkpoint's
``data.dataset`` (sudoku or maze). Conditions:
    described  "[ <label of k> ]" in the prompt, router free.
    forced     neutral prompt, every window routed along strategy k's path.

A path is one expert per (window index inside the document, recurrent
depth); ``--derive-paths`` selects the highest mean router probability on
the per-strategy validation splits and records that mean as its residual
multiplier. Forced generation always applies both tables. Generation is greedy, token
by token, so a window's route is fixed before its first token.

Scoring. Sudoku: ``sudoku_data.format.classify`` (strategy from step
syntax), ``sudoku_data.verify.replay`` (legal steps reaching the true grid,
``valid``), ``faithful`` (every step is the deduction it claims) and the
answer digits. Maze: ``maze_data.identify.identify`` against the five
reference traces of the maze (``strategy`` is the one the trace equals,
``valid`` whether it equals any, ``faithful`` the fraction of the expected
strategy's trace the generated trace shares as a prefix) and the answer
path.

Usage:
    python -m evaluation.generate_strategy_routes \
        --checkpoint checkpoints/<run>/latest_<id>.pt \
        --paths checkpoints/<run>/paths.json --derive-paths \
        --n-puzzles 200 --out evaluation/results/<run>_generation
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataloader_multiproc import DataLoaderLit_MultiProc  # noqa: E402
from data.maze_data.format import (  # noqa: E402
    LABELS as MAZE_LABELS, PROMPT_LENGTH as MAZE_PROMPT_LENGTH, parse_sequence,
)
from data.maze_data.grid import CELL_INDEX, CELL_NAMES  # noqa: E402
from data.maze_data.identify import identify  # noqa: E402
from data.maze_data.tracers import STRATEGIES as MAZE_TRACERS  # noqa: E402
from data.maze_data.vocab import (  # noqa: E402
    SYMBOLS as MAZE_SYMBOLS, VOCAB as MAZE_VOCAB, decode as maze_decode,
)
from data.prepare_sudoku import (  # noqa: E402
    PROMPT_LENGTH, STRATEGIES, description_tokens,
)
from data.sudoku_data.format import classify  # noqa: E402
from data.strategy_corpus import iter_split  # noqa: E402
from data.sudoku_data.verify import (  # noqa: E402
    VerificationError, faithful, replay,
)
from data.sudoku_data.vocab import EOS, PAD, SYMBOLS, VOCAB  # noqa: E402
from evaluation.model_loading import build_evaluation_model  # noqa: E402

CONDITIONS = ("described", "forced")


@dataclass(frozen=True)
class Task:
    """What differs between the two corpora for generation and scoring.

    ``questions(source, split, n)`` yields ``(id, question, prompt_ids)``
    for distinct questions; ``label_body(strategy)`` gives the label
    tokens between the brackets; ``score(ids, question, expected)`` reads
    the strategy, validity and answer off one continuation.
    """

    name: str
    symbols: list
    vocab: dict
    id_field: str
    max_new_tokens: int
    label_body: Callable
    questions: Callable
    score: Callable

    @property
    def neutral_filler(self):
        return self.vocab["."]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dir", default=None,
        help="Packed corpus with manifest; default: the checkpoint's.",
    )
    parser.add_argument(
        "--source", default=None,
        help="Built corpus holding the questions; default: the packed "
             "manifest's dataset.source.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-puzzles", type=int, default=100,
                        help="Distinct questions (puzzles or mazes).")
    parser.add_argument(
        "--shard", default="0/1",
        help="K/N: keep every N-th question starting at K, so N processes on "
             "N GPUs can split one run; merge their generations.jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Default: 950 for sudoku, 200 for the maze.",
    )
    parser.add_argument(
        "--conditions", default=",".join(CONDITIONS),
        help=f"Comma-separated subset of {CONDITIONS}.",
    )
    parser.add_argument("--paths", required=True, help="Route table JSON.")
    parser.add_argument(
        "--derive-paths", action="store_true",
        help="(Re)build soft paths and residual weights from strategy validation.",
    )
    parser.add_argument("--derive-batches", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    conditions = tuple(name.strip() for name in args.conditions.split(",") if name.strip())
    if not conditions or set(conditions) - set(CONDITIONS):
        parser.error(f"--conditions must select from {CONDITIONS}")
    args.conditions = ",".join(conditions)
    return args


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    cfg = checkpoint["config"]
    model, _ = build_evaluation_model(checkpoint["model_state_dict"], cfg)
    model.to(device).eval()
    return model, cfg, int(checkpoint.get("step", -1))


def label_tokens(task, layout, strategy, neutral):
    """Tokens between the prompt and "R :" for the packed corpus's layout."""

    span = layout["tag_span"]
    width = int(span[1]) - int(span[0]) - 2
    body = (
        [task.neutral_filler] * width if neutral
        else task.label_body(strategy)
    )
    if len(body) != width:
        raise ValueError(
            f"label body has {len(body)} tokens, layout span holds {width}"
        )
    return [task.vocab["["]] + body + [task.vocab["]"]]


# --------------------------------------------------------------------------
# Sudoku
# --------------------------------------------------------------------------


def sudoku_label_body(strategy):
    return [int(t) for t in description_tokens(strategy)]


def load_puzzles(source, split, n_puzzles):
    """Yield (puzzle_id, grid, solution, prompt_tokens) for distinct puzzles."""

    seen = set()
    for meta, tokens in iter_split(source, split):
        if meta["puzzle_id"] in seen:
            continue
        seen.add(meta["puzzle_id"])
        tokens = tokens.astype(np.int64)
        if int(tokens[-1]) != EOS or len(tokens) < 83 + 82:
            raise ValueError(f"malformed sequence for {meta['puzzle_id']}")
        grid = [
            0 if int(t) == VOCAB["."] else int(SYMBOLS[int(t)])
            for t in tokens[2:PROMPT_LENGTH]
        ]
        solution = [int(SYMBOLS[int(t)]) for t in tokens[-82:-1]]
        yield meta["puzzle_id"], grid, solution, tokens[:PROMPT_LENGTH].tolist()
        if len(seen) >= n_puzzles:
            return


def sudoku_questions(source, split, n):
    for puzzle_id, grid, solution, prompt in load_puzzles(source, split, n):
        yield puzzle_id, {"grid": grid, "solution": solution}, prompt


def _live_tokens(generated, symbols):
    """Cleaned continuation and padding measurements before the first EOS."""

    ids = [int(t) for t in generated]
    ended = EOS in ids
    if ended:
        ids = ids[:ids.index(EOS)]
    live_ids = [t for t in ids if t != PAD]
    invalid = sorted({t for t in live_ids if not 0 <= t < len(symbols)})
    n_generated = len(ids)
    n_pad = n_generated - len(live_ids)
    padding = {
        "n_generated_tokens": n_generated,
        "n_pad_tokens": n_pad,
        "padding_fraction": n_pad / n_generated if n_generated else 0.0,
        "requires_cleanup": n_pad > 0,
    }
    return ended, live_ids, invalid, padding


def score(generated, grid, solution):
    """Strategy, legality and answer of one generated sudoku continuation."""

    ended, live_ids, invalid_token_ids, padding = _live_tokens(generated, SYMBOLS)
    record = {
        "ended": ended, "n_tokens": len(live_ids), "strategy": None,
        "steps": 0, "valid": False, "faithful": False, "answer_correct": False,
        "invalid_token_ids": invalid_token_ids,
        **padding,
    }
    if invalid_token_ids:
        return record
    tokens = [SYMBOLS[t] for t in live_ids]
    if "A" not in tokens:
        return record
    answer_at = tokens.index("A")
    body = tokens[:answer_at]
    steps = " ".join(body).split(" ; ") if body else []
    answer = tokens[answer_at + 2:answer_at + 83]
    record["steps"] = len(steps)
    strategy = classify(steps)
    record["strategy"] = None if strategy is None else f"S{strategy}"
    record["answer_correct"] = answer == [str(v) for v in solution]
    # valid: legal placements reaching the answer; faithful: every step is
    # the deduction its syntax claims (see sudoku_data.verify.faithful).
    for field, check in (("valid", replay), ("faithful", faithful)):
        if strategy is None:
            break
        try:
            check(grid, strategy, steps, solution)
            record[field] = True
        except (VerificationError, ValueError, KeyError, IndexError):
            break
    return record


# --------------------------------------------------------------------------
# Maze
# --------------------------------------------------------------------------


def maze_label_body(strategy):
    return [MAZE_VOCAB[cell] for cell in MAZE_LABELS[int(strategy[1:])].split(" ")]


def maze_questions(source, split, n):
    """Yield (maze_id, question, prompt) for distinct mazes.

    The question carries the maze, start, goal, the answer path and the five
    reference traces the generated trace is identified against.
    """

    seen = set()
    for meta, tokens in iter_split(source, split):
        if meta["maze_id"] in seen:
            continue
        seen.add(meta["maze_id"])
        tokens = tokens.astype(np.int64)
        maze, start, goal, _k, _trace, path = parse_sequence(maze_decode(tokens))
        question = {
            "maze": maze, "start": start, "goal": goal, "path": path,
            "references": {
                k: tracer(maze, start, goal) for k, tracer in MAZE_TRACERS.items()
            },
        }
        yield meta["maze_id"], question, tokens[:MAZE_PROMPT_LENGTH].tolist()
        if len(seen) >= n:
            return


def maze_score(generated, question, expected=None):
    """Strategy, identity and answer of one generated maze continuation.

    ``strategy`` is the strategy whose reference trace the generated trace
    equals (None when it equals none or is malformed), ``valid`` whether it
    equals any, ``faithful`` the fraction of ``expected``'s reference trace
    it shares as a prefix (the best strategy's fraction when no strategy is
    expected), ``answer_correct`` whether the path after "A :" is the maze's.
    """

    ended, live_ids, invalid_token_ids, padding = _live_tokens(generated, MAZE_SYMBOLS)
    record = {
        "ended": ended, "n_tokens": len(live_ids), "strategy": None,
        "n_trace": 0, "valid": False, "faithful": 0.0,
        "answer_correct": False, "invalid_token_ids": invalid_token_ids,
        "prefix": {},
        **padding,
    }
    if invalid_token_ids:
        return record
    tokens = [MAZE_SYMBOLS[t] for t in live_ids]
    if "A" not in tokens:
        return record
    answer_at = tokens.index("A")
    body = tokens[:answer_at]
    answer = tokens[answer_at + 2:]
    record["answer_correct"] = answer == [
        CELL_NAMES[i] for i in question["path"]
    ]
    if any(cell not in CELL_INDEX for cell in body):
        return record
    trace = [CELL_INDEX[cell] for cell in body]
    references = question["references"]
    prefixes, match = identify(
        trace, question["maze"], question["start"], question["goal"], references,
    )
    fractions = {
        f"S{k}": prefixes[k] / len(references[k]) for k in references
    }
    record["n_trace"] = len(trace)
    record["prefix"] = {f"S{k}": n for k, n in prefixes.items()}
    record["strategy"] = None if match is None else f"S{match}"
    record["valid"] = match is not None
    record["faithful"] = (
        fractions[expected] if expected in fractions else max(fractions.values())
    )
    return record


TASKS = {
    "sudoku": Task(
        "sudoku", SYMBOLS, VOCAB, "puzzle_id", 950, sudoku_label_body,
        sudoku_questions,
        lambda ids, question, expected: score(
            ids, question["grid"], question["solution"]
        ),
    ),
    "maze": Task(
        "maze", MAZE_SYMBOLS, MAZE_VOCAB, "maze_id", 200, maze_label_body,
        maze_questions, maze_score,
    ),
}


def task_of(cfg):
    """The checkpoint's task; checkpoints that predate ``data.dataset``
    are sudoku."""
    return TASKS[cfg.get("data", {}).get("dataset", "sudoku")]


# --------------------------------------------------------------------------
# Route tables
# --------------------------------------------------------------------------


@torch.no_grad()
def derive_paths(model, cfg, data_dir, batches, device):
    """Highest mean probability per strategy/window/depth and its multiplier.

    Uses the model's own traced routes on the per-strategy validation splits,
    so the table is what the router does when the label is present.
    """

    depth = int(cfg["model"]["recurrent_depth"])
    experts = int(cfg["model"]["n_experts"])
    if batches < 1:
        raise ValueError("derive batches must be positive")
    table = {}
    for strategy in STRATEGIES:
        loader = DataLoaderLit_MultiProc(
            data_dir, 16, int(cfg["model"]["block_size"]),
            split=f"val_{strategy}", process_rank=0, num_processes=1,
            eot_token_id=int(cfg["model"]["eot_token_id"]),
            model_vocab_size=int(cfg["model"]["vocab_size"]),
        )
        counts = collections.defaultdict(
            lambda: np.zeros((depth, experts), dtype=np.int64)
        )
        probability_sums = collections.defaultdict(
            lambda: np.zeros((depth, experts), dtype=np.float64)
        )
        for _ in range(batches):
            x, _, doc_ids, position_ids, input_valid = (
                loader.get_batch_with_meta(device, include_input_valid=True)
                if getattr(loader, "validity", None) is not None else
                (*loader.get_batch_with_meta(device), None)
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                traces = model.trace_experts(
                    x, doc_id=doc_ids, position_ids=position_ids,
                    input_valid=input_valid,
                    return_scores=True,
                )
            live = (input_valid if input_valid is not None else x != PAD).reshape(-1).float()
            sequences, length = doc_ids.shape
            global_doc = (
                torch.arange(sequences, device=device).unsqueeze(-1)
                * (length + 1) + doc_ids
            ).reshape(-1)
            for current_depth, (indices, window_ids, probabilities) in enumerate(traces):
                window_of_token = window_ids.reshape(-1)
                total = int(window_of_token.max()) + 1
                tokens = torch.zeros(total, device=device).index_add_(
                    0, window_of_token, live
                )
                order = torch.arange(window_of_token.numel(), device=device)
                first = torch.full(
                    (total,), window_of_token.numel(), device=device
                )
                first.scatter_reduce_(
                    0, window_of_token, order, reduce="amin"
                )
                first.clamp_(max=window_of_token.numel() - 1)
                documents = global_doc[first]
                first_of_doc = torch.ones_like(documents, dtype=torch.bool)
                first_of_doc[1:] = documents[1:] != documents[:-1]
                ordinal = model._win_in_doc_idx(
                    first_of_doc, total, device
                )
                live_window = tokens > 0
                ordinals = ordinal[live_window].cpu().numpy()
                chosen = indices[live_window].cpu().numpy()
                routed = probabilities[live_window].float().cpu().numpy()
                for window_ordinal in np.unique(ordinals):
                    rows_of = ordinals == window_ordinal
                    counts[int(window_ordinal)][current_depth] += np.bincount(
                        chosen[rows_of], minlength=experts
                    )
                    probability_sums[int(window_ordinal)][current_depth] += (
                        routed[rows_of].sum(0)
                    )
        rows = []
        agreement = []
        weights = []
        observations = []
        for ordinal in range(max(counts) + 1):
            cell = counts[ordinal]
            count = cell.sum(1)
            # Shifted depths can lack a trailing ordinal. Retain the native
            # empty cell: expert 0, zero agreement and zero residual weight;
            # observations distinguishes it from a measured zero share.
            means = probability_sums[ordinal] / np.maximum(count, 1)[:, None]
            selected = means.argmax(1)
            rows.append(selected.tolist())
            weights.append(means[np.arange(depth), selected].round(4).tolist())
            agreement.append(
                (cell[np.arange(depth), selected] / np.maximum(count, 1)).round(3).tolist()
            )
            observations.append(count.tolist())
        table[strategy] = {"path": rows, "weights": weights, "agreement": agreement,
                           "observations": observations}
        print(
            f"{strategy}: {len(rows)} window rows; window-1 agreement "
            f"{np.mean(agreement[1]) if len(agreement) > 1 else float('nan'):.2f}",
            flush=True,
        )
    return table


# --------------------------------------------------------------------------
# Generation and scoring
# --------------------------------------------------------------------------


@torch.no_grad()
def generate(model, prompts, path, max_new_tokens, task=TASKS["sudoku"], *, weights=None):
    """Greedy continuation of equal-length ``prompts`` ([B, L] long).

    ``path`` is a ``[windows, depth]`` route table or ``None`` for the
    router's own choices. Every step re-runs the full sequence: the model has
    no KV cache, and routes must be recomputed from completed windows. Only
    the task's real symbols can be chosen.
    """

    idx = prompts.clone()
    if (path is None) != (weights is None):
        raise ValueError("forced generation requires both a soft path and its recorded weights")
    prompt_length = idx.size(1)
    if prompt_length > model.config.block_size:
        raise ValueError(
            f"prompt length {prompt_length} exceeds model.block_size "
            f"{model.config.block_size}"
        )
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
    for _ in range(min(max_new_tokens, model.config.block_size - prompt_length)):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if path is None:
                logits = model(idx)[0]
            else:
                logits = model.inference_fixed_path(
                    idx[:, :prompt_length], idx, fixed_path=path,
                    fixed_weights=weights,
                )[0]
        next_token = logits[:, -1, :len(task.symbols)].float().argmax(-1)
        next_token = torch.where(
            finished, torch.full_like(next_token, PAD), next_token
        )
        idx = torch.cat([idx, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == EOS
        if bool(finished.all()):
            break
    return idx[:, prompt_length:]


def build_jobs(task, questions, layout, conditions, paths):
    """One job per (question, condition, strategy) with its prompt and route."""

    if set(conditions) - set(CONDITIONS):
        raise ValueError(f"conditions must select from {CONDITIONS}")
    tail = [task.vocab["R"], task.vocab[":"]]
    for question_id, question, prompt in questions:
        for condition in conditions:
            for strategy in STRATEGIES:
                described = strategy
                forced = None
                if condition == "forced":
                    described = None
                    forced = strategy
                label = label_tokens(
                    task, layout, described or strategy, described is None
                )
                yield {
                    "id": question_id,
                    "condition": condition,
                    "described": described,
                    "forced": forced,
                    "expected": forced or described,
                    "question": question,
                    "prompt": prompt + label + tail,
                    "path": None if forced is None else paths[forced]["path"],
                    "weights": None if forced is None else paths[forced]["weights"],
                }


def padding_metrics(records):
    """Equal-weight per-generation padding means and measurement coverage.

    Older cleaned records cannot reveal pre-EOS padding. Exclude missing
    measurements and report unavailable means as None, never as zero.
    """

    records = list(records)
    measured = [
        record for record in records
        if record.get("padding_fraction") is not None
        and record.get("requires_cleanup") is not None
    ]
    n = len(measured)
    return {
        "padding_fraction": (
            sum(record["padding_fraction"] for record in measured) / n
            if n else None
        ),
        "cleanup_fraction": (
            sum(record["requires_cleanup"] for record in measured) / n
            if n else None
        ),
        "padding_n": n,
        "padding_coverage": n / len(records) if records else 0.0,
    }


def summarize(records):
    """Per (condition, expected strategy) rates, plus a pooled row each."""

    groups = collections.defaultdict(list)
    for record in records:
        groups[(record["condition"], record["expected"] or "none")].append(
            record
        )
        groups[(record["condition"], "all")].append(record)
    summary = {}
    for (condition, expected), rows in sorted(groups.items()):
        n = len(rows)
        summary[f"{condition}/{expected}"] = {
            "n": n,
            "strategy_match": sum(
                r["expected"] is not None and r["strategy"] == r["expected"]
                for r in rows
            ) / n,
            "described_match": sum(
                r["described"] is not None and r["strategy"] == r["described"]
                for r in rows
            ) / n,
            "classified": sum(r["strategy"] is not None for r in rows) / n,
            "valid_trace": sum(r["valid"] for r in rows) / n,
            "faithful_trace": sum(r.get("faithful", False) for r in rows) / n,
            "answer_correct": sum(r["answer_correct"] for r in rows) / n,
            "ended": sum(r["ended"] for r in rows) / n,
            **padding_metrics(rows),
        }
    return summary


def main(argv=None):
    args = parse_args(argv)
    model, cfg, step = load_model(args.checkpoint, args.device)
    task = task_of(cfg)
    data_dir = args.data_dir or cfg["data"]["data_dir"]
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    layout = manifest.get("layout", {})
    source = args.source or manifest["dataset"]["source"]
    max_new_tokens = args.max_new_tokens or task.max_new_tokens
    conditions = tuple(
        name.strip() for name in args.conditions.split(",") if name.strip()
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paths_file = Path(args.paths)
    if args.derive_paths or not paths_file.exists():
        started = time.monotonic()
        paths = derive_paths(
            model, cfg, data_dir, args.derive_batches, args.device
        )
        paths_file.parent.mkdir(parents=True, exist_ok=True)
        paths_file.write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "step": step,
            "data_dir": str(data_dir), "batches": args.derive_batches,
            "path_policy": "soft",
            "paths": paths,
        }, indent=1))
        print(
            f"derived route table in {time.monotonic() - started:.0f}s -> "
            f"{paths_file}"
        )
    paths_text = paths_file.read_text()
    paths = json.loads(paths_text)["paths"]
    if json.loads(paths_text).get("path_policy") != "soft":
        raise ValueError("route tables must be derived with the soft policy; use --derive-paths")
    if paths_file.resolve() != (out / "paths.json").resolve():
        (out / "paths.json").write_text(paths_text)

    shard, shards = (int(part) for part in args.shard.split("/"))
    questions = [
        question for index, question
        in enumerate(task.questions(source, args.split, args.n_puzzles))
        if index % shards == shard
    ]
    jobs = list(build_jobs(task, questions, layout, conditions, paths))
    print(
        f"{len(questions)} {task.name} questions x {conditions} -> "
        f"{len(jobs)} generations",
        flush=True,
    )
    # Batch jobs that share a route table so one forward serves the batch.
    jobs.sort(key=lambda job: (job["forced"] or "", job["condition"]))
    records = []
    started = time.monotonic()
    with open(out / "generations.jsonl", "w") as handle:
        position = 0
        while position < len(jobs):
            batch = [jobs[position]]
            while (
                position + len(batch) < len(jobs)
                and len(batch) < args.batch_size
                and jobs[position + len(batch)]["forced"] == batch[0]["forced"]
            ):
                batch.append(jobs[position + len(batch)])
            position += len(batch)
            prompts = torch.tensor(
                [job["prompt"] for job in batch], device=args.device
            )
            generated = generate(
                model, prompts, batch[0]["path"], max_new_tokens, task,
                weights=batch[0]["weights"],
            ).cpu()
            for job, continuation in zip(batch, generated):
                record = {task.id_field: job["id"]}
                record.update({
                    key: job[key] for key in (
                        "condition", "described", "forced", "expected",
                    )
                })
                record.update(
                    task.score(continuation, job["question"], job["expected"])
                )
                # Keep generated PADs for exact future rescoring; discard
                # batch padding and any other tokens after the first EOS.
                raw_length = record["n_generated_tokens"] + int(record["ended"])
                record["generated_ids"] = continuation[:raw_length].tolist()
                record["text"] = " ".join(
                    task.symbols[int(t)]
                    if 0 <= int(t) < len(task.symbols)
                    else f"<invalid:{int(t)}>"
                    for t in continuation if int(t) != PAD
                )
                records.append(record)
                handle.write(json.dumps(record) + "\n")
            print(
                f"{len(records)}/{len(jobs)} done, "
                f"{time.monotonic() - started:.0f}s", flush=True,
            )

    summary = summarize(records)
    (out / "summary.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "step": step,
        "dataset": task.name, "conditions": conditions,
        "source": str(source), "split": args.split, "data_dir": str(data_dir),
        "max_new_tokens": max_new_tokens, "batch_size": args.batch_size,
        "shard": args.shard, "n_puzzles_requested": args.n_puzzles,
        "padding_metric_version": 1,
        "path_policy": "soft",
        "n_questions": len(questions), "rates": summary,
    }, indent=1))
    header = (
        f"{'condition/expected':22s} {'n':>5s} {'strategy':>9s} "
        f"{'described':>9s} {'valid':>7s} {'faithful':>9s} {'answer':>7s} "
        f"{'ended':>6s}"
    )
    print(header)
    for name, rates in summary.items():
        print(
            f"{name:22s} {rates['n']:5d} {rates['strategy_match']:9.3f} "
            f"{rates['described_match']:9.3f} {rates['valid_trace']:7.3f} "
            f"{rates['faithful_trace']:9.3f} {rates['answer_correct']:7.3f} "
            f"{rates['ended']:6.3f}"
        )


if __name__ == "__main__":
    main()
