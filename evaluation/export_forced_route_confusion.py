"""Export and verify the existing Sudoku and maze forced-route heatmap data.

Run from the repository root with the environment used for plotting:
    python -m evaluation.export_forced_route_confusion \
        --sudoku-run evaluation/results/<sudoku-run> \
        --maze-run evaluation/results/<maze-run>

Reads saved generations only; never loads a checkpoint or generates samples.
Run directories are explicit because several plotted experiments coexist.
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from evaluation.plot_strategy_routes import NAMES, confusion, load


FIELDS = (
    "dataset", "source_run", "checkpoint", "forced_strategy",
    "generated_strategy", "count", "forced_total", "fraction",
)
CATEGORIES = [*NAMES, "mixed/invalid"]


def rows_for_run(dataset, run):
    run = Path(run)
    if dataset not in ("sudoku", "maze"):
        raise ValueError(f"unsupported dataset: {dataset}")
    # Match load(): a merged root file supersedes, rather than supplements, shards.
    root = run / "generations.jsonl"
    sources = [root] if root.exists() else sorted(run.glob("shard_*/generations.jsonl"))
    if not sources:
        raise FileNotFoundError(f"no saved generation records in {run}; generation is not run")
    checkpoints = set()
    for source in sources:
        summary = json.loads(source.with_name("summary.json").read_text())
        checkpoint = summary.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"missing generation checkpoint in {source.parent}")
        if summary.get("dataset") != dataset:
            raise ValueError(f"dataset mismatch in {source.parent}")
        checkpoints.add(checkpoint)
    if len(checkpoints) != 1:
        raise ValueError(f"inconsistent generation checkpoints in {run}")
    checkpoint = checkpoints.pop()

    forced = load(run).get("forced", [])
    id_field = "maze_id" if dataset == "maze" else "puzzle_id"
    other_id = "puzzle_id" if dataset == "maze" else "maze_id"
    counts = collections.Counter()
    totals = collections.Counter()
    for record in forced:
        if id_field not in record or other_id in record:
            raise ValueError(f"record dataset mismatch in {run}")
        strategy = record["forced"]
        if strategy not in NAMES:
            raise ValueError(f"invalid forced strategy {strategy!r} in {run}")
        generated = record.get("strategy")
        if generated not in NAMES:
            generated = "mixed/invalid"
        counts[strategy, generated] += 1
        totals[strategy] += 1
    missing = [s for s in NAMES if not totals[s]]
    if missing:
        raise ValueError(f"missing forced records for {missing} in {run}")

    rows = [
        dict(zip(FIELDS, (
            dataset, run.as_posix(), checkpoint, strategy, generated,
            counts[strategy, generated], totals[strategy],
            counts[strategy, generated] / totals[strategy],
        )))
        for strategy in NAMES for generated in CATEGORIES
    ]
    count_matrix = np.array([row["count"] for row in rows]).reshape(5, 6)
    fractions = np.array([row["fraction"] for row in rows]).reshape(5, 6)
    np.testing.assert_array_equal(count_matrix.sum(axis=1), [totals[s] for s in NAMES])
    np.testing.assert_allclose(fractions.sum(axis=1), 1, rtol=0, atol=1e-15)
    # The request also treats absent classification keys as mixed/invalid.
    # confusion() expects the key, so supply None only in temporary copies.
    reference = confusion([{**r, "strategy": r.get("strategy")} for r in forced], "forced")
    np.testing.assert_allclose(fractions, reference, rtol=0, atol=1e-15)
    return rows


def export_csv(sudoku_run, maze_run, output):
    rows = rows_for_run("sudoku", sudoku_run) + rows_for_run("maze", maze_run)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Validate the serialized handoff, including exact count and fraction text.
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames) != FIELDS:
            raise ValueError("CSV column order changed")
        saved = list(reader)
    if len(saved) != 60 or saved != [{key: str(value) for key, value in row.items()} for row in rows]:
        raise ValueError("CSV did not preserve all 60 verified cells")
    for dataset in ("sudoku", "maze"):
        selected = [row for row in saved if row["dataset"] == dataset]
        print(f"{dataset}: {selected[0]['source_run']}")
        print(f"  checkpoint: {selected[0]['checkpoint']}")
        print("  forced totals: " + ", ".join(
            f"{row['forced_strategy']}={row['forced_total']}"
            for row in selected if row["generated_strategy"] == "S1"
        ))
        print("  verified 30 cells, integer count totals, fraction sums, and confusion() parity")
    print(f"Wrote {output} (60 rows); no generation performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sudoku-run", type=Path, required=True)
    parser.add_argument("--maze-run", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("plotting/assets/forced_route_strategy_confusion.csv"))
    args = parser.parse_args()
    export_csv(args.sudoku_run, args.maze_run, args.output)


if __name__ == "__main__":
    main()
