"""Build a five-strategy corpus: ``python -m data.build --sudoku|--maze``.

    python -m data.build --maze   --out datasets/maze_strategies/v1 \\
                         --n-mazes 7200000 --workers 32 --seed 0
    python -m data.build --sudoku --out datasets/sudoku_strategies/v3 \\
                         --n-puzzles 4000000 --workers 32 --seed 0

The dataset flag selects the policy (``data/<name>_data/corpus.py``) and its
own options (``--maze -h`` lists them); generation workers, verification,
hash splits, deduplication and sharding are ``data.strategy_corpus``.
"""

import argparse
import json

from data.strategy_corpus import build, dataset_module


def main(argv=None):
    choice = argparse.ArgumentParser(add_help=False)
    group = choice.add_mutually_exclusive_group(required=True)
    group.add_argument("--sudoku", dest="dataset", action="store_const",
                       const="sudoku", help="Build the sudoku corpus.")
    group.add_argument("--maze", dest="dataset", action="store_const",
                       const="maze", help="Build the maze corpus.")
    corpus = dataset_module(choice.parse_known_args(argv)[0].dataset)

    parser = argparse.ArgumentParser(description=__doc__, parents=[choice])
    parser.add_argument("--out", required=True)
    parser.add_argument(f"--n-{corpus.UNIT}", dest="n", type=int, required=True,
                        help=f"{corpus.UNIT.capitalize()} to keep.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    corpus.add_arguments(parser)
    args = parser.parse_args(argv)

    config = build(
        corpus.NAME, args.out, args.n, args.workers, args.seed, corpus.params(args)
    )
    print(json.dumps(config["run"], indent=2))


if __name__ == "__main__":
    main()
