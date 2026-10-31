# Library of Layers (LoL)

Code to reproduce **Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime**. A joint project by Abdellah Rahmani, Sean McLeish, Ang Li, Sanae Lotfi, Pascal Frossard, and Micah Goldblum.

# Citing Our Work
```
@article{rahmani2026library,
    title={Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime}, 
    author={Abdellah Rahmani and Sean McLeish and Ang Li and Sanae Lotfi and Pascal Frossard and Micah Goldblum},
    journal={arXiv preprint arXiv:2611.},
    year={2026}
}
```

# Reproduction Instructions
Use Python 3.12 and the dependencies in [pyproject.toml](pyproject.toml).

## Data preparation

Training requires prepared corpora; select their location with `data.data_dir`.
Use the preparation modules' `--help` for source, quota, and output options:

| Corpus | Preparation entry point |
| --- | --- |
| FineWeb-Edu | `python -m data.prepare_fineweb_edu_30b --help` |
| Math | `python -m data.prepare_math --help` |
| Multilingual FineWeb2 | `python -m data.prepare_fineweb2_multilingual --help` |
| Safety | `python -m data.prepare_fineweb_safety --help` |
| Sudoku | Generate with `python -m data.build --sudoku --help`, then pack with `python -m data.prepare_sudoku --help`. |
| Maze | Generate with `python -m data.build --maze --help`, then pack with `python -m data.prepare_maze --help`. |

For Sudoku and maze, point training/evaluation at the packed output and use the
corresponding shell recipe's matching label masks and first-window settings.

## Training and Evaluation
The commands below reproduce the paper's figures and results. Each training workflow owns its plotting commands and has
`TRAIN`, `EVALUATE`, and `PLOT` sections. Run the commands you need from the
repository root; comments identify shared runs used by several figures.

| Shell file | What it reproduces |
| --- | --- |
| [math.sh](shells/math.sh) | All math results: five-model training/evaluation, main and extended loss/FLOP figures, expert utilization, and benchmark tables. |
| [lol_vs_huginn.sh](shells/lol_vs_huginn.sh) | LoL versus Huginn: validation loss, training loss over FLOPs, and expert utilization across seeds. |
| [lol_vs_moe.sh](shells/lol_vs_moe.sh) | LoL versus mixture-of-experts models, including expert utilization across seeds. |
| [safety_probes.sh](shells/safety_probes.sh) | Safety prediction from router probabilities. |
| [lol_backprop_depth.sh](shells/lol_backprop_depth.sh) | Growing depth and backpropagation ablation. |
| [lol_load_balancing.sh](shells/lol_load_balancing.sh) | Loss-free versus auxiliary-loss load balancing. |
| [lol_router_sharing.sh](shells/lol_router_sharing.sh) | Tied versus untied router weights. |
| [vanilla_attention.sh](shells/vanilla_attention.sh) | Full, sliding-window, and shifted-window attention in dense models. |
| [multilingual.sh](shells/multilingual.sh) | Multilingual routing progression and the combined language/Sudoku/maze routing overview. |
| [sudoku.sh](shells/sudoku.sh) | Sudoku strategy adherence, solution accuracy, routing agreement, and routing paths. |
| [maze.sh](shells/maze.sh) | Maze strategy and solution generation, routing agreement, and routing paths. |

Before evaluation, set the `CHECKPOINT_*` variables shown in the script to the corresponding trained checkpoints, for example:

```bash
export CHECKPOINT_sudoku_lol=/path/to/checkpoint.pt
```

## Plotting
Copy [the manifest template](plotting/assets/datasets.example.json) to `datasets.json` in your data directory, replace null placeholders, and provide the files required by your selected plot. The default directory is `plotting/assets`; select another with `--data-dir`. Relative input paths must stay inside that directory.
