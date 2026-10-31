# Paper figure app-fig:maze_results: Strategy and maze solution generation under forced routing.
# Paper figure app-fig:maze_agree: Routing agreement on mazes.
# Paper figure app-fig:maze_forced_path: Mode routing paths on mazes.

### TRAIN ###
# LoL trained on five maze strategies (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.block_size=2048 model.vocab_size=256 model.window=64 model.window_stride=32 model.eot_token_id=3 'model.hidden_key_span=[196,204]' model.n_head=8 model.n_embd=512 model.n_experts=16 model.first_window_routing=learned_token_depth model.first_window_length=204 model.recurrent_depth=16 model.n_prelude_layers=4 model.n_coda_layers=4 model.weight_tie_router=true training.max_steps=9800 training.batch_size=16 training.grad_accum_steps=4 training.warmup_steps=1500 training.max_lr=0.0006 data.dataset=maze 'data.validation_sets=["val_S1","val_S2","val_S3","val_S4","val_S5"]' data.loss_sections=reasoning_answer_untagged 'data.tag_span=[196,204]' logging.eval_batches=50 logging.save_every=2000 logging.routing_diagnostics.trace_batches=4 wandb_run_name=maze_lol data.data_dir=data/maze/packed

### EVALUATE ###
# LoL trained on five maze strategies (seed 42); described prompts and forced mode-path generation.
.venv/bin/python -m evaluation.generate_strategy_routes --checkpoint "${CHECKPOINT_maze_lol:?Set CHECKPOINT_maze_lol to the trained maze_lol checkpoint}" --data-dir data/maze/packed --source data/maze/raw --split test_swap --n-puzzles 20000 --conditions described,forced --derive-paths --derive-batches 8 --batch-size 32 --max-new-tokens 200 --device cuda:0 --shard 0/1 --paths paper_runs/maze_lol/generation/paths.json --out paper_runs/maze_lol/generation

### PLOT ###
# One rendering command writes strategy_conditions_maze.pdf,
# maze_route_agreement_all_windows.pdf, and maze_layer_paths_all_windows.pdf.
mkdir -p paper/figs/sudoku_maze_figs
.venv/bin/python -m plotting.strategy_routes --task maze --output-dir paper/figs/sudoku_maze_figs
