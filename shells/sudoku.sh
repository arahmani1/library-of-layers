# Paper figure app-fig:sudoku_results: Strategy adherence and sudoku solution accuracy under forced routing.
# Paper figure app-fig:sudoku_agree: Routing agreement on sudoku.
# Paper figure app-fig:sudoku_forced_path: routing paths on sudoku.

### TRAIN ###
# LoL trained on five sudoku strategies (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.block_size=2048 model.vocab_size=256 model.eot_token_id=3 'model.hidden_key_span=[83,91]' model.n_head=8 model.n_embd=512 model.n_experts=16 model.first_window_routing=learned_token_depth model.first_window_length=91 model.recurrent_depth=16 model.n_prelude_layers=4 model.n_coda_layers=4 model.weight_tie_router=true training.max_steps=9537 training.batch_size=16 training.grad_accum_steps=4 training.warmup_steps=1500 training.max_lr=0.0006 data.dataset=sudoku 'data.validation_sets=["val_S1","val_S2","val_S3","val_S4","val_S5"]' data.loss_sections=reasoning_answer_untagged 'data.tag_span=[83,91]' logging.eval_batches=50 logging.save_every=2000 logging.routing_diagnostics.trace_batches=4 wandb_run_name=sudoku_lol data.data_dir=data/sudoku/packed_min_length_406

### EVALUATE ###
# Forced soft routes calibrated over 16 validation batches; evaluate all selected puzzles.
.venv/bin/python -m evaluation.generate_strategy_routes --checkpoint "${CHECKPOINT_sudoku_lol:?Set CHECKPOINT_sudoku_lol to the trained sudoku_lol checkpoint}" --data-dir data/sudoku/packed_min_length_406 --source data/sudoku/raw --split test --n-puzzles 200 --conditions described,forced --derive-paths --derive-batches 16 --batch-size 32 --max-new-tokens 950 --device cuda:0 --shard 0/1 --paths paper_runs/sudoku_lol/generation/paths.json --out paper_runs/sudoku_lol/generation

### PLOT ###
# One rendering command writes strategy_conditions_sudoku.pdf,
# sudoku_route_agreement_all_windows.pdf, and sudoku_layer_paths_all_windows.pdf.
mkdir -p paper/figs/sudoku_maze_figs
.venv/bin/python -m plotting.strategy_routes --task sudoku --output-dir paper/figs/sudoku_maze_figs
