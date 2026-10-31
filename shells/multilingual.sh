# Paper figure fig:router: Distinct paths for languages and for sudoku and maze strategies.
# Paper figure app-fig:multilang: Evolution of multilingual routing during training.

### TRAIN ###
# LoL trained on an equal mixture of 10 languages (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.vocab_size=100352 model.eot_token_id=100257 model.n_experts=33 model.first_window_routing=learned_token_depth model.recurrent_depth=22 model.n_prelude_layers=1 model.n_coda_layers=1 model.weight_tie_router=true training.max_steps=9390 training.batch_size=26 training.grad_accum_steps=5 training.max_lr=0.0006 data.dataset=fineweb2 'data.validation_sets=["rus_Cyrl","cmn_Hani","spa_Latn","arb_Arab","hin_Deva","deu_Latn","jpn_Jpan","kor_Hang","tur_Latn","fra_Latn"]' wandb_run_name=multilingual_lol data.data_dir=data/fineweb2/10

### EVALUATE ###
# LoL trained on an equal mixture of 10 languages (seed 42); PCA projection of multilingual execution paths.
.venv/bin/python -m plotting.make_pca --model lol_dynamic_window --checkpoint "${CHECKPOINT_multilingual_lol:?Set CHECKPOINT_multilingual_lol to the trained multilingual_lol checkpoint}" --data-dir data/fineweb2/10 --trace-batches 3 --max-windows 4000 --output paper_runs/multilingual_lol/pca/routes.npz

### PLOT ###
mkdir -p paper
.venv/bin/python -m plotting.multilingual_progression --output paper/multilingual_progression.pdf

# The overview requires user-provided local PCA and confusion measurements.
# Use --multilingual-npz and --confusion-csv to select your own exported measurements.
mkdir -p paper/figs
.venv/bin/python -m plotting.routing_overview --output paper/figs/routing_overview.pdf
