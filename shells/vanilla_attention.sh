# Paper figure app-fig:arch-abl-vanilla-attn: Full attention achieves the lowest loss among dense fixed-depth attention patterns.

### TRAIN ###
# Vanilla with full attention (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=vanilla_neo shape=360m seed=42 model.n_layer=24 training.max_steps=9393 training.batch_size=26 training.grad_accum_steps=5 training.load_balancing.strategy=aux_loss training.max_lr=0.0004 wandb_run_name=vanilla_full data.data_dir=data/fineweb
# Vanilla with sliding window 128 (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=vanilla_neo shape=360m seed=42 model.window=128 model.attention_strategy=sliding_window model.n_layer=24 training.max_steps=9393 training.batch_size=26 training.grad_accum_steps=5 training.load_balancing.strategy=aux_loss training.max_lr=0.0004 wandb_run_name=vanilla_sliding data.data_dir=data/fineweb
# Vanilla with shifted window 128 and stride 64 (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=vanilla_neo shape=360m seed=42 model.window=128 model.window_stride=64 model.attention_strategy=shifted_window model.n_layer=24 training.max_steps=9393 training.batch_size=26 training.grad_accum_steps=5 training.load_balancing.strategy=aux_loss training.max_lr=0.0004 wandb_run_name=vanilla_strided data.data_dir=data/fineweb

### EVALUATE ###
# Uses training-time loss logs; no separate evaluation.

### PLOT ###
mkdir -p paper/figs/vanilla_ablations
.venv/bin/python -m plotting.ablations ATTENTION_LOCALITY --output paper/figs/vanilla_ablations/vanilla_attention_locality.pdf
