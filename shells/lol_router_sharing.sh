# Paper figure app-fig:arch-abl-lol-router-tie: Sharing router weights across layers causes minimal change in cross-entropy loss.
# Shared training: LoL with loss-free load balancing and untied routers (seed 42); shells/lol_load_balancing.sh, TRAIN section.

### TRAIN ###
# LoL with tied routers (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.n_experts=22 model.first_window_routing=learned_token_depth model.recurrent_depth=22 model.n_prelude_layers=1 model.n_coda_layers=1 model.weight_tie_router=true training.max_steps=9393 training.batch_size=26 training.grad_accum_steps=5 training.load_balancing.strategy=loss_free training.max_lr=0.0006 wandb_run_name=lol_tied_router data.data_dir=data/fineweb

### EVALUATE ###
# Uses training-time loss logs; no separate evaluation.

### PLOT ###
mkdir -p paper/figs/lol_ablations
.venv/bin/python -m plotting.ablations ROUTER_SHARING --output paper/figs/lol_ablations/lol_router_sharing.pdf
