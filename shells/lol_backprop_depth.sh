# Paper figure app-fig:arch-abl-lol-backprop: Increasing depth during training saves FLOPs with only a minor increase in final loss.

### TRAIN ###
# LoL growing from depth 1 to 16, full backpropagation (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.recurrent_input_injection=diagonal model.n_experts=16 model.first_window_routing=learned_token_depth model.recurrent_depth=16 model.n_prelude_layers=4 model.n_coda_layers=4 model.weight_tie_router=true training.max_steps=9393 training.batch_size=26 training.grad_accum_steps=5 training.max_lr=0.0006 training.recurrence_curriculum.enabled=true training.recurrence_curriculum.mode=linear training.recurrence_curriculum.mean_schedule=one_minus_sqrt training.recurrence_curriculum.start_depth=1 training.recurrence_curriculum.target_mean=16 training.recurrence_curriculum.warmup_frac=0.75 wandb_run_name=lol_growing_depth data.data_dir=data/fineweb
# LoL growing from depth 1 to 16, final-8 backpropagation and full first-window backpropagation (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.recurrent_input_injection=diagonal model.n_experts=16 model.first_window_routing=learned_token_depth model.recurrent_depth=16 model.n_prelude_layers=4 model.n_coda_layers=4 model.weight_tie_router=true training.max_steps=9393 training.batch_size=26 training.grad_accum_steps=5 training.max_lr=0.0006 training.recurrence_curriculum.enabled=true training.recurrence_curriculum.mode=linear training.recurrence_curriculum.mean_schedule=one_minus_sqrt training.recurrence_curriculum.start_depth=1 training.recurrence_curriculum.target_mean=16 training.recurrence_curriculum.warmup_frac=0.75 training.recurrence_curriculum.backprop_depth=8 training.recurrence_curriculum.first_window_full_backprop=true wandb_run_name=lol_truncated_backprop data.data_dir=data/fineweb
# LoL at fixed depth 16, full backpropagation (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.recurrent_input_injection=diagonal model.n_experts=16 model.first_window_routing=learned_token_depth model.recurrent_depth=16 model.n_prelude_layers=4 model.n_coda_layers=4 model.weight_tie_router=true training.max_steps=9393 training.batch_size=26 training.grad_accum_steps=5 training.max_lr=0.0006 wandb_run_name=lol_fixed_depth data.data_dir=data/fineweb

### EVALUATE ###
# Uses training-time loss logs; no separate evaluation.

### PLOT ###
mkdir -p paper/figs/lol_ablations
.venv/bin/python -m plotting.ablations BACKPROP_DEPTH --output paper/figs/lol_ablations/lol_backprop_depth.pdf
