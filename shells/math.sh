# Paper figure app-fig:math: Extended dense-model comparison with strided windows and untied routers.
# Paper table tab:math; app-tab:math: LoL is competitive across benchmarks; extended results include strided windows and untied routers.
# Paper figure fig:math: LoL is competitive with vanilla dense models.

### TRAIN ###
# LoL (tied router) (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.recurrent_input_injection=diagonal model.n_head=32 model.n_embd=2048 model.n_experts=8 model.first_window_routing=learned_token_depth model.recurrent_depth=8 model.n_prelude_layers=4 model.n_coda_layers=4 model.weight_tie_router=true training.max_steps=19074 training.batch_size=16 training.grad_accum_steps=8 training.warmup_steps=3000 training.max_lr=0.0006 training.recurrence_curriculum.enabled=true training.recurrence_curriculum.mode=linear training.recurrence_curriculum.mean_schedule=one_minus_sqrt training.recurrence_curriculum.start_depth=1 training.recurrence_curriculum.target_mean=8 training.recurrence_curriculum.warmup_frac=0.75 training.recurrence_curriculum.backprop_depth=4 training.recurrence_curriculum.first_window_full_backprop=true data.dataset=math wandb_run_name=math_lol data.data_dir=data/math-20b-14-sep
# Vanilla (sliding window) (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=vanilla_neo shape=360m seed=42 model.window=128 model.attention_strategy=sliding_window model.n_head=32 model.n_embd=2048 model.n_layer=16 training.max_steps=15494 training.batch_size=16 training.grad_accum_steps=8 training.warmup_steps=3000 training.flop_budget.enabled=true training.flop_budget.total_flops=9.117343727137867e+19 training.load_balancing.strategy=aux_loss training.max_lr=0.0004 data.dataset=math wandb_run_name=math_vanilla_sliding data.data_dir=data/math-20b-14-sep
# Vanilla (full attention) (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=vanilla_neo shape=360m seed=42 model.n_head=32 model.n_embd=2048 model.n_layer=16 training.max_steps=15007 training.batch_size=16 training.grad_accum_steps=8 training.warmup_steps=3000 training.flop_budget.enabled=true training.flop_budget.total_flops=9.117343727137867e+19 training.load_balancing.strategy=aux_loss training.max_lr=0.0004 data.dataset=math wandb_run_name=math_vanilla_full data.data_dir=data/math-20b-14-sep

# LoL (untied router) (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.recurrent_input_injection=diagonal model.n_head=32 model.n_embd=2048 model.n_experts=8 model.first_window_routing=learned_token_depth model.recurrent_depth=8 model.n_prelude_layers=4 model.n_coda_layers=4 model.weight_tie_router=false training.max_steps=19074 training.batch_size=16 training.grad_accum_steps=8 training.warmup_steps=3000 training.flop_budget.enabled=true training.flop_budget.total_flops=9.117343727137867e+19 training.max_lr=0.0006 training.recurrence_curriculum.enabled=true training.recurrence_curriculum.mode=linear training.recurrence_curriculum.mean_schedule=one_minus_sqrt training.recurrence_curriculum.start_depth=1 training.recurrence_curriculum.target_mean=8 training.recurrence_curriculum.warmup_frac=0.75 data.dataset=math wandb_run_name=math_lol_untied data.data_dir=data/math-20b-14-sep
# Vanilla (strided window) (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=vanilla_neo shape=360m seed=42 model.window=128 model.window_stride=64 model.attention_strategy=shifted_window model.n_head=32 model.n_embd=2048 model.n_layer=16 training.max_steps=14778 training.batch_size=16 training.grad_accum_steps=8 training.warmup_steps=3000 training.flop_budget.enabled=true training.flop_budget.total_flops=9.117343727137867e+19 training.max_lr=0.0004 data.dataset=math wandb_run_name=math_vanilla_strided data.data_dir=data/math-20b-14-sep

### EVALUATE ###
# LoL (tied router) (seed 42); ARC, HellaSwag, PIQA, and WinoGrande evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_lol:?Set CHECKPOINT_math_lol to the trained math_lol checkpoint}" --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande --num_fewshot 5 --device cuda --seed 0,0,0,0 --output_path paper_runs/math_lol/benchmark-arc_easy-arc_challenge-hellaswag-piqa-winogrande
# LoL (tied router) (seed 42); GSM8K chain-of-thought evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_lol:?Set CHECKPOINT_math_lol to the trained math_lol checkpoint}" --tasks gsm8k_cot --num_fewshot 3 --device cuda --seed 0,1234,1234,1234 --output_path paper_runs/math_lol/benchmark-gsm8k_cot --log_samples
# Vanilla (sliding window) (seed 42); ARC, HellaSwag, PIQA, and WinoGrande evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_vanilla_sliding:?Set CHECKPOINT_math_vanilla_sliding to the trained math_vanilla_sliding checkpoint}" --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande --num_fewshot 5 --device cuda --seed 0,0,0,0 --output_path paper_runs/math_vanilla_sliding/benchmark-arc_easy-arc_challenge-hellaswag-piqa-winogrande
# Vanilla (sliding window) (seed 42); GSM8K chain-of-thought evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_vanilla_sliding:?Set CHECKPOINT_math_vanilla_sliding to the trained math_vanilla_sliding checkpoint}" --tasks gsm8k_cot --num_fewshot 3 --device cuda --seed 0,1234,1234,1234 --output_path paper_runs/math_vanilla_sliding/benchmark-gsm8k_cot --log_samples
# Vanilla (full attention) (seed 42); ARC, HellaSwag, PIQA, and WinoGrande evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_vanilla_full:?Set CHECKPOINT_math_vanilla_full to the trained math_vanilla_full checkpoint}" --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande --num_fewshot 5 --device cuda --seed 0,0,0,0 --output_path paper_runs/math_vanilla_full/benchmark-arc_easy-arc_challenge-hellaswag-piqa-winogrande
# Vanilla (full attention) (seed 42); GSM8K chain-of-thought evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_vanilla_full:?Set CHECKPOINT_math_vanilla_full to the trained math_vanilla_full checkpoint}" --tasks gsm8k_cot --num_fewshot 3 --device cuda --seed 0,1234,1234,1234 --output_path paper_runs/math_vanilla_full/benchmark-gsm8k_cot --log_samples

# LoL (untied router) (seed 42); ARC, HellaSwag, PIQA, and WinoGrande evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_lol_untied:?Set CHECKPOINT_math_lol_untied to the trained math_lol_untied checkpoint}" --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande --num_fewshot 5 --device cuda --seed 0,0,0,0 --output_path paper_runs/math_lol_untied/benchmark-arc_easy-arc_challenge-hellaswag-piqa-winogrande
# LoL (untied router) (seed 42); GSM8K chain-of-thought evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_lol_untied:?Set CHECKPOINT_math_lol_untied to the trained math_lol_untied checkpoint}" --tasks gsm8k_cot --num_fewshot 3 --device cuda --seed 0,1234,1234,1234 --output_path paper_runs/math_lol_untied/benchmark-gsm8k_cot --log_samples
# Vanilla (strided window) (seed 42); ARC, HellaSwag, PIQA, and WinoGrande evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_vanilla_strided:?Set CHECKPOINT_math_vanilla_strided to the trained math_vanilla_strided checkpoint}" --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande --num_fewshot 5 --device cuda --seed 0,0,0,0 --output_path paper_runs/math_vanilla_strided/benchmark-arc_easy-arc_challenge-hellaswag-piqa-winogrande
# Vanilla (strided window) (seed 42); GSM8K chain-of-thought evaluation.
.venv/bin/python -m evaluation.customwrapper run --model latentreasoning --model_args checkpoint="${CHECKPOINT_math_vanilla_strided:?Set CHECKPOINT_math_vanilla_strided to the trained math_vanilla_strided checkpoint}" --tasks gsm8k_cot --num_fewshot 3 --device cuda --seed 0,1234,1234,1234 --output_path paper_runs/math_vanilla_strided/benchmark-gsm8k_cot --log_samples

### PLOT ###
mkdir -p paper/figs
.venv/bin/python -m plotting.math_flop_loss --output paper/figs/math_flop_loss.pdf --comparison-output paper/figs/math_flop_loss_comparison.pdf

.venv/bin/python -m plotting.math_eval_table --output paper/math_evals.tex
