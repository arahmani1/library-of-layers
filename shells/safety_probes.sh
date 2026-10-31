# Paper section subsec:exp-router: Safety probes of router probabilities (reported in the text, not a figure caption).

### TRAIN ###
# LoL trained on safe score-0 and unsafe score-4/5 sequences (seed 42).
.venv/bin/torchrun --standalone --nproc_per_node=8 train.py arch=lol_dynamic_window shape=360m seed=42 model.vocab_size=100352 model.eot_token_id=100257 model.n_experts=33 model.first_window_routing=learned_token_depth model.recurrent_depth=22 model.n_prelude_layers=1 model.n_coda_layers=1 model.weight_tie_router=true training.max_steps=9390 training.batch_size=26 training.grad_accum_steps=5 training.max_lr=0.0006 data.dataset=fineweb_safety 'data.validation_sets=["safe","unsafe"]' logging.save_every=1000 wandb_run_name=safety_lol data.data_dir=data/fineweb_safety

### EVALUATE ###
# LoL trained on safe score-0 and unsafe score-4/5 sequences (seed 42); safe validation loss.
.venv/bin/python -m evaluation.evaluate_validation --checkpoint "${CHECKPOINT_safety_lol:?Set CHECKPOINT_safety_lol to the trained safety_lol checkpoint}" --data-dir data/fineweb_safety --validation-set safe --batch-size 26 --max-val-batches 0 --device cuda --json paper_runs/safety_lol/validation-safe/metrics.json
# LoL trained on safe score-0 and unsafe score-4/5 sequences (seed 42); unsafe validation loss.
.venv/bin/python -m evaluation.evaluate_validation --checkpoint "${CHECKPOINT_safety_lol:?Set CHECKPOINT_safety_lol to the trained safety_lol checkpoint}" --data-dir data/fineweb_safety --validation-set unsafe --batch-size 26 --max-val-batches 0 --device cuda --json paper_runs/safety_lol/validation-unsafe/metrics.json
# LoL trained on safe score-0 and unsafe score-4/5 sequences (seed 42); development route extraction for safety probes.
.venv/bin/python evaluation/safety_extract.py --checkpoint "${CHECKPOINT_safety_lol:?Set CHECKPOINT_safety_lol to the trained safety_lol checkpoint}" --data-dir data/fineweb_safety --documents evaluation/assets/safety_probe_documents.npz --partition development --device cuda:0 --output paper_runs/safety_lol/safety-development
# LoL trained on safe score-0 and unsafe score-4/5 sequences (seed 42); heldout route extraction for safety probes.
.venv/bin/python evaluation/safety_extract.py --checkpoint "${CHECKPOINT_safety_lol:?Set CHECKPOINT_safety_lol to the trained safety_lol checkpoint}" --data-dir data/fineweb_safety --documents evaluation/assets/safety_probe_documents.npz --partition heldout --device cuda:0 --output paper_runs/safety_lol/safety-heldout

### PLOT ###
.venv/bin/python -m evaluation.safety_paper --development-routes paper_runs/safety_lol/safety-development/windows.npz --heldout-routes paper_runs/safety_lol/safety-heldout/windows.npz --output paper_runs/safety_lol/safety-probes
