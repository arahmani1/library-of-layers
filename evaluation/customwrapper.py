import contextlib
import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model


@register_model("latentreasoning")
class CustomPyTorchWrapper(LM):
    def __init__(
        self,
        custom_model=None,
        tokenizer=None,
        device="cuda",
        recurrent_depth=None,
        use_autocast=True,
        checkpoint=None,
        batch_size=1,
        max_batch_size=None,
    ):
        if checkpoint is not None:
            from transformers import AutoTokenizer
            from evaluation.model_loading import build_evaluation_model

            ckpt = torch.load(
                str(checkpoint), map_location="cpu", weights_only=False, mmap=True,
            )
            if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                raise ValueError(f"Checkpoint format not supported: {checkpoint}")
            cfg = ckpt.get("config")
            if cfg is None:
                raise ValueError(f"Missing config in checkpoint: {checkpoint}")
            custom_model, _ = build_evaluation_model(ckpt["model_state_dict"], cfg)
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        # lm-eval supplies batch arguments; this wrapper handles requests individually.
        del batch_size, max_batch_size
        super().__init__()
        self.model = custom_model
        self.tokenizer = tokenizer
        self._device = device
        # Score under bf16 autocast on CUDA to match training/inference dtype.
        self.use_autocast = bool(use_autocast) and str(device).startswith("cuda")
        self.recurrent_depth = recurrent_depth
        self.max_seq_len = self.model.config.block_size
        self.vocab_size = self.model.config.vocab_size

        self.model.to(self._device)
        self.model.eval()

    def _infer_logits(self, prompt_idx, prompt_answer_idx):
        if hasattr(self.model, "inference"):
            outputs = self.model.inference(
                prompt_idx, prompt_answer_idx, recurrent_depth=self.recurrent_depth,
            )
        else:
            outputs = self.model(prompt_answer_idx)
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Score every continuation token with routes chosen from the prompt."""
        results = []
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_autocast
            else contextlib.nullcontext()
        )

        with torch.no_grad(), autocast_ctx:
            for request in requests:
                context, continuation = request.args
                if not continuation:
                    results.append((0.0, True))
                    continue

                router_prompt_tokens = (
                    self.tokenizer.encode(context, add_special_tokens=False)
                    if context else []
                )
                # Shift trailing whitespace for scoring, while routing still
                # sees the original prompt, including that whitespace.
                trailing_whitespace = len(context) - len(context.rstrip())
                if trailing_whitespace:
                    continuation = context[-trailing_whitespace:] + continuation
                    context = context[:-trailing_whitespace]

                if context:
                    ctx_tokens = (
                        self.tokenizer.encode(context, add_special_tokens=False)
                        if trailing_whitespace else router_prompt_tokens
                    )
                    full_tokens = self.tokenizer.encode(
                        context + continuation, add_special_tokens=False
                    )
                    if full_tokens[:len(ctx_tokens)] != ctx_tokens:
                        raise ValueError(
                            "Context/continuation token boundary changes prompt "
                            "tokens; cannot score this request causally."
                        )
                    cont_tokens = full_tokens[len(ctx_tokens):]
                else:
                    ctx_tokens = []
                    cont_tokens = self.tokenizer.encode(
                        continuation, add_special_tokens=False
                    )

                if not cont_tokens:
                    results.append((0.0, True))
                    continue
                if not ctx_tokens or not router_prompt_tokens:
                    prefix_token_id = getattr(self.tokenizer, "eos_token_id", None)
                    if prefix_token_id is None:
                        raise ValueError(
                            "Empty scoring or routing context requires a tokenizer "
                            "EOS token."
                        )
                    if not ctx_tokens:
                        ctx_tokens = [prefix_token_id]
                    if not router_prompt_tokens:
                        router_prompt_tokens = [prefix_token_id]

                # This routing view never contains a continuation token.
                router_prompt_tokens = router_prompt_tokens[-self.max_seq_len:]
                full_tokens = ctx_tokens + cont_tokens
                max_id = max(max(full_tokens), max(router_prompt_tokens))
                if max_id >= self.vocab_size:
                    raise RuntimeError(
                        f"Tokenizer/model vocab mismatch: max token id {max_id} "
                        f">= model vocab_size {self.vocab_size}."
                    )

                router_prompt_ids = torch.tensor(
                    [router_prompt_tokens], dtype=torch.long, device=self._device
                )
                ctx_len = len(ctx_tokens)
                if len(cont_tokens) < self.max_seq_len:
                    trim = max(0, len(full_tokens) - self.max_seq_len)
                    score_windows = ((
                        full_tokens[trim:],
                        slice(ctx_len - trim - 1, len(full_tokens) - trim - 1),
                        cont_tokens,
                    ),)
                else:
                    # A continuation that fills the model window needs several
                    # passes. Each target sees the longest available past, while
                    # every pass routes from the same prompt-only view.
                    score_windows = (
                        (
                            full_tokens[max(0, index - self.max_seq_len):index],
                            slice(-1, None),
                            [full_tokens[index]],
                        )
                        for index in range(ctx_len, len(full_tokens))
                    )

                total_logprob = 0.0
                is_greedy = True
                for input_tokens, score_slice, target_tokens in score_windows:
                    input_ids = torch.tensor(
                        [input_tokens], dtype=torch.long, device=self._device
                    )
                    try:
                        logits = self._infer_logits(router_prompt_ids, input_ids)
                    except RuntimeError as error:
                        if "device-side assert" in str(error):
                            raise RuntimeError(
                                "CUDA device-side assert during forward pass. "
                                f"sequence_len={len(input_tokens)}, "
                                f"max_seq_len={self.max_seq_len}, "
                                f"max_token_id={max(input_tokens)}, "
                                f"vocab_size={self.vocab_size}."
                            ) from error
                        raise

                    cont_logits = logits[0, score_slice, :]
                    cont_targets = torch.tensor(
                        target_tokens, dtype=torch.long, device=self._device
                    )
                    log_probs = F.log_softmax(cont_logits.float(), dim=-1)
                    total_logprob += log_probs.gather(
                        -1, cont_targets.unsqueeze(-1)
                    ).sum().item()
                    is_greedy &= bool(
                        (cont_logits.argmax(dim=-1) == cont_targets).all().item()
                    )

                results.append((total_logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError("Not needed for current benchmark suite.")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_autocast
            else contextlib.nullcontext()
        )
        eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad(), autocast_ctx:
            for request in requests:
                context, generation_kwargs = request.args
                if generation_kwargs.get("do_sample", False):
                    raise ValueError("CustomPyTorchWrapper supports greedy generation only.")
                max_gen_toks = int(generation_kwargs.get("max_gen_toks", 256))
                if max_gen_toks < 1:
                    raise ValueError("max_gen_toks must be positive.")
                until = generation_kwargs.get("until", [])
                if isinstance(until, str):
                    until = [until]

                prompt_tokens = self.tokenizer.encode(
                    context, add_special_tokens=False
                )
                prompt_tokens = prompt_tokens[-self.max_seq_len:]
                if not prompt_tokens:
                    if eos_token_id is None:
                        raise ValueError("Empty prompt requires a tokenizer EOS token.")
                    prompt_tokens = [eos_token_id]
                if max(prompt_tokens) >= self.vocab_size:
                    raise RuntimeError("Tokenizer/model vocab mismatch in generation prompt.")

                # Route only from the original prompt. Generated tokens may
                # change the scoring context, but never the routing view.
                prompt_idx = torch.tensor(
                    [prompt_tokens], dtype=torch.long, device=self._device
                )
                generated = []
                stopped_text = None
                for _ in range(max_gen_toks):
                    context_tokens = prompt_tokens + generated
                    context_tokens = context_tokens[-self.max_seq_len:]
                    input_ids = torch.tensor(
                        [context_tokens], dtype=torch.long, device=self._device
                    )
                    logits = self._infer_logits(prompt_idx, input_ids)
                    next_token = int(logits[0, -1].argmax().item())
                    if next_token == eos_token_id:
                        break
                    generated.append(next_token)
                    decoded = self.tokenizer.decode(generated)
                    stop_positions = [
                        decoded.find(stop) for stop in until
                        if stop and stop in decoded
                    ]
                    if stop_positions:
                        stopped_text = decoded[:min(stop_positions)]
                        break
                results.append(
                    stopped_text
                    if stopped_text is not None
                    else self.tokenizer.decode(generated)
                )

        return results


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
