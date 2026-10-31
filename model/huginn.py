"""Depth-recurrent Neo language model in the Huginn layout.

Prelude blocks embed the tokens once, a weight-shared core block is applied
``r`` times to a latent state that starts from noise and is re-fed the prelude
output through an adapter at every recurrence, and coda blocks decode the
final state (Geiping et al., 2025, arXiv:2502.05171). ``num_steps`` selects
the executed recurrence and ``backprop_steps`` limits autograd to the final
recurrences, which is the paper's truncated backpropagation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import validate_huginn_config
from .transformer import SandwichBlockNeo
from .vanillaneo import MiddleResult, VanillaGPTNeo


class AdapterRecurrentInputInjection(nn.Module):
    """Paper adapter ``A: R^{2h} -> R^h`` over the state and prelude output."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=False)

    @staticmethod
    def prepare(prelude_output):
        return prelude_output

    def forward(self, state, prelude_output):
        return self.proj(torch.cat((state, prelude_output), dim=-1))


class HuginnGPTNeo(VanillaGPTNeo):
    """Prelude, weight-shared recurrent core with input injection, coda."""

    def __init__(self, config):
        validate_huginn_config(config)
        super().__init__(config)
        self.recurrent_depth = int(config.recurrent_depth)
        self.recurrent_state_std = float(config.recurrent_state_std)

    def _attention_architecture(self):
        return "huginn"

    def _build_middle(self, config):
        block = SandwichBlockNeo
        return {
            "h": nn.ModuleList(block(config) for _ in range(config.n_layer))
        }

    def _build_middle_boundaries(self, config):
        block = SandwichBlockNeo
        self.prelude_blocks = nn.ModuleList(
            block(config) for _ in range(config.n_prelude_layers)
        )
        self.coda_blocks = nn.ModuleList(
            block(config) for _ in range(config.n_coda_layers)
        )
        self.recurrent_input_injection = AdapterRecurrentInputInjection(
            config.n_embd
        )

    def _initial_state(self, prelude_output):
        std = self.recurrent_state_std
        state = torch.zeros(
            prelude_output.shape, device=prelude_output.device,
            dtype=torch.float32,
        )
        nn.init.trunc_normal_(
            state, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std
        )
        return state.to(prelude_output.dtype)

    def _recur(self, state, prepared_input, context):
        state = self.recurrent_input_injection(state, prepared_input)
        for block in self.transformer.h:
            state = self._apply_block(block, state, context)
        return state

    def _forward_middle(
        self,
        hidden,
        context,
        *,
        return_router_stats=False,
        return_router_diagnostics=True,
        return_router_trace=False,
        num_steps=None,
        backprop_steps=None,
        first_window_full_backprop=False,
    ):
        del (
            return_router_stats,
            return_router_diagnostics,
            return_router_trace,
            first_window_full_backprop,
        )
        depth_limit = (
            self.recurrent_depth
            if num_steps is None
            else max(1, min(int(num_steps), self.recurrent_depth))
        )
        gradient_depth = (
            depth_limit if backprop_steps is None
            else max(1, min(int(backprop_steps), depth_limit))
        )
        no_grad_depth = depth_limit - gradient_depth
        for block in self.prelude_blocks:
            hidden = self._apply_block(block, hidden, context)
        prepared_input = self.recurrent_input_injection.prepare(hidden)
        state = self._initial_state(hidden)
        grad_enabled = torch.is_grad_enabled()
        for depth in range(depth_limit):
            with torch.set_grad_enabled(
                grad_enabled and depth >= no_grad_depth
            ):
                state = self._recur(
                    state,
                    prepared_input,
                    context,
                )
        # The paper closes the recurrence with the head's norm, so the coda
        # reads a normalized state whatever depth was executed.
        state = self.transformer.ln_f(state)
        for block in self.coda_blocks:
            state = self._apply_block(block, state, context)
        return MiddleResult(state)

    def _compile_middle(self):
        for blocks in (self.prelude_blocks, self.coda_blocks):
            for index in range(len(blocks)):
                blocks[index] = torch.compile(blocks[index])
        for index in range(len(self.transformer.h)):
            self.transformer.h[index] = torch.compile(
                self.transformer.h[index],
            )
