from dataclasses import dataclass
from enum import Enum


# The retained router-owned first-window mode can pin a first window.
ROUTER_OWNED_FIRST_WINDOW = frozenset({
    "learned_token_depth",
})


class RecurrentInputInjection(str, Enum):
    NONE = "none"
    DIAGONAL = "diagonal"
    ADAPTER = "adapter"


def validate_paper_normalization(config):
    """Retain affine RMSNorm and QK normalization for every paper model."""
    if str(getattr(config, 'normalization', 'affine_rmsnorm')).lower() != 'affine_rmsnorm':
        raise ValueError("normalization must be 'affine_rmsnorm' in the paper release")
    if not bool(getattr(config, 'qk_norm', True)):
        raise ValueError("qk_norm must be true in the paper release")


def validate_huginn_config(config):
    """Validate the published Huginn options for models and native counters."""
    for name, expected in (
        ('attention_strategy', 'full_causal'),
        ('recurrent_input_injection', 'adapter'),
        ('sandwich_norm', True),
        ('recurrent_state_std', 1.0),
        ('moe_ffn', False),
    ):
        actual = getattr(config, name, expected)
        actual = getattr(actual, 'value', actual)
        if actual != expected:
            raise ValueError(
                f'Huginn paper configuration requires model.{name}={expected!r}, got {actual!r}.'
            )


@dataclass
class GPT2SmallConfig:
    # --- Core architecture ---
    block_size: int = 1024     # Max sequence length
    vocab_size: int = 50432    # GPT-NeoX tokenizer-compatible vocab size
    n_layer: int = 12          # Number of transformer blocks
    n_head: int = 12           # Number of attention heads
    n_embd: int = 768          # Embedding dimension
    normalization: str = "affine_rmsnorm"

    # --- Token-routed FFN MoE ---
    moe_ffn: bool = False
    moe_ffn_scope: str = "all"  # "all" | "prelude_coda" (LoL only)
    moe_ffn_n_experts: int = 8
    moe_ffn_top_k: int = 2
    moe_ffn_hidden_dim: int | None = None  # Exact per-expert intermediate width.
    moe_ffn_aux_loss_coeff: float = 0.01

    # --- Router / expert settings ---
    n_experts: int = 6         # Number of expert transformer blocks in the pool
    attention_strategy: str | None = None
    window: int | None = None
    recurrent_depth: int = 10  # Number of routed depth steps
    n_prelude_layers: int = 1  # Full-sequence blocks before the routed middle
    n_coda_layers: int = 1     # Full-sequence blocks after the routed middle
    weight_tie_router: bool = True
    aux_loss_coeff: float = 0.1  # Coefficient for load-balancing auxiliary loss
    eot_token_id: int = 0      # Delimiter token used for EOT-segmented routing

    # --- Attention stability ---
    qk_norm: bool = True       # Archived field; QK RMSNorm before RoPE is required.

    # --- Residual and recurrent-input strategy ---
    recurrent_input_injection: RecurrentInputInjection | str = (
        RecurrentInputInjection.NONE
    )                                # "none" | "diagonal" | "adapter"

    # --- Huginn-specific ---
    sandwich_norm: bool = True       # Re-normalize the residual stream after
                                     # every sublayer (paper sandwich blocks).
    recurrent_state_std: float = 1.0 # Paper std-1 random latent state,
                                     # truncated at three std. Other values
                                     # are rejected only for Huginn.


    grouped_window_dispatch: bool = False
    depth_expert_film: bool = False

    # --- LoLDynamicWindow-specific ---
    first_window_routing: str = "learned_token_depth"
    first_window_length: int | None = None
                                     # Only for learned_token_depth routing.
                                     # Pins positions [0, first_window_length)
                                     # of every document into one fixed window;
                                     # routing uses the depth query and first-token state.
    window_stride: int | None = None # Per-depth shift applied to window boundaries.
                                     # At depth d, per-doc position p sits in window
                                     # ((p + d * window_stride) // window). Choose
                                     # >= (seq_len - window) / (recurrent_depth - 1)
                                     # so the last token can see the first by the
                                     # final depth.
    hidden_key_span: tuple[int, int] | None = None
                                     # Per-document positions [start, stop) only
                                     # they themselves may attend to in the
                                     # full-sequence blocks.

    def __post_init__(self):
        self.moe_ffn = bool(self.moe_ffn)
        self.moe_ffn_scope = str(self.moe_ffn_scope).lower()
        if self.moe_ffn_scope not in {"all", "prelude_coda"}:
            raise ValueError(
                "moe_ffn_scope must be 'all' or 'prelude_coda', "
                f"got {self.moe_ffn_scope!r}."
            )
        self.normalization = str(self.normalization).lower()
        validate_paper_normalization(self)
        self.first_window_routing = str(self.first_window_routing).lower()
        if self.first_window_routing != "learned_token_depth":
            raise ValueError("first_window_routing must be 'learned_token_depth'.")
        self.moe_ffn_n_experts = int(self.moe_ffn_n_experts)
        self.moe_ffn_top_k = int(self.moe_ffn_top_k)
        if self.moe_ffn_hidden_dim is not None and (
            isinstance(self.moe_ffn_hidden_dim, bool)
            or not isinstance(self.moe_ffn_hidden_dim, int)
            or self.moe_ffn_hidden_dim < 1
        ):
            raise ValueError("moe_ffn_hidden_dim must be a positive integer or None.")
        if self.moe_ffn_n_experts < 1:
            raise ValueError("moe_ffn_n_experts must be positive.")
        if not 2 <= self.moe_ffn_top_k <= self.moe_ffn_n_experts:
            raise ValueError(
                "moe_ffn_top_k must satisfy 2 <= top_k <= "
                "moe_ffn_n_experts."
            )
        if float(self.moe_ffn_aux_loss_coeff) < 0.0:
            raise ValueError(
                "moe_ffn_aux_loss_coeff must be nonnegative."
            )
        self.n_prelude_layers = int(self.n_prelude_layers)
        self.n_coda_layers = int(self.n_coda_layers)
        if self.n_prelude_layers < 1 or self.n_coda_layers < 1:
            raise ValueError(
                "n_prelude_layers and n_coda_layers must be positive."
            )
        self.sandwich_norm = bool(self.sandwich_norm)
        self.recurrent_state_std = float(self.recurrent_state_std)
        if self.recurrent_state_std < 0.0:
            raise ValueError("recurrent_state_std must be nonnegative.")

        try:
            self.recurrent_input_injection = RecurrentInputInjection(
                self.recurrent_input_injection
            )
        except ValueError as exc:
            valid = ", ".join(
                repr(strategy.value)
                for strategy in RecurrentInputInjection
            )
            raise ValueError(
                "recurrent_input_injection must be one of "
                f"{valid}, got {self.recurrent_input_injection!r}."
            ) from exc
        self.weight_tie_router = bool(self.weight_tie_router)
        if self.hidden_key_span is not None:
            span = tuple(int(bound) for bound in self.hidden_key_span)
            if len(span) != 2 or not 0 <= span[0] < span[1]:
                raise ValueError(
                    "hidden_key_span must be [start, stop) with "
                    f"0 <= start < stop, got {self.hidden_key_span!r}."
                )
            self.hidden_key_span = span
        if self.first_window_length is not None:
            # A pin wider than model.window only widens the packed window
            # slot (attention_strategies.pack_width); routing keeps window.
            self.first_window_length = int(self.first_window_length)
            if self.first_window_length < 1:
                raise ValueError("first_window_length must be positive.")
        if type(self.depth_expert_film) is not bool:
            raise ValueError("depth_expert_film must be a boolean")
