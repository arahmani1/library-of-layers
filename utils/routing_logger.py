"""Per-depth block-routing diagnostics for multilingual validation.

One :class:`RoutingAccumulator` spans every validation language of a single
evaluation step. ``train.py`` feeds it the already-reduced per-depth dispatch
histogram each language produces for free, plus a bounded reservoir of raw
window routes taken from the routing trace, and :class:`RoutingLogger` turns
the pair into one W&B payload under ``router/multilingual/*``.

Normalisation is always within ``(language, depth)``. A global normalisation
would turn every panel into a picture of eval-split sizes.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import wandb  # noqa: E402

from utils.routing_viz import (  # noqa: E402
    entropy_bits,
    fig_cluster_space,
    jensen_shannon,
    language_leaf_order,
    modal_block,
    pairwise_language_distance,
    pooled_distribution,
    route_centroid_accuracy,
    signed_displacement,
)
from utils.routing_viz_multi import (  # noqa: E402
    deviation_vmax,
    fig_language_similarity,
    fig_routing_map_multi,
    fig_signed_deviation,
    fig_signed_displacement,
    fig_utilisation_multi,
)

# Canonical, sorted registry of the FineWeb2 languages this repository
# prepares (``data/prepare_fineweb2_multilingual.LANGUAGES``). A
# run evaluates a subset, but panel and colour order come from this tuple, so
# two runs' figures can be compared side by side.
LANGS = (
    "arb_Arab",
    "cmn_Hani",
    "deu_Latn",
    "fra_Latn",
    "hin_Deva",
    "jpn_Jpan",
    "kor_Hang",
    "rus_Cyrl",
    "spa_Latn",
    "tur_Latn",
)

# Fixed constant, deliberately not the run seed and not the step: the
# reservoir must be statistically comparable frame to frame or the cluster
# scatter means nothing across eval steps.
RESERVOIR_SEED = 1_000_003

_DEAD_BLOCK_FRACTION = 0.2

# Window ordinals (index inside the document) with their own cluster panel
# and route-to-language accuracy; a pooled PCA is dominated by position.
PER_ORDINAL_DIAGNOSTICS = (1, 2)
_MIN_ORDINAL_WINDOWS = 10


def active_languages(validation_names) -> tuple[str, ...]:
    """Order a run's validation splits by the canonical registry."""

    names = tuple(str(name) for name in validation_names)
    known = tuple(name for name in LANGS if name in names)
    return known + tuple(sorted(set(names) - set(known)))


def gini(load) -> float:
    """Gini coefficient of a non-negative load vector."""

    values = np.sort(np.clip(np.asarray(load, dtype=np.float64), 0.0, None))
    total = float(values.sum())
    if total <= 0.0 or values.size == 0:
        return 0.0
    count = values.size
    ranks = np.arange(1, count + 1, dtype=np.float64)
    return float(
        2.0 * float((ranks * values).sum()) / (count * total)
        - (count + 1) / count
    )


def mean_pairwise_jsd(routing) -> float:
    """Mean of the upper triangle of the language x language JSD matrix."""

    routing = np.asarray(routing, dtype=np.float64)
    if routing.shape[0] < 2:
        return 0.0
    distance = pairwise_language_distance(routing)
    upper = np.triu_indices(distance.shape[0], k=1)
    return float(distance[upper].mean())


def routing_scalars(routing, languages) -> dict[str, float]:
    """Every cheap routing scalar derived from ``[L, D, K]`` shares."""

    routing = np.asarray(routing, dtype=np.float64)
    languages = tuple(languages)
    pooled = pooled_distribution(routing)
    specialisation = jensen_shannon(routing, pooled[None]).mean(-1)
    load = routing.sum((0, 1))
    load = load / max(float(load.sum()), 1e-12)
    blocks = routing.shape[2]
    modal = modal_block(routing)

    scalars = {
        "mean_specialisation": float(specialisation.mean()),
        "mean_pairwise_jsd": mean_pairwise_jsd(routing),
        "mean_entropy_bits": float(entropy_bits(routing).mean()),
        "load_gini": gini(load),
        "dead_blocks": float(
            (load < _DEAD_BLOCK_FRACTION / max(blocks, 1)).sum()
        ),
        "mean_abs_displacement": float(
            np.abs(signed_displacement(routing)).mean()
        ),
        "frac_depths_lang_specific": float(
            (modal != modal[0][None]).any(0).mean()
        ),
    }
    scalars.update({
        f"spec/{language}": float(specialisation[index])
        for index, language in enumerate(languages)
    })
    return scalars


class RoutingAccumulator:
    """Accumulate per-language routing over one whole evaluation step.

    Two count tensors are kept on purpose. ``counts`` is the window-level
    histogram over every eval batch, fed from the dispatch statistics the
    routed forward already produces. ``counts_window_sample`` and
    ``counts_doc`` come from the same bounded routing trace, so the
    window-versus-document comparison that detects a tokenizer-fertility
    confound is drawn from one sample rather than two.
    """

    def __init__(
        self,
        languages,
        n_depths: int,
        n_blocks: int,
        *,
        max_windows: int = 4000,
        seed: int = RESERVOIR_SEED,
        rank: int = 0,
        device=None,
    ):
        self.languages = tuple(languages)
        self.n_depths = int(n_depths)
        self.n_blocks = int(n_blocks)
        self.max_windows = int(max_windows)
        shape = (len(self.languages), self.n_depths, self.n_blocks)
        self.counts = torch.zeros(shape, dtype=torch.int64, device=device)
        self.counts_window_sample = torch.zeros(
            shape, dtype=torch.int64, device=device
        )
        self.counts_doc = torch.zeros(
            shape, dtype=torch.float64, device=device
        )
        self.doc_totals = torch.zeros(
            len(self.languages), dtype=torch.float64, device=device
        )
        # Offset by rank only: fixed across eval steps as required, but
        # decorrelated across shards so the gathered reservoir is not the same
        # positional slice taken from every rank.
        self._generator = torch.Generator()
        self._generator.manual_seed(int(seed) + int(rank))
        self._keys = torch.empty(0, dtype=torch.float64)
        self._routes = torch.empty((0, self.n_depths), dtype=torch.int64)
        self._langs = torch.empty(0, dtype=torch.int64)
        self._tokens = torch.empty(0, dtype=torch.int64)
        # Window index inside its document; -1 when the caller gave none.
        self._ordinals = torch.empty(0, dtype=torch.int64)

    # -- ingestion ---------------------------------------------------------

    def add_reduced_counts(self, lang_index: int, per_depth_block) -> None:
        """Add an already all-reduced ``[D, K]`` window histogram.

        ``all_reduce`` deliberately leaves ``counts`` alone, so this must be
        called with the same values on every rank.
        """

        counts = torch.as_tensor(per_depth_block).detach()
        if tuple(counts.shape) != (self.n_depths, self.n_blocks):
            raise ValueError(
                f"expected a [{self.n_depths}, {self.n_blocks}] histogram, "
                f"got {tuple(counts.shape)}"
            )
        rounded = counts.round()
        assert torch.equal(counts, rounded), (
            "reduced routing counts must be integer-valued before int64 "
            "conversion"
        )
        self.counts[int(lang_index)] += rounded.to(
            device=self.counts.device, dtype=torch.int64
        )

    def update(
        self, routes, lang_ids, n_tokens, *, doc_ids=None, valid=None,
        ordinals=None,
    ) -> None:
        """Accumulate one batch of raw window routes.

        ``routes`` is ``[B, D]`` block ids, ``lang_ids`` and ``n_tokens`` are
        ``[B]``. Windows with no live tokens, and windows the caller marks
        invalid, are dropped before anything is counted: their routing is
        padding-driven and would otherwise pollute every panel. ``ordinals``
        (``[B]``, the window's index inside its document) feeds the
        per-ordinal panels; without it those panels are skipped.
        """

        routes = torch.as_tensor(routes).detach().to("cpu", torch.int64)
        lang_ids = torch.as_tensor(lang_ids).detach().to("cpu", torch.int64)
        n_tokens = torch.as_tensor(n_tokens).detach().to("cpu", torch.int64)
        if routes.ndim != 2 or routes.shape[1] != self.n_depths:
            raise ValueError(
                f"expected [B, {self.n_depths}] routes, "
                f"got {tuple(routes.shape)}"
            )
        if routes.numel():
            assert (
                int(routes.min()) >= 0 and int(routes.max()) < self.n_blocks
            ), (
                f"block ids must lie in [0, {self.n_blocks}); got "
                f"[{int(routes.min())}, {int(routes.max())}]"
            )

        keep = n_tokens > 0
        if valid is not None:
            keep &= torch.as_tensor(valid).detach().to("cpu", torch.bool)
        if ordinals is None:
            ordinals = torch.full_like(n_tokens, -1)
        else:
            ordinals = torch.as_tensor(ordinals).detach().to(
                "cpu", torch.int64
            )
        routes = routes[keep]
        lang_ids = lang_ids[keep]
        n_tokens = n_tokens[keep]
        ordinals = ordinals[keep]
        if doc_ids is not None:
            doc_ids = torch.as_tensor(doc_ids).detach().to(
                "cpu", torch.int64
            )[keep]
        if routes.numel() == 0:
            return

        depths = torch.arange(self.n_depths).expand_as(routes)
        blocks, languages = self.n_blocks, len(self.languages)
        flat = (lang_ids[:, None] * self.n_depths + depths) * blocks + routes
        self.counts_window_sample.view(-1).add_(
            torch.bincount(
                flat.reshape(-1),
                minlength=languages * self.n_depths * blocks,
            ).to(self.counts_window_sample.device)
        )

        if doc_ids is not None:
            documents, index = torch.unique(doc_ids, return_inverse=True)
            total = documents.numel()
            per_document = torch.bincount(
                ((index[:, None] * self.n_depths + depths) * blocks
                 + routes).reshape(-1),
                minlength=total * self.n_depths * blocks,
            ).view(total, self.n_depths, blocks).to(torch.float64)
            shares = per_document / per_document.sum(
                -1, keepdim=True
            ).clamp(min=1.0)
            owner = torch.zeros(total, dtype=torch.int64)
            owner.scatter_(0, index, lang_ids)
            self.counts_doc.index_add_(
                0, owner.to(self.counts_doc.device),
                shares.to(self.counts_doc.device),
            )
            self.doc_totals.index_add_(
                0, owner.to(self.doc_totals.device),
                torch.ones(total, dtype=torch.float64,
                           device=self.doc_totals.device),
            )

        # Priority (A-Res) sampling: give every window an i.i.d. key and keep
        # the smallest ``max_windows``. Exactly uniform without replacement,
        # order independent, and trivially mergeable across ranks.
        self._keys = torch.cat([
            self._keys,
            torch.rand(
                routes.shape[0], generator=self._generator,
                dtype=torch.float64,
            ),
        ])
        self._routes = torch.cat([self._routes, routes])
        self._langs = torch.cat([self._langs, lang_ids])
        self._tokens = torch.cat([self._tokens, n_tokens])
        self._ordinals = torch.cat([self._ordinals, ordinals])
        self._trim()

    def _trim(self) -> None:
        order = torch.argsort(self._keys, stable=True)[:self.max_windows]
        self._keys = self._keys[order]
        self._routes = self._routes[order]
        self._langs = self._langs[order]
        self._tokens = self._tokens[order]
        self._ordinals = self._ordinals[order]

    # -- reduction ---------------------------------------------------------

    def all_reduce(self) -> bool:
        """Sum sampled counts across ranks and gather the reservoir.

        Returns ``True`` on the rank that should proceed to logging.
        """

        if not (dist.is_available() and dist.is_initialized()):
            self._trim()
            return True
        dist.all_reduce(self.counts_window_sample)
        dist.all_reduce(self.counts_doc)
        dist.all_reduce(self.doc_totals)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, {
            "keys": self._keys,
            "routes": self._routes,
            "langs": self._langs,
            "tokens": self._tokens,
            "ordinals": self._ordinals,
        })
        if dist.get_rank() != 0:
            return False
        self._keys = torch.cat([shard["keys"] for shard in gathered])
        self._routes = torch.cat([shard["routes"] for shard in gathered])
        self._langs = torch.cat([shard["langs"] for shard in gathered])
        self._tokens = torch.cat([shard["tokens"] for shard in gathered])
        self._ordinals = torch.cat([shard["ordinals"] for shard in gathered])
        self._trim()
        return True

    # -- outputs -----------------------------------------------------------

    @staticmethod
    def _normalise(counts) -> torch.Tensor:
        counts = counts.detach().cpu().to(torch.float64)
        return counts / counts.sum(-1, keepdim=True).clamp(min=1.0)

    def finalize(self) -> torch.Tensor:
        """``[L, D, K]`` window-level shares normalised within (lang, depth)."""

        return self._normalise(self.counts)

    def finalize_window_sample(self) -> torch.Tensor:
        """Window-level shares over the traced sample only."""

        return self._normalise(self.counts_window_sample)

    def finalize_doc(self) -> torch.Tensor:
        """Document-macro-averaged shares over the traced sample only."""

        totals = self.doc_totals.detach().cpu().to(torch.float64)
        averaged = self.counts_doc.detach().cpu().to(torch.float64) / (
            totals.clamp(min=1.0)[:, None, None]
        )
        return averaged / averaged.sum(-1, keepdim=True).clamp(min=1e-12)

    @property
    def reservoir(self):
        """``(routes, lang_ids, n_tokens)`` of the retained subsample."""

        return self._routes, self._langs, self._tokens

    @property
    def reservoir_ordinals(self):
        """Window-in-document index of each retained window (-1: unknown)."""

        return self._ordinals

    def ordinal_scalars(self) -> dict[str, float]:
        """Route-to-language accuracy for each diagnosed window ordinal."""

        scalars = {}
        for ordinal in PER_ORDINAL_DIAGNOSTICS:
            selected = self._ordinals == ordinal
            if int(selected.sum()) < _MIN_ORDINAL_WINDOWS:
                continue
            scalars[f"route_lang_accuracy_w{ordinal}"] = (
                route_centroid_accuracy(
                    self._routes[selected].numpy(),
                    self._langs[selected].numpy(),
                    self.n_blocks,
                )
            )
        return scalars


@dataclass(frozen=True)
class RoutingLoggerConfig:
    """Hydra-facing knobs for the multilingual routing diagnostics."""

    enabled: bool = True
    namespace: str = "router/multilingual"
    pivot: str = "pooled"
    max_windows: int = 4000
    trace_batches: int = 1
    fertility_alert: float = 0.05
    max_table_rows: int = 50_000

    @classmethod
    def from_config(cls, cfg) -> "RoutingLoggerConfig":
        section = cfg.logging.get("routing_diagnostics", {}) or {}
        defaults = cls()
        return cls(
            enabled=bool(section.get("enabled", defaults.enabled)),
            namespace=str(section.get("namespace", defaults.namespace)),
            pivot=str(section.get("pivot") or defaults.pivot),
            max_windows=int(
                section.get("max_windows", defaults.max_windows)
            ),
            trace_batches=int(
                section.get("trace_batches", defaults.trace_batches)
            ),
            fertility_alert=float(
                section.get("fertility_alert", defaults.fertility_alert)
            ),
            max_table_rows=int(
                section.get("max_table_rows", defaults.max_table_rows)
            ),
        )


class RoutingLogger:
    """Turn one evaluation's accumulator into a single W&B payload."""

    def __init__(
        self,
        config: RoutingLoggerConfig,
        languages,
        n_depths: int,
        n_blocks: int,
    ):
        self.config = config
        self.languages = tuple(languages)
        self.n_depths = int(n_depths)
        self.n_blocks = int(n_blocks)
        # Cached once at the first eval. Recomputing either per step would
        # make consecutive frames incomparable: rows would jump
        # around, and a rescaled colourbar would hide a real change.
        self._leaf_order: tuple[int, ...] | None = None
        self._deviation_vmax: float | None = None
        wandb.define_metric(
            f"{config.namespace}/*", step_metric="global_step"
        )

    def language_index(self, name: str) -> int:
        return self.languages.index(str(name))

    def _pivot(self) -> str:
        pivot = self.config.pivot
        if pivot in (None, "", "pooled") or pivot in self.languages:
            return pivot or "pooled"
        print(
            f"WARNING: routing pivot {pivot!r} is not an evaluated language "
            f"{self.languages}; falling back to the pooled reference."
        )
        return "pooled"

    def _figure_fields(self, routing, accumulator) -> dict[str, object]:
        namespace = self.config.namespace
        pivot = self._pivot()
        array = np.asarray(routing)
        if self._leaf_order is None:
            self._leaf_order = language_leaf_order(array)
        if self._deviation_vmax is None:
            self._deviation_vmax = deviation_vmax(
                array, self.languages, pivot
            )

        routes, lang_ids, tokens = accumulator.reservoir
        builders = (
            ("signed_deviation", lambda: fig_signed_deviation(
                array, self.languages, self.n_blocks,
                reference=pivot, vmax=self._deviation_vmax,
            )),
            ("map", lambda: fig_routing_map_multi(
                array, self.languages, self.n_blocks,
                order=self._leaf_order,
            )),
            ("utilisation", lambda: fig_utilisation_multi(
                array, self.languages, self.n_blocks,
                order=self._leaf_order,
            )),
            ("signed_displacement", lambda: fig_signed_displacement(
                array, self.languages,
            )),
            ("language_similarity", lambda: fig_language_similarity(
                array, self.languages,
            )),
            ("cluster_space", lambda: fig_cluster_space(
                routes.numpy(), lang_ids.numpy(), self.languages,
                self.n_blocks,
                n_tokens=tokens.numpy(),
            )),
        )
        ordinals = accumulator.reservoir_ordinals
        for ordinal in PER_ORDINAL_DIAGNOSTICS:
            selected = ordinals == ordinal
            if int(selected.sum()) < _MIN_ORDINAL_WINDOWS:
                continue
            builders += ((
                f"cluster_space_w{ordinal}",
                lambda selected=selected, ordinal=ordinal: fig_cluster_space(
                    routes[selected].numpy(), lang_ids[selected].numpy(),
                    self.languages, self.n_blocks,
                    n_tokens=tokens[selected].numpy(),
                    title=(
                        f"Window {ordinal} of each document only "
                        "(PCA of one-hot routes, jittered)"
                    ),
                ),
            ),)

        fields: dict[str, object] = {
            f"{namespace}/deviation_vmax": float(self._deviation_vmax),
        }
        for name, builder in builders:
            figure = None
            try:
                figure = builder()
                fields[f"{namespace}/{name}"] = wandb.Image(figure)
            except Exception as error:
                print(
                    f"WARNING: routing figure {name!r} failed "
                    f"({type(error).__name__}: {error}); continuing."
                )
            finally:
                if figure is not None:
                    plt.close(figure)
        return fields

    def _table_field(self, routing) -> dict[str, object]:
        namespace = self.config.namespace
        array = np.asarray(routing)
        rows = int(array.size)
        if rows < self.config.max_table_rows:
            table = wandb.Table(columns=["lang", "depth", "block", "p"])
            for language_index, language in enumerate(self.languages):
                for depth in range(array.shape[1]):
                    for block in range(array.shape[2]):
                        table.add_data(
                            language, depth, block,
                            float(array[language_index, depth, block]),
                        )
            return {f"{namespace}/distribution": table}
        table = wandb.Table(columns=["lang", "depth", "block", "p"])
        modal = modal_block(array)
        for language_index, language in enumerate(self.languages):
            for depth in range(array.shape[1]):
                block = int(modal[language_index, depth])
                table.add_data(
                    language, depth, block,
                    float(array[language_index, depth, block]),
                )
        print(
            f"WARNING: routing table would need {rows} rows "
            f"(>= {self.config.max_table_rows}); logging the argmax slice."
        )
        return {f"{namespace}/distribution_argmax": table}

    def _log_artifact(self, accumulator, step: int) -> None:
        run = wandb.run
        if run is None:
            return
        schema_version = 2
        artifact = wandb.Artifact(
            f"routing-counts-{run.id}",
            type="eval-diagnostics",
            metadata={
                "schema_version": schema_version,
                "global_step": int(step),
            },
        )
        routes, lang_ids, n_tokens = accumulator.reservoir
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "counts.npz")
            np.savez_compressed(
                path,
                counts=accumulator.counts.cpu().numpy(),
                counts_window_sample=(
                    accumulator.counts_window_sample.cpu().numpy()
                ),
                counts_doc=accumulator.counts_doc.cpu().numpy(),
                doc_totals=accumulator.doc_totals.cpu().numpy(),
                routes=routes.numpy(),
                lang_ids=lang_ids.numpy(),
                n_tokens=n_tokens.numpy(),
                ordinals=accumulator.reservoir_ordinals.numpy(),
                langs=np.array(self.languages),
                K=np.array(self.n_blocks),
                D=np.array(self.n_depths),
                step=np.array(int(step)),
                schema_version=np.array(schema_version),
            )
            artifact.add_file(path, name="counts.npz")
            run.log_artifact(
                artifact, aliases=["latest", f"step-{int(step)}"]
            )

    def log_fields(self, accumulator, step: int) -> dict[str, object]:
        """Build the merged scalar/figure/table payload for one eval step."""

        namespace = self.config.namespace
        routing = accumulator.finalize().numpy()
        fields = {
            f"{namespace}/{key}": value
            for key, value in routing_scalars(routing, self.languages).items()
        }

        sampled_window = accumulator.finalize_window_sample().numpy()
        sampled_doc = accumulator.finalize_doc().numpy()
        window_jsd = mean_pairwise_jsd(sampled_window)
        document_jsd = mean_pairwise_jsd(sampled_doc)
        gap = abs(document_jsd - window_jsd)
        fields.update({
            f"{namespace}/mean_pairwise_jsd_doc": document_jsd,
            f"{namespace}/mean_pairwise_jsd_window_sample": window_jsd,
            f"{namespace}/fertility_gap": gap,
        })
        fields.update({
            f"{namespace}/{key}": value
            for key, value in accumulator.ordinal_scalars().items()
        })
        if gap > self.config.fertility_alert:
            print(
                f"WARNING: window-level and document-level pairwise JSD "
                f"differ by {gap:.4f} (> {self.config.fertility_alert}); the "
                "apparent language effect is partly a tokenizer-fertility "
                "length effect."
            )

        # Artifact and presentation failures are independent: neither may
        # cost the other or the scalars collected by this evaluation.
        try:
            self._log_artifact(accumulator, step)
        except Exception as error:
            print(
                f"WARNING: routing artifact upload failed "
                f"({type(error).__name__}: {error}); continuing."
            )
        try:
            fields.update(self._figure_fields(routing, accumulator))
            fields.update(self._table_field(routing))
        except Exception as error:
            print(
                f"WARNING: routing figure payload failed "
                f"({type(error).__name__}: {error}); "
                "logging scalars only."
            )
        return fields
