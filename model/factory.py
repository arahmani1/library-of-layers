"""Central construction for the retained model hierarchy."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from .config import GPT2SmallConfig
from .lol_dynamic_window import LoLDynamicWindow
from .huginn import HuginnGPTNeo
from .vanillaneo import VanillaGPTNeo


_ROUTED_ARCHES = frozenset({"lol_dynamic_window"})
# Architectures with a recurrent middle between a prelude and a coda.
_RECURRENT_MIDDLE_ARCHES = _ROUTED_ARCHES | {"huginn"}
_COMPOSITION_ONLY_MODEL_KEYS = frozenset({"arch", "routed"})
_CONSTRUCTORS = {
    "vanilla_neo": VanillaGPTNeo,
    "lol_dynamic_window": LoLDynamicWindow,
    "huginn": HuginnGPTNeo,
}


def architecture_is_routed(arch: str) -> bool:
    return arch in _ROUTED_ARCHES


def build_model(
    model_cfg_dict: Mapping[str, Any],
    *,
    allowed_arches: Collection[str] | None = None,
):
    arch = str(model_cfg_dict.get("arch", "vanilla_neo"))
    supported = set(_CONSTRUCTORS)
    if allowed_arches is not None:
        supported &= set(allowed_arches)
    if arch not in supported:
        expected = ", ".join(sorted(supported))
        raise ValueError(
            f"Unsupported model.arch={arch!r}; expected one of: {expected}"
        )

    declared_routed = model_cfg_dict.get("routed")
    expected_routed = architecture_is_routed(arch)
    if (
        declared_routed is not None
        and bool(declared_routed) != expected_routed
    ):
        raise ValueError(
            f"model.routed={declared_routed!r} conflicts with "
            f"model.arch={arch!r}"
        )

    if arch not in _RECURRENT_MIDDLE_ARCHES:
        for key in ("n_prelude_layers", "n_coda_layers"):
            if int(model_cfg_dict.get(key, 1)) != 1:
                raise ValueError(
                    f"model.{key} is only supported by routed or "
                    "recurrent-depth architectures; VanillaGPTNeo depth is "
                    "set by model.n_layer."
                )

    params = {
        key: value
        for key, value in model_cfg_dict.items()
        if key not in _COMPOSITION_ONLY_MODEL_KEYS and not key.startswith("_")
    }
    return _CONSTRUCTORS[arch](GPT2SmallConfig(**params)), arch
