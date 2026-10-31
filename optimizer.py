"""Muon with AdamW for embeddings, vectors, and other ineligible parameters."""

from typing import Any

import torch
from torch import nn


class _MatrixBatchMuon(torch.optim.Muon):
    """Run PyTorch Muon on 2-D views of marked matrix batches."""

    def __init__(self, param_groups, **kwargs):
        expanded_groups = []
        self._matrix_batches = []
        for group in param_groups:
            expanded = []
            for parameter in group["params"]:
                if not bool(
                    getattr(parameter, "_muon_matrix_batch", False)
                ):
                    expanded.append(parameter)
                    continue
                if parameter.ndim != 3:
                    raise ValueError(
                        "Muon matrix batches must have shape "
                        "[batch, rows, columns]"
                    )
                views = tuple(
                    nn.Parameter(
                        matrix.detach(),
                        requires_grad=parameter.requires_grad,
                    )
                    for matrix in parameter
                )
                expanded.extend(views)
                self._matrix_batches.append((parameter, views))
            expanded_groups.append({**group, "params": expanded})
        super().__init__(expanded_groups, **kwargs)

    def _sync_matrix_batch_gradients(self):
        for source, views in self._matrix_batches:
            for expert, view in enumerate(views):
                view.grad = (
                    None if source.grad is None else source.grad[expert]
                )

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for source, _ in self._matrix_batches:
            if source.grad is None:
                continue
            if set_to_none:
                source.grad = None
            else:
                source.grad.zero_()

    def step(self, closure=None):
        if closure is None:
            self._sync_matrix_batch_gradients()
            return super().step()

        def sync_after_closure():
            loss = closure()
            self._sync_matrix_batch_gradients()
            return loss

        return super().step(sync_after_closure)


class CompositeOptimizer:
    """Make Muon and its fallback AdamW optimizer behave like one optimizer."""

    def __init__(self, muon, adamw=None):
        self.optimizers = {"muon": muon}
        if adamw is not None:
            self.optimizers["adamw"] = adamw

    @property
    def param_groups(self):
        return [
            group
            for optimizer in self.optimizers.values()
            for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        return {
            "format_version": 1,
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
        }

    def load_state_dict(self, state_dict):
        saved = state_dict.get("optimizers")
        if state_dict.get("format_version") != 1 or not isinstance(saved, dict):
            raise ValueError("Checkpoint does not contain a compatible Muon state")
        if set(saved) != set(self.optimizers):
            raise ValueError(
                "Checkpoint optimizer components do not match the current model"
            )
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(saved[name])


def _split_for_muon(model, param_groups):
    """Split hidden matrices from embeddings and all remaining parameters."""

    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    muon_groups, adamw_groups = [], []

    for group in param_groups:
        muon_eligible = bool(group.get("_muon_eligible", True))
        metadata = {
            key: value
            for key, value in group.items()
            if key not in {"params", "_muon_eligible"}
        }
        params = [parameter for parameter in group["params"] if parameter.requires_grad]
        muon = [
            parameter
            for parameter in params
            if (
                muon_eligible
                and (
                    parameter.ndim == 2
                    or bool(getattr(parameter, "_muon_matrix_batch", False))
                )
                and id(parameter) not in embedding_ids
                and not bool(getattr(parameter, "_force_adamw", False))
            )
        ]
        muon_ids = {id(parameter) for parameter in muon}
        adamw = [parameter for parameter in params if id(parameter) not in muon_ids]

        if muon:
            muon_groups.append({"params": muon, **metadata})
        if adamw:
            adamw_groups.append({"params": adamw, **metadata})

    return muon_groups, adamw_groups


def build_optimizer(
    model: nn.Module,
    training_config: dict[str, Any],
    param_groups=None,
):
    """Build Muon plus AdamW for unsupported parameters."""

    name = str(training_config.get("optimizer", "muon")).lower()
    if name != "muon":
        raise ValueError(f"Unsupported optimizer {name!r}; expected Muon")

    max_lr = float(training_config["max_lr"])
    min_lr = float(training_config["min_lr"])
    weight_decay = float(training_config["weight_decay"])
    if param_groups is None:
        param_groups = [{
            "params": list(model.parameters()),
            "lr": max_lr,
            "_max_lr": max_lr,
            "_min_lr": float(training_config["min_lr"]),
            "_lr_group": "main",
        }]
    else:
        param_groups = [
            {**group, "params": list(group["params"])}
            for group in param_groups
        ]

    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    decay_groups = []
    for group in param_groups:
        if "weight_decay" in group:
            decay_groups.append(group)
            continue
        metadata = {key: value for key, value in group.items() if key != "params"}
        decay = []
        no_decay = []
        for parameter in group["params"]:
            target = (
                no_decay
                if (
                    parameter.ndim < 2
                    or id(parameter) in embedding_ids
                    or bool(getattr(parameter, "_no_weight_decay", False))
                )
                else decay
            )
            target.append(parameter)
        if decay:
            decay_groups.append(
                {"params": decay, "weight_decay": weight_decay, **metadata}
            )
        if no_decay:
            decay_groups.append(
                {"params": no_decay, "weight_decay": 0.0, **metadata}
            )
    param_groups = decay_groups

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("Muon requires PyTorch 2.9 or newer")

    muon_groups, adamw_groups = _split_for_muon(model, param_groups)
    if not muon_groups:
        raise ValueError("Muon was selected, but the model has no eligible parameters")
    for group in adamw_groups:
        if group.get("_lr_group", "main") == "main":
            group.update({
                "lr": max_lr,
                "_max_lr": max_lr,
                "_min_lr": min_lr,
            })

    muon = _MatrixBatchMuon(
        muon_groups,
        lr=max_lr,
        weight_decay=weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = (
        torch.optim.AdamW(
            adamw_groups, lr=max_lr, weight_decay=weight_decay, fused=True
        )
        if adamw_groups
        else None
    )
    for group in muon.param_groups:
        group["_optimizer_component"] = "muon"
    if adamw is not None:
        for group in adamw.param_groups:
            group["_optimizer_component"] = "adamw"
    return CompositeOptimizer(muon, adamw)


def restore_lr_bounds(optimizer, bounds):
    """Reapply config-owned schedule bounds after loading a checkpoint."""

    for component_name, component in optimizer.optimizers.items():
        for group in component.param_groups:
            name = group.get("_lr_group")
            if name not in bounds:
                raise ValueError(f"Unknown optimizer LR group {name!r}")
            group["_max_lr"], group["_min_lr"] = bounds[name]
            group["_optimizer_component"] = component_name
