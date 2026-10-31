"""Interactive W&B visualizations built from router diagnostics."""

from __future__ import annotations

import colorsys
import math
from typing import TYPE_CHECKING, Protocol

import plotly.graph_objects as go
import wandb

if TYPE_CHECKING:
    from router import FFNRouterPlotData, RouterPlotData

_PLOT_TEMPLATE = "plotly_dark"
_DEFAULT_MAX_LINKS_PER_LAYER = 12


class _RoutePlotData(Protocol):
    n_experts: int
    token_ids: list[int]
    token_paths: list[list[int]]
    transition_counts: list[list[list[int]]]


def _expert_color(expert: int, n_experts: int, alpha: float = 1.0) -> str:
    red, green, blue = colorsys.hsv_to_rgb(
        expert / max(n_experts, 1), 0.68, 0.92
    )
    return (
        f"rgba({round(255 * red)},{round(255 * green)},"
        f"{round(255 * blue)},{alpha})"
    )


def _utilization_heatmap(
    plot_data: _RoutePlotData,
    utilization: list[list[float]],
    title: str,
) -> go.Figure:
    layers = [f"L{layer}" for layer in range(len(utilization))]
    experts = [f"E{expert}" for expert in range(plot_data.n_experts)]
    figure = go.Figure(
        go.Heatmap(
            z=utilization,
            x=experts,
            y=layers,
            colorscale="Turbo",
            zmin=0.0,
            colorbar={"title": "Window share"},
            hovertemplate=(
                "Layer %{y}<br>Expert %{x}<br>Utilization %{z:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Expert",
        yaxis_title="Recurrent layer",
        yaxis={"autorange": "reversed"},
        template=_PLOT_TEMPLATE,
        height=max(460, 26 * len(layers) + 180),
        margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return figure


def _layer_load_balance(
    utilization_series: tuple[tuple[str, list[list[float]]], ...],
    *,
    title: str = "Expert load imbalance by recurrent layer",
    xaxis_title: str = "Recurrent layer",
) -> go.Figure:
    def coefficient_of_variation(values: list[float]) -> float | None:
        mean = sum(values) / len(values)
        if mean == 0.0:
            return None
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return math.sqrt(variance) / mean

    layers = list(range(len(utilization_series[0][1])))
    figure = go.Figure()
    for name, utilization in utilization_series:
        figure.add_trace(
            go.Scatter(
                x=layers,
                y=[
                    coefficient_of_variation(layer)
                    for layer in utilization
                ],
                mode="lines+markers",
                name=name,
                hovertemplate=(
                    "Layer L%{x}<br>Load CV %{y:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Coefficient of variation",
        template=_PLOT_TEMPLATE,
        height=460,
        margin={"l": 70, "r": 40, "t": 70, "b": 60},
    )
    return figure


def _token_path_parallel_coordinates(
    plot_data: _RoutePlotData,
    *,
    title="Sample token routes",
) -> go.Figure:
    experts = [f"E{expert}" for expert in range(plot_data.n_experts)]
    depth = len(plot_data.token_paths[0]) if plot_data.token_paths else 0
    dimensions = [
        {
            "label": f"L{layer}",
            "values": [path[layer] for path in plot_data.token_paths],
            "range": [-0.5, plot_data.n_experts - 0.5],
            "tickvals": list(range(plot_data.n_experts)),
            "ticktext": experts,
        }
        for layer in range(depth)
    ]
    figure = go.Figure(
        go.Parcoords(
            line={
                "color": plot_data.token_ids,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": "Token position"},
            },
            dimensions=dimensions,
            labelfont={"size": 12},
            tickfont={"size": 10},
        )
    )
    figure.update_layout(
        title=title,
        template=_PLOT_TEMPLATE,
        height=620,
        margin={"l": 50, "r": 100, "t": 90, "b": 50},
    )
    return figure


def _dominant_transition_edges(
    plot_data: _RoutePlotData, max_links_per_layer: int
) -> list[tuple[int, int, int, int]]:
    selected = []
    for layer, matrix in enumerate(plot_data.transition_counts):
        edges = [
            (matrix[source][destination], layer, source, destination)
            for source in range(plot_data.n_experts)
            for destination in range(plot_data.n_experts)
            if matrix[source][destination] > 0
        ]
        selected.extend(sorted(edges, reverse=True)[:max_links_per_layer])
    return selected


def _routing_sankey(
    plot_data: _RoutePlotData,
    max_links_per_layer: int,
    *,
    title: str | None = None,
    value_label: str = "traced tokens",
) -> go.Figure:
    selected_edges = _dominant_transition_edges(
        plot_data, max_links_per_layer
    )
    active_nodes = sorted(
        {(layer, source) for _, layer, source, _ in selected_edges}
        | {
            (layer + 1, destination)
            for _, layer, _, destination in selected_edges
        }
    )
    if not active_nodes:
        active_nodes = [
            (0, expert) for expert in range(plot_data.n_experts)
        ]
    node_index = {node: index for index, node in enumerate(active_nodes)}
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            orientation="h",
            node={
                "label": [
                    f"L{layer} · E{expert}"
                    for layer, expert in active_nodes
                ],
                "color": [
                    _expert_color(expert, plot_data.n_experts)
                    for _, expert in active_nodes
                ],
                "pad": 10,
                "thickness": 12,
                "line": {
                    "color": "rgba(255,255,255,0.25)",
                    "width": 0.5,
                },
            },
            link={
                "source": [
                    node_index[(layer, source)]
                    for _, layer, source, _ in selected_edges
                ],
                "target": [
                    node_index[(layer + 1, destination)]
                    for _, layer, _, destination in selected_edges
                ],
                "value": [count for count, _, _, _ in selected_edges],
                "color": [
                    _expert_color(source, plot_data.n_experts, 0.28)
                    for _, _, source, _ in selected_edges
                ],
                "customdata": [
                    f"L{layer}: E{source} → E{destination}"
                    for _, layer, source, destination in selected_edges
                ],
                "hovertemplate": (
                    "%{customdata}<br>%{value} "
                    + value_label
                    + "<extra></extra>"
                ),
            },
        )
    )
    figure.update_layout(
        title=title or (
            "Dominant routing flows ("
            f"top {max_links_per_layer} transitions per layer)"
        ),
        template=_PLOT_TEMPLATE,
        height=max(620, 24 * plot_data.n_experts),
        margin={"l": 30, "r": 30, "t": 90, "b": 30},
        font={"size": 10},
    )
    return figure


def router_plotly_figures(
    plot_data: RouterPlotData,
    *,
    max_links_per_layer: int = _DEFAULT_MAX_LINKS_PER_LAYER,
) -> dict[str, go.Figure]:
    """Build interactive router figures without backing tables."""

    return {
        "layer_expert_utilization": _utilization_heatmap(
            plot_data,
            plot_data.actual_utilization,
            "Expert utilization by recurrent layer",
        ),
        "layer_load_balance": _layer_load_balance(
            (("Actual dispatch", plot_data.actual_utilization),)
        ),
        "token_route_map": _token_path_parallel_coordinates(plot_data),
        "layer_route_flow": _routing_sankey(
            plot_data, max_links_per_layer
        ),
    }


def ffn_router_plotly_figures(
    plot_data: FFNRouterPlotData,
    *,
    max_links_per_layer: int = _DEFAULT_MAX_LINKS_PER_LAYER,
) -> dict[str, go.Figure]:
    """Build the applicable outer-router figures for token-level FFN MoE."""

    assignment = _utilization_heatmap(
        plot_data,
        plot_data.assignment_utilization,
        "FFN-MoE expert utilization by transformer layer",
    )
    importance = _utilization_heatmap(
        plot_data,
        plot_data.importance_share,
        "FFN-MoE gate-weight importance by transformer layer",
    )
    for figure, colorbar_title in (
        (assignment, "Token assignment share"),
        (importance, "Gate-weight share"),
    ):
        figure.update_layout(yaxis_title="Transformer layer")
        figure.data[0].colorbar.title.text = colorbar_title
    figures = {
        "layer_expert_utilization": assignment,
        "layer_expert_importance": importance,
        "layer_load_balance": _layer_load_balance(
            (
                ("Top-k assignments", plot_data.assignment_utilization),
                ("Gate-weight importance", plot_data.importance_share),
            ),
            title="FFN-MoE load imbalance by transformer layer",
            xaxis_title="Transformer layer",
        ),
    }
    if plot_data.token_paths is None:
        # No single-pass token trace: the sites ran once per recurrence.
        return figures
    figures["token_route_map"] = _token_path_parallel_coordinates(
        plot_data,
        title="Sample FFN top-k routes (one line per token and rank)",
    )
    figures["layer_route_flow"] = _routing_sankey(
        plot_data,
        max_links_per_layer,
        title=(
            "Dominant FFN top-k routing flows "
            f"(rank-aligned; top {max_links_per_layer} "
            "transitions per layer)"
        ),
        value_label="token-rank routes",
    )
    return figures


def _wrap_wandb_plots(
    figures: dict[str, go.Figure],
) -> dict[str, wandb.Plotly]:
    return {
        name: wandb.Plotly(figure)
        for name, figure in figures.items()
    }


def ffn_router_wandb_plots(
    plot_data: FFNRouterPlotData,
) -> dict[str, wandb.Plotly]:
    """Wrap token-level FFN-MoE figures as interactive W&B objects."""

    return _wrap_wandb_plots(ffn_router_plotly_figures(plot_data))


def router_wandb_plots(plot_data: RouterPlotData) -> dict[str, wandb.Plotly]:
    """Wrap router figures as interactive W&B Plotly objects."""

    return _wrap_wandb_plots(router_plotly_figures(plot_data))
