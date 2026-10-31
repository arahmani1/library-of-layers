#!/usr/bin/env python3
"""Plot math-training loss with routing and a separate five-model comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

if __package__:
    from . import loss_history, loss_router_plot
    from .loss_router_plot import PAPER_RC, load_router_heatmap
    from .local_data import ASSETS, load_manifest, source_path, provenance
else:
    import loss_history
    import loss_router_plot
    from loss_router_plot import PAPER_RC, load_router_heatmap
    from local_data import ASSETS, load_manifest, source_path, provenance

X_KEY = loss_history.X_KEY
Y_KEY = loss_history.Y_KEY
running_average = loss_history.running_average
TOKENS_PER_STEP = 8 * 16 * 8 * 1024


def read_accuracy(path: Path) -> tuple[float, float]:
    """Read pinned GSM8K flexible-extract accuracy and standard error."""

    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    try:
        task = data["results"]["gsm8k_cot"]
        accuracy = float(task["exact_match,flexible-extract"])
        stderr = float(task["exact_match_stderr,flexible-extract"])
        samples = int(task["sample_len"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{path.name} is not a full GSM8K flexible-extract result"
        ) from error
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"invalid flexible-extract accuracy {accuracy}")
    if not 0.0 <= stderr <= 1.0:
        raise ValueError(f"invalid flexible-extract standard error {stderr}")
    if samples != 1319:
        raise ValueError(f"expected 1319 GSM8K samples, found {samples}")
    return accuracy, stderr


def load_accuracy(specification, data_dir=ASSETS):
    return read_accuracy(source_path(data_dir, specification["result_file"]))


def load_curve(specification, data_dir=ASSETS):
    return loss_history.load_curve(specification, data_dir)


def make_figure(curves, smoothing_window: int, router_heatmap=None):
    """Prepare math curves for the shared loss-and-experts renderer."""

    prepared = []
    for curve in curves:
        prepared.append({
            **curve,
            "raw_y": curve["y"],
            "y": running_average(curve["y"], smoothing_window),
        })
    return loss_router_plot.make_figure(
        prepared, {**router_heatmap, "label": ""} if router_heatmap is not None else None,
        legend_ncol=3, legend_fontsize=12.5,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window", type=int, default=200,
        help="trailing running-average window in steps (default: 200)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".pdf"),
        help="output PDF (default: plotting/math_flop_loss.pdf)",
    )
    parser.add_argument(
        "--comparison-output", type=Path,
        default=Path(__file__).with_name("math_flop_loss_comparison.pdf"),
        help="five-model loss-only PDF (default: plotting/math_flop_loss_comparison.pdf)",
    )
    parser.add_argument("--data-dir", type=Path, default=ASSETS)
    args = parser.parse_args(argv)
    if args.window <= 0:
        parser.error("--window must be positive")
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    if args.comparison_output.suffix.lower() != ".pdf":
        parser.error("--comparison-output must end in .pdf")
    if args.output.resolve() == args.comparison_output.resolve():
        parser.error("--output and --comparison-output must be different paths")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_manifest(args.data_dir)["math_flop_loss"]
    router_specification = config["router"]
    curves = []
    for specification in config["runs"]:
        curve = load_curve(specification, args.data_dir)
        if "result_file" in specification:
            curve["accuracy"], curve["stderr"] = load_accuracy(specification, args.data_dir)
        curves.append(curve)
    router_heatmap = load_router_heatmap(router_specification, args.data_dir)
    router_source = json.dumps(provenance(args.data_dir, [router_specification["file"]]), sort_keys=True)
    for output, selected, heatmap in (
        (args.output, [curve for curve in curves if curve["id"] in config["main"]], router_heatmap),
        (args.comparison_output, curves, None),
    ):
        figure = make_figure(selected, args.window, heatmap)
        output.parent.mkdir(parents=True, exist_ok=True)
        sources = json.dumps(provenance(args.data_dir, [filename for curve in selected for filename in (curve["file"], curve.get("result_file")) if filename]), sort_keys=True)
        router_metadata = (
            f"; learned router shares exclude mandatory routes; router source {router_source}"
            if heatmap is not None else ""
        )
        with plt.rc_context(PAPER_RC):
            figure.savefig(
                output, format="pdf", bbox_inches="tight", transparent=True,
                metadata={
                    "Title": (
                        "Math training loss and learned-router utilization"
                        if heatmap is not None else "Math training loss comparison"
                    ),
                    "Subject": (
                        f"{Y_KEY} against {X_KEY}; trailing running average over "
                        f"{args.window} steps; complete local histories in "
                        f"CSV; GSM8K flexible-extract accuracy ± standard error "
                        f"from local result JSON; "
                        f"loss sources {sources}{router_metadata}"
                    ),
                },
            )
        plt.close(figure)
        summaries = ", ".join(
            (
                f"{curve['id']}: {len(curve['x'])} points, "
                f"{curve['steps'] * TOKENS_PER_STEP} tokens"
                + (
                    f", {curve['accuracy']:.6f} ± {curve['stderr']:.6f} "
                    "flexible-extract (SE)" if "accuracy" in curve else ""
                )
            )
            for curve in selected
        )
        print(
            f"Wrote {output.name} from {sources} ({summaries}; "
            f"trailing running-average window {args.window}){router_metadata}"
        )


if __name__ == "__main__":
    main()
