#!/usr/bin/env python3
"""Generate the math evaluation LaTeX table from local numerical inputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from plotting.local_data import ASSETS, load_manifest, source_path, provenance

TASKS = (
    ("arc_easy", 2376), ("arc_challenge", 1172),
    ("hellaswag", 10042), ("piqa", 1838), ("gsm8k_cot", 1319),
)


def load_result(filename, data_dir=ASSETS):
    """Read a result inside the selected local dataset directory."""
    try:
        with source_path(data_dir, filename).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read local result {filename}") from error


def build_table(models=None, data_dir=ASSETS):
    """Validate all source results before returning a complete LaTeX table."""
    models = load_manifest(data_dir)["math_eval_table"]["models"] if models is None else models
    rows = []
    for label, lm_file, gsm8k_file in models:
        results = {filename: load_result(filename, data_dir) for filename in (lm_file, gsm8k_file)}
        cells = []
        for task, count in TASKS:
            gsm8k = task == "gsm8k_cot"
            filename = gsm8k_file if gsm8k else lm_file
            data = results[filename]
            metric = "exact_match,flexible-extract" if gsm8k else "acc_norm,none"
            stderr_key = "exact_match_stderr,flexible-extract" if gsm8k else "acc_norm_stderr,none"
            try:
                result = data["results"][task]
                score, stderr = result[metric], result[stderr_key]
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1
                    for value in (score, stderr)
                ):
                    raise ValueError("score and standard error must be finite numbers in [0, 1]")
                shots = 3 if gsm8k else 5
                if data["n-shot"][task] != shots:
                    raise ValueError(f"expected {shots}-shot evaluation")
                coverage = data["n-samples"][task]
                if (result["sample_len"] != count or coverage["original"] != count
                        or coverage["effective"] != count):
                    raise ValueError(f"expected all {count} benchmark examples")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {task} result in {filename}: {error}"
                ) from error
            cells.append(f"${100 * score:.1f} \\pm {100 * stderr:.1f}$")
        rows.append(label + " & " + " & ".join(cells) + r" \\")

    # Theoretical chance levels, rounded to one decimal; not empirical results.
    return (
        "% Source: " + json.dumps(provenance(data_dir, [filename for _, lm, gsm in models for filename in (lm, gsm)]), sort_keys=True) + "\n"
        + r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lccccc}
\toprule
Model & ARC-E & ARC-C & HellaSwag & PIQA & GSM8K \\
\midrule
Random choice & 25.0 & 25.0 & 25.0 & 50.0 & 0.0 \\
\midrule
"""
        + "\n".join(rows)
        + "\n"
        + r"""\bottomrule
\end{tabular}
\caption{}
\label{tab:math}
\end{table}
"""
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write LaTeX to this path instead of stdout")
    parser.add_argument("--data-dir", type=Path, default=ASSETS)
    args = parser.parse_args(argv)
    try:
        models = load_manifest(args.data_dir)["math_eval_table"]["models"]
        appendix = build_table(models, args.data_dir).split("\n", 1)[1].replace(
            r"\label{tab:math}", r"\label{app-tab:math}")
        table = build_table(models[:3], args.data_dir) + "\n" + appendix
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    if args.output is None:
        print(table, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
