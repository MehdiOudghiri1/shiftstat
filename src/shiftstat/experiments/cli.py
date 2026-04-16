"""CLI entry point for config-driven experiments."""

from __future__ import annotations

import argparse

from shiftstat.experiments.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    """Run the experiment CLI."""

    parser = argparse.ArgumentParser(description="Run ShiftStat benchmark experiments.")
    parser.add_argument("config_path", help="Path to a JSON or YAML experiment config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the experiment artifact directory.",
    )
    parser.add_argument(
        "--figure-format",
        default=None,
        help="Optional override for saved figure format, for example png or svg.",
    )
    args = parser.parse_args(argv)

    result = run_experiment(
        args.config_path,
        output_dir=args.output_dir,
        figure_format=args.figure_format,
    )
    if result.manifest_path is not None:
        print(result.manifest_path)
    return 0
