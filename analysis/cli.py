#!/usr/bin/env python3
"""
CLI entry point for the analysis pipeline.

Usage:
  python -m analysis.cli --data-root data --format analytics --output payload.json
  python -m analysis.cli --data-root data --format reference --output reference.json
  python -m analysis.cli --data-root data --models claude45 gpt5 --format analytics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import aggregate, orchestrate
from .models import MODELS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SWE-Bench Pro trajectory analysis pipeline",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing model subdirectories (default: data)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to process (default: all registered: {', '.join(MODELS)})",
    )
    parser.add_argument(
        "--format",
        choices=["analytics", "reference", "all"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: auto)",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: data root '{data_root}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Validate model names
    if args.models:
        unknown = set(args.models) - set(MODELS)
        if unknown:
            print(f"Error: unknown models: {', '.join(unknown)}", file=sys.stderr)
            print(f"Known models: {', '.join(MODELS)}", file=sys.stderr)
            sys.exit(1)

    print(f"Processing trajectories from {data_root}...", file=sys.stderr)
    results = orchestrate.process_all(
        data_root, models=args.models, max_workers=args.workers,
    )

    if not results:
        print("No trajectory files found.", file=sys.stderr)
        sys.exit(1)

    for model, frs in sorted(results.items()):
        print(f"  {model}: {len(frs)} trajectories", file=sys.stderr)

    # Build requested payload
    fmt = args.format
    payload: dict = {}

    if fmt in ("analytics", "all"):
        payload["analytics"] = aggregate.build_analytics_payload(results)

    if fmt in ("reference", "all"):
        payload["reference"] = {
            "metadata": aggregate.metadata_summary(results),
            "base_intents": aggregate.base_intent_frequencies(results),
            "high_level": aggregate.high_level_frequencies(results),
            "phases": aggregate.phase_frequencies(results),
            "verify_outcomes": aggregate.verify_outcomes(results),
            "sequence_labels": aggregate.sequence_labels(results),
            "structural_markers": aggregate.structural_markers(results),
            "per_repo": aggregate.per_repo_breakdown(results),
            "step_distribution": aggregate.step_distribution(results),
            "work_done_resolved": aggregate.work_done_vs_resolved(results),
        }

    # If only one format was requested, unwrap the outer key
    if fmt != "all":
        payload = payload[fmt]

    # Output
    output_json = json.dumps(payload, indent=2, default=str)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json)
        print(f"Wrote {out_path}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
