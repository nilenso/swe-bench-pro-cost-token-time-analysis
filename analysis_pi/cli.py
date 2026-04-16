#!/usr/bin/env python3
"""CLI entry point for Pi transcript analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import aggregate, orchestrate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pi transcript analysis pipeline")
    parser.add_argument("--data-root", default="data/pi-mono")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--format", choices=["analytics", "reference", "all"], default="all")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: data root '{data_root}' does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Processing Pi transcripts from {data_root}...", file=sys.stderr)
    results = orchestrate.process_all(data_root, models=args.models, max_workers=args.workers)
    if not results:
        print("No transcript files found.", file=sys.stderr)
        sys.exit(1)

    for model, frs in sorted(results.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"  {model}: {len(frs)} sessions", file=sys.stderr)

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
            "work_done_completed": aggregate.work_done_vs_completed(results),
        }
    if fmt != "all":
        payload = payload[fmt]

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
