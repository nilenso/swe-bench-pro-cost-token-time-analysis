"""
Layer 2: Thin wrapper around scripts/classify_intent.py.

Exposes a single `classify_file()` entry point that returns a structured
FileResult dataclass. All classification logic lives in classify_intent.py --
this module only marshals data.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure scripts/ is importable
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import classify_intent as ci  # noqa: E402

from .models import HIGH_LEVEL_LETTER  # noqa: E402

# Re-export the sets that downstream code may want for filtering
SOURCE_EDIT_INTENTS = ci.SOURCE_EDIT_INTENTS
SEQUENCE_VERIFY_INTENTS = ci.SEQUENCE_VERIFY_INTENTS


@dataclass
class FileResult:
    """Everything computed from a single .traj file."""

    # Identity
    model: str
    path: str
    instance_id: str
    repo: str

    # Classification outputs
    base_intents: list[str] = field(default_factory=list)
    high_intents: list[str] = field(default_factory=list)
    high_letters: list[str] = field(default_factory=list)
    verify_outcomes: list[str] = field(default_factory=list)
    seq_labels: list[str] = field(default_factory=list)

    # Aggregated counts (per-file)
    base_intent_counts: dict[str, int] = field(default_factory=dict)
    high_intent_counts: dict[str, int] = field(default_factory=dict)
    verify_outcome_counts: dict[str, int] = field(default_factory=dict)
    seq_label_counts: dict[str, int] = field(default_factory=dict)

    # Structural markers (as % of trajectory, None if not found)
    positions: dict[str, float | None] = field(default_factory=dict)

    # Metadata from the traj file
    steps: int = 0
    exit_status: str = ""
    submitted: bool = False   # agent produced a submission (from traj metadata)
    resolved: bool = False    # patch actually fixes the tests (from eval _output.json)
    work_done: bool = False

    # Phase profile (letter -> list of bin proportions)
    phase_profile: dict[str, list[float]] = field(default_factory=dict)

    # High-level sequence string (e.g. "RRSSEEVV")
    high_seq: str = ""

    # Bigram counts
    bigram_counts: dict[str, int] = field(default_factory=dict)


def _parse_repo(instance_id: str) -> str:
    """Extract org/repo from an instance_id string."""
    core = instance_id
    if core.startswith("instance_"):
        core = core[len("instance_"):]
    if "__" not in core:
        return "unknown/unknown"
    org, rest = core.split("__", 1)
    m = re.search(r"-(?:[0-9a-f]{7,40})(?:-v.*)?$", rest)
    repo = rest[:m.start()] if m else rest
    return f"{org}/{repo}"


def _load_resolution_map() -> dict[str, dict[str, bool]]:
    """Load resolution data from the benchmark CSV.

    Returns {model_dir_name: {instance_id: resolved}}.
    The CSV has the authoritative resolution status from the benchmark
    platform, which applies the patch and runs tests. The eval/ directories
    in the data folder are NOT reliable for this (they may test the baseline
    code when no patch was submitted).
    """
    csv_path = Path(__file__).parent.parent / "data" / "agent_runs_data.csv"
    if not csv_path.exists():
        return {}

    import csv
    # Map CSV model names to our directory names
    csv_to_dir = {
        "Claude 4.5 Sonnet - 10132025": "claude45",
        "GPT-5 - 10132025": "gpt5",
        "GLM-4.5 -- 10222025": "glm45",
        # Gemini: no matching CSV entry for our data
    }

    result: dict[str, dict[str, bool]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            dir_name = csv_to_dir.get(row["metadata.model_name"])
            if dir_name:
                result.setdefault(dir_name, {})[row["metadata.instance_id"]] = (
                    row["metadata.resolved"] == "true"
                )
    return result


# Loaded once at import time
_RESOLUTION_MAP: dict[str, dict[str, bool]] = _load_resolution_map()


def classify_file(model: str, path: str, phase_bins: int = 20) -> FileResult | None:
    """Classify a single .traj file. Returns None for empty trajectories."""
    data = ci._load_json(path)
    trajectory = data.get("trajectory", [])
    info = data.get("info", {})
    instance_id = Path(path).stem

    if not trajectory:
        return None

    # --- Run all classification layers ---
    base_intents = ci.classify_trajectory(trajectory)
    hierarchical = ci.classify_hierarchical_layer(base_intents)
    verify_outcomes = ci.classify_verify_outcomes(trajectory, base_intents)
    seq_labels = ci.classify_sequence_layer(trajectory, base_intents, verify_outcomes)

    # --- Derived fields ---
    highs = [h.split(".", 1)[0] for h in hierarchical]
    high_letters = [HIGH_LEVEL_LETTER.get(h, "?") for h in highs]
    high_seq = "".join(high_letters)

    n = len(base_intents)

    # Counts
    base_intent_counts = dict(Counter(base_intents))
    high_intent_counts = dict(Counter(highs))
    verify_outcome_counts = dict(Counter(verify_outcomes))
    seq_label_counts = dict(Counter(seq_labels))

    # Bigrams
    bigram_c: Counter = Counter()
    for i in range(len(high_letters) - 1):
        bigram_c[high_letters[i] + high_letters[i + 1]] += 1

    # Structural positions (as % of trajectory)
    def pct(idx: int) -> float | None:
        return round(idx / max(n - 1, 1) * 100, 1) if idx >= 0 else None

    first_edit = next((i for i, b in enumerate(base_intents) if b in ci.SOURCE_EDIT_INTENTS), -1)
    last_edit = next((i for i in range(n - 1, -1, -1) if base_intents[i] in ci.SOURCE_EDIT_INTENTS), -1)
    first_verify = next((i for i, b in enumerate(base_intents) if b in ci.SEQUENCE_VERIFY_INTENTS), -1)
    first_verify_pass = next((i for i in range(n) if verify_outcomes[i] == "pass"), -1)
    submit_idx = next((i for i, b in enumerate(base_intents) if b == "submit"), -1)

    positions = {
        "first_edit": pct(first_edit),
        "last_edit": pct(last_edit),
        "first_verify": pct(first_verify),
        "first_verify_pass": pct(first_verify_pass),
        "submit": pct(submit_idx),
    }

    # Submission: did the agent produce a patch?
    submitted = (info.get("exit_status") or "").startswith("submitted") or bool(info.get("submission"))

    # Resolution: did the patch actually fix the tests?
    # From the benchmark CSV (authoritative), not from eval/ dirs on disk.
    resolved = _RESOLUTION_MAP.get(model, {}).get(instance_id, False)

    # Phase profile: binned activity proportions across trajectory
    phase_profile: dict[str, list[float]] = {}
    if n >= 5:
        for letter in HIGH_LEVEL_LETTER.values():
            counts_in_bin = []
            for b in range(phase_bins):
                start = int(b * n / phase_bins)
                end = int((b + 1) * n / phase_bins)
                segment = high_letters[start:end]
                counts_in_bin.append(
                    segment.count(letter) / len(segment) if segment else 0.0
                )
            phase_profile[letter] = counts_in_bin

    return FileResult(
        model=model,
        path=path,
        instance_id=instance_id,
        repo=_parse_repo(instance_id),
        base_intents=base_intents,
        high_intents=highs,
        high_letters=high_letters,
        verify_outcomes=verify_outcomes,
        seq_labels=seq_labels,
        base_intent_counts=base_intent_counts,
        high_intent_counts=high_intent_counts,
        verify_outcome_counts=verify_outcome_counts,
        seq_label_counts=seq_label_counts,
        positions=positions,
        steps=n,
        exit_status=info.get("exit_status", "") or "",
        submitted=submitted,
        resolved=resolved,
        work_done="seq-first-all-pass" in seq_labels or "seq-work-done" in seq_labels,
        phase_profile=phase_profile,
        high_seq=high_seq,
        bigram_counts=dict(bigram_c),
    )
