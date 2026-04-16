"""
Layer 5: Metrics computation from per-file results.

Pure functions that take {model: [FileResult, ...]} and return structured
data dicts. No hardcoded model names -- everything iterates over whatever
models appear in the results.
"""

from __future__ import annotations

import statistics
from collections import Counter

from .classify import FileResult
from .models import (
    HIGH_LEVEL_LETTER,
    INTENT_TO_HIGH_LEVEL,
    LETTER_TO_NAME,
    ORDERED_LETTERS,
    PHASES,
)


# ---------------------------------------------------------------------------
# Type alias for the standard results shape
# ---------------------------------------------------------------------------
Results = dict[str, list[FileResult]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_proportions(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total > 0 else "0%"


def _median_safe(vals: list) -> float | None:
    return round(statistics.median(vals), 1) if vals else None


def _p25(vals: list) -> float | None:
    return round(sorted(vals)[len(vals) // 4], 1) if vals else None


def _p75(vals: list) -> float | None:
    return round(sorted(vals)[len(vals) * 3 // 4], 1) if vals else None


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def base_intent_frequencies(results: Results) -> dict:
    """Per-model base intent counts and proportions.

    Returns:
        {
            "counts": {model: {intent: count, ...}, ...},
            "proportions": {model: {intent: proportion, ...}, ...},
            "top_intents": [intent, ...],  # sorted by total count desc
        }
    """
    counts: dict[str, Counter] = {}
    for model, file_results in results.items():
        c = Counter()
        for fr in file_results:
            c.update(fr.base_intent_counts)
        counts[model] = c

    # Find all intents that appear in any model, excluding failed category
    displayed_categories = {"read", "search", "reproduce", "edit", "verify",
                            "git", "housekeeping", "other"}
    intent_set: set[str] = set()
    for intent, high in INTENT_TO_HIGH_LEVEL.items():
        if high in displayed_categories:
            if any(counts[m].get(intent, 0) > 0 for m in counts):
                intent_set.add(intent)

    # Sort by total count across all models
    top_intents = sorted(
        intent_set,
        key=lambda i: -sum(counts[m].get(i, 0) for m in counts),
    )

    return {
        "counts": {m: dict(c.most_common()) for m, c in counts.items()},
        "proportions": {m: _to_proportions(c) for m, c in counts.items()},
        "top_intents": top_intents,
    }


def high_level_frequencies(results: Results) -> dict:
    """Per-model high-level category counts (letter-based) and proportions.

    Returns:
        {
            "counts": {model: {letter: count, ...}, ...},
            "proportions": {model: {name: proportion, ...}, ...},
            "per_traj": {model: {category: avg_per_traj, ...}, ...},
        }
    """
    letter_counts: dict[str, Counter] = {}
    per_traj: dict[str, dict[str, float]] = {}

    for model, file_results in results.items():
        lc = Counter()
        for fr in file_results:
            for h, cnt in fr.high_intent_counts.items():
                letter = HIGH_LEVEL_LETTER.get(h, "?")
                lc[letter] += cnt
        letter_counts[model] = lc

        # Per-category averages
        n_trajs = len(file_results)
        cat_totals: Counter = Counter()
        for fr in file_results:
            cat_totals.update(fr.high_intent_counts)
        per_traj[model] = {
            cat: round(cat_totals.get(cat, 0) / n_trajs, 1) if n_trajs else 0
            for cat in HIGH_LEVEL_LETTER
        }

    # Name-based proportions
    proportions: dict[str, dict[str, float]] = {}
    for model, lc in letter_counts.items():
        prop = _to_proportions(lc)
        proportions[model] = {
            LETTER_TO_NAME.get(k, k): v for k, v in prop.items()
        }

    return {
        "counts": {m: dict(c) for m, c in letter_counts.items()},
        "proportions": proportions,
        "per_traj": per_traj,
    }


def phase_frequencies(results: Results) -> dict:
    """Per-model phase proportions.

    Returns:
        {model: {phase_name: pct, ...}, ...}
    """
    out: dict[str, dict[str, float]] = {}
    for model, file_results in results.items():
        total_steps = sum(fr.steps for fr in file_results)
        phase_data: dict[str, float] = {}
        for phase_name, phase_def in PHASES.items():
            phase_steps = sum(
                fr.high_intent_counts.get(cat, 0)
                for fr in file_results
                for cat in phase_def["categories"]
            )
            phase_data[phase_name] = (
                phase_steps / total_steps * 100 if total_steps else 0
            )
        out[model] = phase_data
    return out


def verify_outcomes(results: Results) -> dict:
    """Per-model verify outcome breakdown.

    Returns:
        {model: {"pass": n, "fail": n, "unknown": n, "total": n, "pass_rate": str}, ...}
    """
    out: dict[str, dict] = {}
    for model, file_results in results.items():
        c = Counter()
        for fr in file_results:
            c.update(fr.verify_outcome_counts)
        total = c.get("pass", 0) + c.get("fail", 0) + c.get("", 0)
        detectable = c.get("pass", 0) + c.get("fail", 0)
        out[model] = {
            "pass": c.get("pass", 0),
            "fail": c.get("fail", 0),
            "unknown": c.get("", 0),
            "total": total,
            "pass_rate": _pct(c.get("pass", 0), detectable) if detectable else "n/a",
        }
    return out


def sequence_labels(results: Results) -> dict:
    """Per-model sequence label counts (excluding empty/seq-none).

    Returns:
        {
            "counts": {model: {label: count, ...}, ...},
            "all_labels": [label, ...],
        }
    """
    all_labels: set[str] = set()
    counts: dict[str, Counter] = {}
    for model, file_results in results.items():
        c = Counter()
        for fr in file_results:
            c.update(fr.seq_label_counts)
        counts[model] = c
        all_labels.update(c.keys())

    # Remove empty and seq-none
    all_labels.discard("")
    all_labels.discard("seq-none")

    sorted_labels = sorted(
        all_labels,
        key=lambda l: -sum(counts[m].get(l, 0) for m in counts),
    )

    return {
        "counts": {m: dict(c) for m, c in counts.items()},
        "all_labels": sorted_labels,
    }


def structural_markers(results: Results) -> dict:
    """Per-model structural marker positions (as % of trajectory).

    Returns:
        {
            marker: {
                model: {"n": n, "median": m, "p25": p, "p75": p,
                        "resolved_median": m, "unresolved_median": m},
                ...
            },
            ...
        }
    """
    markers = ["first_edit", "last_edit", "first_verify", "first_verify_pass", "submit"]
    out: dict[str, dict[str, dict]] = {}

    for marker in markers:
        out[marker] = {}
        for model, file_results in results.items():
            vals = [
                fr.positions[marker]
                for fr in file_results
                if fr.positions.get(marker) is not None
            ]
            resolved_vals = [
                fr.positions[marker]
                for fr in file_results
                if fr.positions.get(marker) is not None and fr.resolved
            ]
            unresolved_vals = [
                fr.positions[marker]
                for fr in file_results
                if fr.positions.get(marker) is not None and not fr.resolved
            ]
            out[marker][model] = {
                "n": len(vals),
                "median": _median_safe(vals),
                "p25": _p25(vals),
                "p75": _p75(vals),
                "resolved_median": _median_safe(resolved_vals),
                "unresolved_median": _median_safe(unresolved_vals),
            }

    return out


def per_repo_breakdown(results: Results) -> dict:
    """Per-repo stats for each model.

    Returns:
        {
            repo: {
                model: {"n": n, "avg_steps": f, "resolve_rate": f,
                        "verify_pct": f, "edit_pct": f, "ve_ratio": f},
                ...
            },
            ...
        }
    """
    # Collect all repos
    repos: set[str] = set()
    for model, file_results in results.items():
        for fr in file_results:
            repos.add(fr.repo)

    out: dict[str, dict[str, dict]] = {}
    for repo in sorted(repos):
        out[repo] = {}
        for model, file_results in results.items():
            data = [fr for fr in file_results if fr.repo == repo]
            if not data:
                continue
            steps = [fr.steps for fr in data]
            resolved = sum(1 for fr in data if fr.resolved)
            total_steps = sum(steps)
            verify_steps = sum(fr.high_intent_counts.get("verify", 0) for fr in data)
            edit_steps = sum(fr.high_intent_counts.get("edit", 0) for fr in data)
            out[repo][model] = {
                "n": len(data),
                "avg_steps": round(statistics.mean(steps), 1),
                "resolve_rate": round(resolved / len(data) * 100, 1),
                "verify_pct": round(verify_steps / total_steps * 100, 1) if total_steps else 0,
                "edit_pct": round(edit_steps / total_steps * 100, 1) if total_steps else 0,
                "ve_ratio": round(verify_steps / edit_steps, 2) if edit_steps else 0,
            }

    return out


def step_distribution(results: Results, bin_size: int = 5) -> dict:
    """Per-model step count distribution (binned).

    Returns:
        {model: {bin_start: count, ...}, ...}
    """
    out: dict[str, dict[int, int]] = {}
    for model, file_results in results.items():
        bins: Counter = Counter()
        for fr in file_results:
            b = (fr.steps // bin_size) * bin_size
            bins[b] += 1
        out[model] = dict(sorted(bins.items()))
    return out


def phase_profiles(results: Results, bins: int = 20) -> dict:
    """Per-model average phase profiles across trajectory lifetime.

    Returns:
        {model: {letter: [bin_avg, ...], ...}, ...}
    """
    out: dict[str, dict[str, list[float]]] = {}
    for model, file_results in results.items():
        # Collect all per-file profiles
        all_profiles: dict[str, list[list[float]]] = {
            letter: [] for letter in HIGH_LEVEL_LETTER.values()
        }
        for fr in file_results:
            for letter, profile in fr.phase_profile.items():
                if len(profile) == bins:
                    all_profiles[letter].append(profile)

        # Average across trajectories
        avg: dict[str, list[float]] = {}
        for letter in HIGH_LEVEL_LETTER.values():
            profiles = all_profiles[letter]
            if not profiles:
                avg[letter] = [0.0] * bins
            else:
                avg[letter] = [
                    sum(p[b] for p in profiles) / len(profiles)
                    for b in range(bins)
                ]
        out[model] = avg

    return out


def bigram_matrix(results: Results) -> dict:
    """Per-model transition matrices (from-letter -> to-letter).

    Returns:
        {
            "matrix": {model: [[float, ...], ...], ...},
            "letters": ["R", "S", ...],
        }
    """
    bigram_counts: dict[str, Counter] = {}
    for model, file_results in results.items():
        bc = Counter()
        for fr in file_results:
            bc.update(fr.bigram_counts)
        bigram_counts[model] = bc

    matrices: dict[str, list[list[float]]] = {}
    for model, bc in bigram_counts.items():
        total = sum(bc.values())
        if total == 0:
            matrices[model] = [
                [0.0] * len(ORDERED_LETTERS) for _ in ORDERED_LETTERS
            ]
        else:
            matrix = []
            for fr_letter in ORDERED_LETTERS:
                row = []
                for to_letter in ORDERED_LETTERS:
                    row.append(bc.get(fr_letter + to_letter, 0) / total)
                matrix.append(row)
            matrices[model] = matrix

    return {
        "matrix": matrices,
        "letters": ORDERED_LETTERS,
    }


def metadata_summary(results: Results) -> dict:
    """Per-model trajectory metadata summary.

    Returns:
        {model: {"n": n, "total_steps": n, "avg": f, "median": f,
                 "p25": f, "p75": f, "min": n, "max": n,
                 "resolved": n, "resolve_rate": str,
                 "exits": {status: count, ...}}, ...}
    """
    out: dict[str, dict] = {}
    for model, file_results in results.items():
        steps = [fr.steps for fr in file_results]
        exits = Counter(fr.exit_status for fr in file_results)
        resolved = sum(1 for fr in file_results if fr.resolved)
        out[model] = {
            "n": len(file_results),
            "total_steps": sum(steps),
            "avg": round(statistics.mean(steps), 1) if steps else 0,
            "median": _median_safe(steps),
            "p25": _p25(steps),
            "p75": _p75(steps),
            "min": min(steps) if steps else 0,
            "max": max(steps) if steps else 0,
            "resolved": resolved,
            "resolve_rate": _pct(resolved, len(file_results)),
            "exits": dict(exits.most_common()),
        }
    return out


def work_done_vs_resolved(results: Results) -> dict:
    """Per-model work-done vs resolved confusion matrix.

    Returns:
        {model: {"wd_resolved": n, "wd_unresolved": n,
                 "no_wd_resolved": n, "no_wd_unresolved": n, "n": n}, ...}
    """
    out: dict[str, dict] = {}
    for model, file_results in results.items():
        buckets = {
            "wd_resolved": 0, "wd_unresolved": 0,
            "no_wd_resolved": 0, "no_wd_unresolved": 0,
        }
        for fr in file_results:
            if fr.work_done and fr.resolved:
                buckets["wd_resolved"] += 1
            elif fr.work_done and not fr.resolved:
                buckets["wd_unresolved"] += 1
            elif not fr.work_done and fr.resolved:
                buckets["no_wd_resolved"] += 1
            else:
                buckets["no_wd_unresolved"] += 1
        buckets["n"] = len(file_results)
        out[model] = buckets
    return out


# ---------------------------------------------------------------------------
# Convenience: build the full analytics payload (replaces build_analytics.build_payload)
# ---------------------------------------------------------------------------

def build_analytics_payload(results: Results) -> dict:
    """Build the complete data payload that the analytics HTML template needs.

    This replaces build_analytics.build_payload() but works with N models.
    """
    from .models import (
        HIGH_LEVEL_COLORS,
        HIGH_LEVEL_LETTER,
        INTENT_DESCRIPTIONS,
        INTENT_DISPLAY_NAMES,
        INTENT_TO_HIGH_LEVEL,
        LETTER_COLORS,
        LETTER_TO_NAME,
        MODELS,
    )

    bf = base_intent_frequencies(results)
    hlf = high_level_frequencies(results)
    bg = bigram_matrix(results)

    # Only include models that are in the results
    active_models = [m for m in MODELS if m in results]

    return {
        "models": active_models,
        "model_display_names": {m: MODELS[m]["label"] for m in active_models},
        "model_colors": {m: MODELS[m]["color"] for m in active_models},
        "high_counts": hlf["counts"],
        "high_proportions": hlf["proportions"],
        "low_proportions": bf["proportions"],
        "top_low_intents": bf["top_intents"],
        "bigram_matrix": bg["matrix"],
        "bigram_letters": bg["letters"],
        "avg_phase": phase_profiles(results),
        "step_dist": step_distribution(results),
        "num_trajs": {m: len(frs) for m, frs in results.items()},
        "high_level_letter": HIGH_LEVEL_LETTER,
        "letter_to_name": LETTER_TO_NAME,
        "name_to_letter": HIGH_LEVEL_LETTER,
        "letter_colors": LETTER_COLORS,
        "name_colors": HIGH_LEVEL_COLORS,
        "intent_to_category": dict(INTENT_TO_HIGH_LEVEL),
        "intent_descriptions": INTENT_DESCRIPTIONS,
        "intent_display_names": INTENT_DISPLAY_NAMES,
    }
