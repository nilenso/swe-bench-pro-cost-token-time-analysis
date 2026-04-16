"""
Aggregation helpers for Pi transcript classification results.
"""

from __future__ import annotations

import statistics
from collections import Counter

from .classify import FileResult
from .models import (
    HIGH_LEVEL_LETTER,
    INTENT_DESCRIPTIONS,
    INTENT_DISPLAY_NAMES,
    INTENT_TO_HIGH_LEVEL,
    LETTER_COLORS,
    LETTER_TO_NAME,
    HIGH_LEVEL_COLORS,
    ORDERED_LETTERS,
    PHASES,
    build_model_registry,
)

Results = dict[str, list[FileResult]]


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


def ordered_models(results: Results) -> list[str]:
    return sorted(results.keys(), key=lambda m: (-len(results[m]), m))


def base_intent_frequencies(results: Results) -> dict:
    counts: dict[str, Counter] = {}
    for model, file_results in results.items():
        c = Counter()
        for fr in file_results:
            c.update(fr.base_intent_counts)
        counts[model] = c

    displayed_categories = {
        "read",
        "search",
        "reproduce",
        "edit",
        "verify",
        "git",
        "housekeeping",
        "other",
        "failed",
    }
    intent_set: set[str] = set()
    for intent, high in INTENT_TO_HIGH_LEVEL.items():
        if high in displayed_categories and any(counts[m].get(intent, 0) > 0 for m in counts):
            intent_set.add(intent)

    top_intents = sorted(intent_set, key=lambda i: -sum(counts[m].get(i, 0) for m in counts))
    return {
        "counts": {m: dict(c.most_common()) for m, c in counts.items()},
        "proportions": {m: _to_proportions(c) for m, c in counts.items()},
        "top_intents": top_intents,
    }


def high_level_frequencies(results: Results) -> dict:
    letter_counts: dict[str, Counter] = {}
    per_traj: dict[str, dict[str, float]] = {}

    for model, file_results in results.items():
        lc = Counter()
        for fr in file_results:
            for h, cnt in fr.high_intent_counts.items():
                lc[HIGH_LEVEL_LETTER.get(h, "?")] += cnt
        letter_counts[model] = lc

        n_trajs = len(file_results)
        cat_totals: Counter = Counter()
        for fr in file_results:
            cat_totals.update(fr.high_intent_counts)
        per_traj[model] = {
            cat: round(cat_totals.get(cat, 0) / n_trajs, 1) if n_trajs else 0
            for cat in HIGH_LEVEL_LETTER
        }

    proportions: dict[str, dict[str, float]] = {}
    for model, lc in letter_counts.items():
        prop = _to_proportions(lc)
        proportions[model] = {LETTER_TO_NAME.get(k, k): v for k, v in prop.items()}

    return {
        "counts": {m: dict(c) for m, c in letter_counts.items()},
        "proportions": proportions,
        "per_traj": per_traj,
    }


def phase_frequencies(results: Results) -> dict:
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
            phase_data[phase_name] = phase_steps / total_steps * 100 if total_steps else 0
        out[model] = phase_data
    return out


def verify_outcomes(results: Results) -> dict:
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
    all_labels: set[str] = set()
    counts: dict[str, Counter] = {}
    for model, file_results in results.items():
        c = Counter()
        for fr in file_results:
            c.update(fr.seq_label_counts)
        counts[model] = c
        all_labels.update(c.keys())

    all_labels.discard("")
    all_labels.discard("seq-none")
    sorted_labels = sorted(all_labels, key=lambda l: -sum(counts[m].get(l, 0) for m in counts))
    return {
        "counts": {m: dict(c) for m, c in counts.items()},
        "all_labels": sorted_labels,
    }


def structural_markers(results: Results) -> dict:
    markers = ["first_edit", "last_edit", "first_verify", "first_verify_pass", "submit"]
    out: dict[str, dict[str, dict]] = {}
    for marker in markers:
        out[marker] = {}
        for model, file_results in results.items():
            vals = [fr.positions[marker] for fr in file_results if fr.positions.get(marker) is not None]
            completed_vals = [
                fr.positions[marker]
                for fr in file_results
                if fr.positions.get(marker) is not None and fr.completed
            ]
            incomplete_vals = [
                fr.positions[marker]
                for fr in file_results
                if fr.positions.get(marker) is not None and not fr.completed
            ]
            out[marker][model] = {
                "n": len(vals),
                "median": _median_safe(vals),
                "p25": _p25(vals),
                "p75": _p75(vals),
                "completed_median": _median_safe(completed_vals),
                "incomplete_median": _median_safe(incomplete_vals),
            }
    return out


def per_repo_breakdown(results: Results) -> dict:
    repos: set[str] = set()
    for _, file_results in results.items():
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
            completed = sum(1 for fr in data if fr.completed)
            total_steps = sum(steps)
            verify_steps = sum(fr.high_intent_counts.get("verify", 0) for fr in data)
            edit_steps = sum(fr.high_intent_counts.get("edit", 0) for fr in data)
            out[repo][model] = {
                "n": len(data),
                "avg_steps": round(statistics.mean(steps), 1),
                "completion_rate": round(completed / len(data) * 100, 1),
                "resolve_rate": round(completed / len(data) * 100, 1),
                "verify_pct": round(verify_steps / total_steps * 100, 1) if total_steps else 0,
                "edit_pct": round(edit_steps / total_steps * 100, 1) if total_steps else 0,
                "ve_ratio": round(verify_steps / edit_steps, 2) if edit_steps else 0,
            }
    return out


def step_distribution(results: Results, bin_size: int = 5) -> dict:
    out: dict[str, dict[int, int]] = {}
    for model, file_results in results.items():
        bins: Counter = Counter()
        for fr in file_results:
            b = (fr.steps // bin_size) * bin_size
            bins[b] += 1
        out[model] = dict(sorted(bins.items()))
    return out


def phase_profiles(results: Results, bins: int = 20) -> dict:
    out: dict[str, dict[str, list[float]]] = {}
    for model, file_results in results.items():
        all_profiles: dict[str, list[list[float]]] = {letter: [] for letter in HIGH_LEVEL_LETTER.values()}
        for fr in file_results:
            for letter, profile in fr.phase_profile.items():
                if len(profile) == bins:
                    all_profiles[letter].append(profile)
        avg: dict[str, list[float]] = {}
        for letter in HIGH_LEVEL_LETTER.values():
            profiles = all_profiles[letter]
            if not profiles:
                avg[letter] = [0.0] * bins
            else:
                avg[letter] = [sum(p[b] for p in profiles) / len(profiles) for b in range(bins)]
        out[model] = avg
    return out


def bigram_matrix(results: Results) -> dict:
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
            matrices[model] = [[0.0] * len(ORDERED_LETTERS) for _ in ORDERED_LETTERS]
        else:
            matrix = []
            for fr_letter in ORDERED_LETTERS:
                row = []
                for to_letter in ORDERED_LETTERS:
                    row.append(bc.get(fr_letter + to_letter, 0) / total)
                matrix.append(row)
            matrices[model] = matrix
    return {"matrix": matrices, "letters": ORDERED_LETTERS}


def metadata_summary(results: Results) -> dict:
    out: dict[str, dict] = {}
    for model, file_results in results.items():
        steps = [fr.steps for fr in file_results]
        exits = Counter(fr.exit_status or "unknown" for fr in file_results)
        completed = sum(1 for fr in file_results if fr.completed)
        out[model] = {
            "n": len(file_results),
            "total_steps": sum(steps),
            "avg": round(statistics.mean(steps), 1) if steps else 0,
            "median": _median_safe(steps),
            "p25": _p25(steps),
            "p75": _p75(steps),
            "min": min(steps) if steps else 0,
            "max": max(steps) if steps else 0,
            "completed": completed,
            "completion_rate": _pct(completed, len(file_results)),
            "resolved": completed,
            "resolve_rate": _pct(completed, len(file_results)),
            "exits": dict(exits.most_common()),
        }
    return out


def work_done_vs_completed(results: Results) -> dict:
    out: dict[str, dict] = {}
    for model, file_results in results.items():
        buckets = {
            "wd_completed": 0,
            "wd_incomplete": 0,
            "no_wd_completed": 0,
            "no_wd_incomplete": 0,
        }
        for fr in file_results:
            if fr.work_done and fr.completed:
                buckets["wd_completed"] += 1
            elif fr.work_done and not fr.completed:
                buckets["wd_incomplete"] += 1
            elif not fr.work_done and fr.completed:
                buckets["no_wd_completed"] += 1
            else:
                buckets["no_wd_incomplete"] += 1
        buckets["n"] = len(file_results)
        out[model] = buckets
    return out


def build_analytics_payload(results: Results) -> dict:
    bf = base_intent_frequencies(results)
    hlf = high_level_frequencies(results)
    bg = bigram_matrix(results)
    models = ordered_models(results)
    model_meta = build_model_registry(models)

    median_last_edit = {}
    for model, frs in results.items():
        vals = [fr.positions["last_edit"] for fr in frs if fr.positions.get("last_edit") is not None]
        median_last_edit[model] = round(statistics.median(vals)) if vals else None

    completion_rate = {}
    for model, frs in results.items():
        n = len(frs)
        completion_rate[model] = round(sum(1 for fr in frs if fr.completed) / n * 100, 1) if n else 0.0

    return {
        "models": models,
        "model_display_names": {m: model_meta[m]["label"] for m in models},
        "model_colors": {m: model_meta[m]["color"] for m in models},
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
        "median_last_edit": median_last_edit,
        "completion_rate": completion_rate,
        # Alias retained so copied rendering code can be lightly adapted.
        "resolve_rate": completion_rate,
    }
