#!/usr/bin/env python3
"""
Build an HTML analytics page comparing model trajectories.

Charts:
  1. High-level letter frequencies
  2. Low-level intent frequencies (issue-only sessions)
  3. Low-level intent frequencies (all strict single-model sessions)
  4. Steps per trajectory
  5. Typical trajectory shape visualisations

Usage:
  python scripts/build_pi_analytics.py --data-root data/pi-mono --output docs/pi-analytics.html
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# Add project root to path for analysis package
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.orchestrate import process_all as process_benchmark_all
from analysis.aggregate import build_analytics_payload as build_benchmark_payload, phase_profiles as benchmark_phase_profiles
from analysis_pi.orchestrate import process_all
from analysis_pi.aggregate import build_analytics_payload, phase_profiles as pi_phase_profiles
from analysis_pi.session_filter import DEFAULT_EXACT_MODELS, SessionFilter, collect_filtered_paths
from analysis_pi.user_messages import analyze_user_messages


BENCHMARK_PAIR_FOR_PI_MODEL = {
    "claude-opus-4-5": "claude45",
    "claude-opus-4-6": "claude45",
    "gpt-5.2-codex": "gpt5",
    "gpt-5.3-codex": "gpt5",
    "gpt-5.4": "gpt5",
}

INTERVENTION_MACROS = [
    {
        "key": "authorization",
        "members": ["authorize_work"],
        "label": "authorization",
        "symbol": "△",
        "color": "#4a8a5a",
    },
    {
        "key": "steering",
        "members": ["solution_steer", "evidence_or_repro", "qa_or_critique", "validation_request"],
        "label": "steering",
        "symbol": "○",
        "color": "#b56a50",
    },
    {
        "key": "closeout",
        "members": ["workflow_closeout"],
        "label": "closeout",
        "symbol": "□",
        "color": "#8a6a9a",
    },
]

FIRST_EDIT_MARKER = {
    "key": "first_edit",
    "label": "first edit",
    "symbol": "◆",
    "color": "#4b5563",
}

LAST_EDIT_MARKER = {
    "key": "last_edit",
    "label": "last edit",
    "symbol": "◇",
    "color": "#6b7280",
}


def _filter_results_to_paths(results: dict[str, list], allowed_paths: dict[str, set[str]]) -> dict[str, list]:
    out: dict[str, list] = {}
    for model, rows in results.items():
        keep = allowed_paths.get(model)
        if not keep:
            continue
        filtered = [row for row in rows if row.path in keep]
        if filtered:
            out[model] = filtered
    return out


def _summarize_positions(vals: list[float]) -> dict[str, float | None]:
    vals = sorted(vals)
    if not vals:
        return {"median": None, "p25": None, "p75": None}
    if len(vals) >= 2:
        p25 = round(statistics.quantiles(vals, n=4)[0], 1)
        p75 = round(statistics.quantiles(vals, n=4)[2], 1)
    else:
        p25 = p75 = round(vals[0], 1)
    return {
        "median": round(statistics.median(vals), 1),
        "p25": p25,
        "p75": p75,
    }


def _compute_intervention_markers(allowed_paths: dict[str, set[str]], models: list[str]) -> dict[str, dict]:
    user_data = analyze_user_messages(allowed_paths)
    per_model = user_data.get("per_model", {})
    markers: dict[str, dict] = {}

    def compute_rows(classes: dict[str, dict], num_sessions: int) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        for macro in INTERVENTION_MACROS:
            recs = []
            for label in macro["members"]:
                recs.extend(classes.get(label, {}).get("messages", []))
            firsts_by_path: dict[str, float] = {}
            for rec in sorted(recs, key=lambda r: (r["path"], r["message_index"], r["progress_pct"])):
                firsts_by_path.setdefault(rec["path"], rec["progress_pct"])
            summary = _summarize_positions(list(firsts_by_path.values()))
            rows[macro["key"]] = {
                "label": macro["label"],
                "symbol": macro["symbol"],
                "color": macro["color"],
                "median": summary["median"],
                "p25": summary["p25"],
                "p75": summary["p75"],
                "session_count": len(firsts_by_path),
                "session_pct": round(len(firsts_by_path) / num_sessions * 100, 1) if num_sessions else 0.0,
            }
        return rows

    overall_classes = {label: user_data.get("overall", {}).get(label, {}) for label in user_data.get("class_order", [])}
    markers["__all__"] = compute_rows(overall_classes, user_data.get("total_sessions", 0))

    for model in models:
        pdata = per_model.get(model, {})
        classes = pdata.get("classes", {})
        markers[model] = compute_rows(classes, pdata.get("num_sessions", 0))
    return markers


def _build_position_marker(file_results: list, marker_def: dict, *, label: str | None = None) -> dict:
    key = marker_def["key"]
    vals = [fr.positions[key] for fr in file_results if fr.positions.get(key) is not None]
    summary = _summarize_positions(vals)
    return {
        "key": key,
        "label": label or marker_def["label"],
        "symbol": marker_def["symbol"],
        "color": marker_def["color"],
        "median": summary["median"],
        "p25": summary["p25"],
        "p75": summary["p75"],
    }


def _build_first_edit_marker(file_results: list, *, label: str | None = None) -> dict:
    return _build_position_marker(file_results, FIRST_EDIT_MARKER, label=label)


def _build_last_edit_marker(file_results: list, *, label: str | None = None) -> dict:
    return _build_position_marker(file_results, LAST_EDIT_MARKER, label=label)


def _build_combined_pi_summary(results: dict[str, list], raw_counts: dict[str, int]) -> dict:
    combined_results = [fr for rows in results.values() for fr in rows]
    key = "all-models-combined"
    phase = pi_phase_profiles({key: combined_results}).get(key, {}) if combined_results else {}
    n = len(combined_results)
    return {
        "label": "All models combined",
        "avg_phase": phase,
        "first_edit_marker": _build_first_edit_marker(combined_results),
        "last_edit_marker": _build_last_edit_marker(combined_results),
        "resolve_rate": round(sum(1 for fr in combined_results if fr.completed) / n * 100, 1) if n else 0.0,
        "num_trajs": n,
        "raw_single_model_count": sum(raw_counts.values()),
    }


def _build_combined_benchmark_summary(results: dict[str, list]) -> dict:
    combined_results = [fr for rows in results.values() for fr in rows]
    key = "all-models-combined"
    phase = benchmark_phase_profiles({key: combined_results}).get(key, {}) if combined_results else {}
    n = len(combined_results)
    return {
        "label": "All benchmark models combined",
        "avg_phase": phase,
        "first_edit_marker": _build_first_edit_marker(combined_results),
        "last_edit_marker": _build_last_edit_marker(combined_results),
        "resolve_rate": round(sum(1 for fr in combined_results if fr.resolved) / n * 100, 1) if n else 0.0,
        "num_trajs": n,
    }


def _parse_merge_specs(specs: list[str] | None) -> list[dict]:
    """Parse ``--merge-models`` flags of the form ``SRC1,SRC2=KEY:LABEL``."""
    out: list[dict] = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"invalid --merge-models spec (missing '='): {spec!r}")
        src_side, dst_side = spec.split("=", 1)
        sources = [s.strip() for s in src_side.split(",") if s.strip()]
        if ":" in dst_side:
            key, label = dst_side.split(":", 1)
        else:
            key, label = dst_side, dst_side
        out.append({"sources": sources, "key": key.strip(), "label": label.strip()})
    return out


def _write_intent_csv(path: Path, data: dict, model_display_names: dict[str, str], models: list[str]) -> None:
    low_prop = data.get("low_proportions", {})
    top_intents = data.get("top_low_intents", [])
    intent_cat = data.get("intent_to_category", {})
    display = data.get("intent_display_names", {})

    headers = ["intent", "category", "display_name", "max_per_100_steps"] + [
        f"{model_display_names.get(m, m)}_per_100_steps" for m in models
    ]
    rows: list[list[str]] = []
    for intent in top_intents:
        vals = [low_prop.get(m, {}).get(intent, 0) * 100 for m in models]
        rows.append([
            intent,
            intent_cat.get(intent, ""),
            display.get(intent, intent),
            f"{max(vals) if vals else 0.0:.2f}",
            *[f"{v:.2f}" for v in vals],
        ])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)



def _write_sidecar_intent_csvs(output_path: Path, payload: dict) -> None:
    models = payload.get("models", [])
    model_display_names = payload.get("model_display_names", {})
    issue_data = {
        "low_proportions": payload.get("low_proportions", {}),
        "top_low_intents": payload.get("top_low_intents", []),
        "intent_to_category": payload.get("intent_to_category", {}),
        "intent_display_names": payload.get("intent_display_names", {}),
    }
    all_single_data = payload.get("all_single_model_intents", {})

    _write_intent_csv(
        output_path.with_name(f"{output_path.stem}.issue-intents.csv"),
        issue_data,
        model_display_names,
        models,
    )
    _write_intent_csv(
        output_path.with_name(f"{output_path.stem}.all-single-model-intents.csv"),
        all_single_data,
        model_display_names,
        models,
    )


def _apply_model_merges(
    merges: list[dict],
    results: dict[str, list],
    allowed_paths: dict[str, set[str]],
    raw_counts,
) -> None:
    """Merge multiple input model keys into a single synthetic key in place."""
    for merge in merges:
        key = merge["key"]
        merged_results: list = []
        merged_paths: set[str] = set()
        merged_count = 0
        for src in merge["sources"]:
            merged_results.extend(results.pop(src, []))
            merged_paths.update(allowed_paths.pop(src, set()))
            merged_count += raw_counts.pop(src, 0)
        if merged_results:
            results[key] = merged_results
        if merged_paths:
            allowed_paths[key] = merged_paths
        if merged_count:
            raw_counts[key] = merged_count

        # Wire the merged key into the benchmark pair map if all sources agree.
        src_benchmarks = {
            BENCHMARK_PAIR_FOR_PI_MODEL[s]
            for s in merge["sources"]
            if s in BENCHMARK_PAIR_FOR_PI_MODEL
        }
        if len(src_benchmarks) == 1:
            BENCHMARK_PAIR_FOR_PI_MODEL[key] = next(iter(src_benchmarks))


def build_payload(
    data_root: Path,
    models: list[str] | None = None,
    session_name_prefixes: list[str] | None = None,
    benchmark_data_root: Path | None = None,
    merge_specs: list[dict] | None = None,
) -> dict:
    chosen_models = models or list(DEFAULT_EXACT_MODELS)
    all_results = process_all(data_root, models=chosen_models)

    issue_filter = SessionFilter(
        allowed_models=chosen_models,
        require_single_model=True,
        session_name_prefixes=session_name_prefixes,
    )
    allowed_paths, raw_counts, _ = collect_filtered_paths(data_root, issue_filter)
    results = _filter_results_to_paths(all_results, allowed_paths)
    for model in chosen_models:
        raw_n = raw_counts.get(model, 0)
        analyzed_n = len(results.get(model, []))
        print(f"  {model}: {raw_n} strict single-model sessions, {analyzed_n} analyzed")

    all_single_filter = SessionFilter(
        allowed_models=chosen_models,
        require_single_model=True,
        session_name_prefixes=None,
    )
    all_single_allowed_paths, all_single_raw_counts, _ = collect_filtered_paths(data_root, all_single_filter)
    all_single_results = _filter_results_to_paths(all_results, all_single_allowed_paths)

    merges = merge_specs or []
    if merges:
        _apply_model_merges(merges, results, allowed_paths, raw_counts)
        _apply_model_merges(merges, all_single_results, all_single_allowed_paths, all_single_raw_counts)
        for merge in merges:
            key = merge["key"]
            raw_n = raw_counts.get(key, 0)
            analyzed_n = len(results.get(key, []))
            print(f"  → merged {'+'.join(merge['sources'])} as {key}: "
                  f"{raw_n} sessions, {analyzed_n} analyzed")

    present_keys = list({*chosen_models, *(m["key"] for m in merges)})
    sorted_models = sorted(
        [m for m in present_keys if raw_counts.get(m, 0) > 0],
        key=lambda m: (-raw_counts.get(m, 0), -len(results.get(m, [])), m),
    )
    payload = build_analytics_payload(results)
    all_single_intent_payload = build_analytics_payload(all_single_results)
    for merge in merges:
        if merge["key"] in payload.get("model_display_names", {}):
            payload["model_display_names"][merge["key"]] = merge["label"]
        if merge["key"] in all_single_intent_payload.get("model_display_names", {}):
            all_single_intent_payload["model_display_names"][merge["key"]] = merge["label"]
    payload["raw_single_model_counts"] = {m: raw_counts.get(m, 0) for m in sorted_models}
    payload["analyzed_counts"] = {m: len(results.get(m, [])) for m in sorted_models}
    payload["models"] = sorted_models
    payload["intervention_markers"] = _compute_intervention_markers(allowed_paths, sorted_models)
    payload["first_edit_markers"] = {m: _build_first_edit_marker(results.get(m, [])) for m in sorted_models}
    payload["last_edit_markers"] = {m: _build_last_edit_marker(results.get(m, [])) for m in sorted_models}
    payload["all_single_model_intents"] = {
        "models": sorted_models,
        "low_proportions": all_single_intent_payload["low_proportions"],
        "top_low_intents": all_single_intent_payload["top_low_intents"],
        "intent_to_category": all_single_intent_payload["intent_to_category"],
        "intent_display_names": all_single_intent_payload["intent_display_names"],
        "raw_single_model_counts": {m: all_single_raw_counts.get(m, 0) for m in sorted_models},
        "analyzed_counts": {m: len(all_single_results.get(m, [])) for m in sorted_models},
        "num_trajs": {m: len(all_single_results.get(m, [])) for m in sorted_models},
    }

    benchmark_root = benchmark_data_root or Path("data")
    benchmark_models = sorted({BENCHMARK_PAIR_FOR_PI_MODEL[m] for m in sorted_models if m in BENCHMARK_PAIR_FOR_PI_MODEL})
    if benchmark_models:
        benchmark_results = process_benchmark_all(benchmark_root, models=benchmark_models)
        benchmark_payload = build_benchmark_payload(benchmark_results)
        payload["benchmark"] = {
            "pair_for_pi_model": {m: BENCHMARK_PAIR_FOR_PI_MODEL[m] for m in sorted_models if m in BENCHMARK_PAIR_FOR_PI_MODEL},
            "avg_phase": benchmark_payload["avg_phase"],
            "median_last_edit": benchmark_payload["median_last_edit"],
            "first_edit_markers": {m: _build_first_edit_marker(benchmark_results.get(m, [])) for m in benchmark_models},
            "last_edit_markers": {m: _build_last_edit_marker(benchmark_results.get(m, [])) for m in benchmark_models},
            "resolve_rate": benchmark_payload["resolve_rate"],
            "model_display_names": benchmark_payload["model_display_names"],
            "num_trajs": benchmark_payload["num_trajs"],
        }
    else:
        payload["benchmark"] = {
            "pair_for_pi_model": {},
            "avg_phase": {},
            "median_last_edit": {},
            "first_edit_markers": {},
            "last_edit_markers": {},
            "resolve_rate": {},
            "model_display_names": {},
            "num_trajs": {},
        }
    return payload


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Trajectory Analytics — Pi transcripts</title>
  <style>
    :root {{
      --bg: #fffff8;
      --panel: #fffff8;
      --muted: #777;
      --text: #333;
      --accent: #5a7d9a;
      --border: #ddd;
      --claude: #b8785e;
      --gpt: #6a8da8;
      --glm: #9a6a9a;
      --gemini: #6a9a6a;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; font-weight: 400; letter-spacing: -0.3px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 36px; font-style: italic; }}
    h2 {{
      font-size: 16px;
      font-weight: 400;
      font-style: italic;
      margin: 44px 0 6px 0;
      padding-top: 22px;
      border-top: 1px solid #e0e0e0;
      color: var(--text);
    }}
    .chart-desc {{ color: var(--muted); font-size: 12.5px; margin-bottom: 16px; }}
    .filter-control {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin: -6px 0 14px 0;
      font-size: 12px;
      color: var(--muted);
    }}
    .filter-control label {{
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .filter-control input[type="range"] {{
      width: 220px;
    }}
    .filter-value {{
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }}
    .filter-meta {{
      font-size: 11px;
      color: var(--muted);
    }}
    .legend {{
      display: flex; gap: 20px; margin-bottom: 16px; font-size: 12.5px;
    }}
    .legend-item {{
      display: flex; align-items: center; gap: 6px;
    }}
    .legend-swatch {{
      width: 12px; height: 12px; border-radius: 2px;
    }}
    .chart-wrapper {{
      padding: 16px 0;
      margin-bottom: 6px;
      overflow-x: auto;
    }}
    canvas {{ display: block; }}

    .model-tag {{
      display: inline-block;
      font-size: 12px;
      font-style: italic;
    }}
    .model-tag.gpt {{ color: var(--gpt); }}
    .model-tag.claude {{ color: var(--claude); }}
    .model-tag.glm {{ color: var(--glm); }}
    .model-tag.gemini {{ color: var(--gemini); }}

    .side-by-side {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 800px) {{
      .side-by-side {{ grid-template-columns: 1fr; }}
    }}
    .side-by-side .chart-wrapper {{ margin-bottom: 0; }}
    .side-label {{
      font-size: 13px; font-style: italic; margin-bottom: 10px; color: var(--text);
    }}
    .stacked-panel-header {{
      display: flex;
      align-items: baseline;
      gap: 14px;
      margin-bottom: 2px;
    }}
    .stacked-panel-header .model-tag {{
      font-size: 13px;
    }}
    .stacked-panel-header .panel-subhead {{
      font-size: 11.5px;
      font-style: italic;
      color: var(--muted);
    }}
    #stackedPanels .chart-wrapper {{
      padding: 4px 0;
      margin-bottom: 0;
    }}
    .stacked-pair-row {{
      margin: 6px 0 12px 0;
    }}
    .stacked-pair-row .side-label {{
      margin-bottom: 4px;
    }}
    .stacked-pair-row .panel-subhead {{
      font-size: 11px;
      color: var(--muted);
      padding-left: 8px;
    }}

    /* Dumbbell chart */
    .dumbbell-row {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 8px;
      align-items: center;
      padding: 4px 0;
      border-bottom: 1px solid #eee;
    }}
    .dumbbell-row:last-child {{ border-bottom: none; }}
    .dumbbell-label {{
      text-align: right;
      font-size: 12px;
      color: var(--text);
    }}
    .dumbbell-track {{
      position: relative;
      height: 22px;
    }}
    .dumbbell-line {{
      position: absolute;
      top: 10px;
      height: 2px;
      border-radius: 1px;
    }}
    .dumbbell-dot {{
      position: absolute;
      top: 3px;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .dumbbell-val {{
      position: absolute;
      top: 2px;
      font-size: 10px;
      white-space: nowrap;
    }}
    .dumbbell-header {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 8px;
      font-size: 11px;
      color: var(--muted);
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 4px;
    }}
    .dumbbell-scale {{
      position: relative;
      height: 16px;
    }}
    .dumbbell-scale-tick {{
      position: absolute;
      top: 0;
      font-size: 9px;
      color: var(--muted);
      transform: translateX(-50%);
    }}

    /* Paired bar table */
    .paired-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
    }}
    .paired-table th {{
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 400;
      font-style: italic;
      color: var(--muted);
    }}
    .paired-table td {{
      padding: 0;
    }}
    .paired-table tr.paired-row {{
      border-bottom: 8px solid transparent;
    }}
    .paired-table tr.paired-row.zebra td {{
      background: rgba(0,0,0,0.015);
    }}
    .paired-table td.paired-name {{
      text-align: right;
      padding-right: 12px;
      color: var(--text);
      white-space: nowrap;
      width: 1%;
      vertical-align: middle;
    }}
    .paired-table td.paired-bars {{
      padding: 4px 4px;
    }}
    .paired-bar-row {{
      display: flex;
      align-items: center;
      height: 11px;
      gap: 4px;
    }}
    .paired-bar {{
      height: 11px;
      border-radius: 2px;
      opacity: 0.85;
    }}
    .paired-bar-val {{
      font-size: 9.5px;
      min-width: 24px;
      color: var(--muted);
    }}
    .paired-table .cat-header td {{
      padding: 14px 0 4px 0;
      font-weight: 400;
      font-style: italic;
      font-size: 13px;
      color: var(--text);
      letter-spacing: 0.3px;
    }}
    .cat-annotation {{
      font-style: normal;
      font-size: 11.5px;
      color: var(--muted);
      margin-top: 2px;
      line-height: 1.4;
    }}

    /* Stacked area legend */
    .stacked-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin-top: 10px;
      font-size: 11.5px;
      color: var(--text);
    }}
    .stacked-legend .item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .stacked-legend .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
      flex-shrink: 0;
    }}
    .trajectory-controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin: 10px 0 14px;
    }}
    .segmented-control {{
      display: inline-flex;
      border: 1px solid var(--border);
      border-radius: 999px;
      overflow: hidden;
      background: rgba(0,0,0,0.02);
    }}
    .segmented-control button {{
      border: 0;
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-size: 12px;
      padding: 6px 12px;
      cursor: pointer;
      transition: background 0.12s ease, color 0.12s ease;
    }}
    .segmented-control button.active {{
      background: var(--text);
      color: var(--bg);
    }}
    .control-note {{
      font-size: 11.5px;
      color: var(--muted);
      font-style: italic;
    }}

    /* Transition matrix */
    .tmatrix {{
      display: inline-grid;
      gap: 2px;
      font-size: 11px;
      font-family: ui-monospace, monospace;
    }}
    .tmatrix .corner {{ }}
    .tmatrix .col-hdr {{
      text-align: center;
      color: var(--text);
      font-weight: 700;
      padding: 2px 0;
    }}
    .tmatrix .row-hdr {{
      display: flex; align-items: center; justify-content: flex-end;
      color: var(--text);
      font-weight: 700;
      padding-right: 6px;
    }}
    .tmatrix .cell {{
      width: 48px; height: 32px;
      border-radius: 3px;
      display: flex; align-items: center; justify-content: center;
      font-size: 10px;
      color: rgba(255,255,255,0.85);
      transition: transform 0.1s;
    }}
    .tmatrix .cell:hover {{
      transform: scale(1.15);
      z-index: 2;
      outline: 1px solid var(--text);
    }}
  </style>
</head>
<body>
<div class="container">
  <h1>Trajectory Analytics</h1>
  <div class="subtitle">Pi transcript sessions &mdash; configurable session filter, currently strict single-model issue sessions</div>

  <div class="legend" id="topLegend"></div>

  <h2>1. High-Level Action Frequencies</h2>
  <p class="chart-desc">Proportion of steps in each high-level category. Normalised so models are comparable despite different step counts.</p>
  <div class="chart-wrapper">
    <div id="highPaired"></div>
  </div>

  <h2>2. Intent Comparison</h2>
  <p class="chart-desc">Frequency per 100 steps, compared across all models. For Pi, the git rows use a more semantic sub-taxonomy: GitHub context, repo inspection, diff review, sync/integrate, local state change, and publish.</p>
  <p class="chart-desc" id="issueIntentDesc"></p>
  <div class="filter-control">
    <label for="issueIntentMaxVSlider">show rows where maxV &gt; <span class="filter-value" id="issueIntentMaxVValue">0.0</span> per 100 steps</label>
    <input id="issueIntentMaxVSlider" type="range" min="0" max="5" step="0.1" value="0" />
    <span class="filter-meta" id="issueIntentMaxVCount"></span>
  </div>
  <div class="chart-wrapper">
    <div id="heatTable"></div>
  </div>

  <h2>3. Intent Comparison (all single-model sessions)</h2>
  <p class="chart-desc">Same intent aggregation as Section 2, but using every available strict single-model Pi session for the selected models, not just sessions whose final session name starts with <code>Issue:</code>.</p>
  <p class="chart-desc" id="allSingleIntentDesc"></p>
  <div class="filter-control">
    <label for="allSingleIntentMaxVSlider">show rows where maxV &gt; <span class="filter-value" id="allSingleIntentMaxVValue">0.0</span> per 100 steps</label>
    <input id="allSingleIntentMaxVSlider" type="range" min="0" max="5" step="0.1" value="0" />
    <span class="filter-meta" id="allSingleIntentMaxVCount"></span>
  </div>
  <div class="chart-wrapper">
    <div id="heatTableAllSingle"></div>
  </div>

  <h2>4. Steps per trajectory, by model</h2>
  <p class="chart-desc">Cumulative share of runs that finished within N steps. Dashed line marks the 250-step cap.</p>
  <p class="chart-desc" id="stepDistDesc"></p>
  <div class="chart-wrapper">
    <canvas id="stepDistChart" height="320"></canvas>
  </div>

  <h2>5. Typical Trajectory Shape</h2>
  <p class="chart-desc">Each model is shown as a pair: benchmark (agent alone) above, maintainer-guided Pi sessions below. Markers are median-only, with no bands: △ = authorization, ○ = steering, □ = closeout, ◆ = first edit, ◇ = last edit.</p>
  <p class="chart-desc">Where a direct public benchmark run is unavailable in this repo, the benchmark row uses the closest family baseline we do have: GPT-5 for the <code>gpt-5.*</code> models and Sonnet 4.5 for the <code>claude-opus-4-*</code> models.</p>
  <div class="trajectory-controls">
    <div class="segmented-control" id="trajectoryGitToggle" role="tablist" aria-label="Trajectory shape git view">
      <button type="button" class="active" data-include-git="true">with git</button>
      <button type="button" data-include-git="false">without git</button>
    </div>
    <div class="control-note" id="trajectoryGitNote"></div>
  </div>
  <div id="stackedPanels"></div>


</div>

<script>
const D = {payload_json};

// ── Helpers ──────────────────────────────────────────────
function getCtx(id) {{
  const c = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const cssW = c.parentElement.clientWidth - 40;
  const cssH = c.height;
  c.width = cssW * dpr;
  c.height = cssH * dpr;
  c.style.width = cssW + 'px';
  c.style.height = cssH + 'px';
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  return {{ canvas: c, ctx, w: cssW, h: cssH }};
}}

const MODEL_COLORS = D.model_colors;
const MODEL_NAMES = D.model_display_names;
// Single source of truth for model order: resolve rate, descending.
// Every section consumes ALL_MODELS, so they all line up.
const ALL_MODELS = [...D.models].sort(
  (a, b) => (D.resolve_rate[b] ?? -Infinity) - (D.resolve_rate[a] ?? -Infinity)
);
const tagClass = {{
  'gpt5': 'gpt',
  'claude45': 'claude',
  'glm45': 'glm',
  'gemini25pro': 'gemini',
}};

(function() {{
  const el = document.getElementById('topLegend');
  if (!el) return;
  el.innerHTML = ALL_MODELS.map(m => (
    `<div class="legend-item">` +
      `<div class="legend-swatch" style="background:${{MODEL_COLORS[m]}}"></div>` +
      `<span>${{MODEL_NAMES[m]}}</span>` +
    `</div>`
  )).join('');
}})();
const CLAUDE_COLOR = '#b8785e';
const GPT_COLOR = '#6a8da8';
const GLM_COLOR = '#6a9a6a';
const GEMINI_COLOR = '#9a6a9a';
const MUTED = '#6b7280';
const TEXT = '#1a1a1a';
const STACKED_GROUPS = [
  {{ name: 'understand',   letters: ['R','S'], color: '#5a7d9a' }},
  {{ name: 'reproduce',    letters: ['P'],     color: '#b0956a' }},
  {{ name: 'edit',         letters: ['E'],     color: '#4a8a5a' }},
  {{ name: 'verify',       letters: ['V'],     color: '#b56a50' }},
  {{ name: 'git',          letters: ['G'],     color: '#8a7a5a' }},
  {{ name: 'housekeeping', letters: ['H'],     color: '#7a9a52' }},
];
let includeGitInStacked = true;

function fmtPct(v) {{
  if (v == null || Number.isNaN(v)) return '—';
  return Math.abs(v - Math.round(v)) < 0.05 ? String(Math.round(v)) : v.toFixed(1);
}}

// Names for display (everywhere except transition matrices)
const CATEGORY_NAMES = ['read','search','reproduce','edit','verify','git','housekeeping'];
const NAME_COLORS = D.name_colors;

// Letters for transition matrices only
const LETTERS = Object.values(D.name_to_letter);
const LETTER_COLORS = D.letter_colors;
const LETTER_TO_NAME = D.letter_to_name;

// ── 1. High-Level Bar Chart ──────────────────────────────
function drawGroupedBar(canvasId, labels, gptVals, claudeVals, labelMap) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 50, right = 20, top = 20, bot = 60;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const n = labels.length;
  const groupW = plotW / n;
  const barW = groupW * 0.35;
  const maxVal = Math.max(...gptVals, ...claudeVals, 0.01);

  // Y axis
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = top + plotH - (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText((maxVal * i / 4 * 100).toFixed(1) + '%', left - 6, y + 4);
  }}

  for (let i = 0; i < n; i++) {{
    const x = left + i * groupW + groupW * 0.1;
    const gH = (gptVals[i] / maxVal) * plotH;
    const cH = (claudeVals[i] / maxVal) * plotH;

    ctx.fillStyle = GPT_COLOR;
    ctx.fillRect(x, top + plotH - gH, barW, gH);

    ctx.fillStyle = CLAUDE_COLOR;
    ctx.fillRect(x + barW + 2, top + plotH - cH, barW, cH);

    // Label
    ctx.fillStyle = TEXT;
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    const lx = x + barW + 1;
    const label = labels[i];
    ctx.fillText(label, lx, top + plotH + 16);
    if (labelMap && labelMap[label]) {{
      ctx.fillStyle = MUTED;
      ctx.font = '10px sans-serif';
      ctx.fillText(labelMap[label], lx, top + plotH + 30);
    }}
  }}
}}

// ── 2. Horizontal grouped bar (for long labels) ─────────
function drawHorizontalGroupedBar(canvasId, labels, gptVals, claudeVals) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 160, right = 30, top = 10, bot = 30;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const n = labels.length;
  const rowH = plotH / n;
  const barH = rowH * 0.35;
  const maxVal = Math.max(...gptVals, ...claudeVals, 0.001);

  // Vertical grid
  ctx.strokeStyle = '#e0e0e0'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const x = left + (i / 4) * plotW;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    ctx.fillText((maxVal * i / 4 * 100).toFixed(1) + '%', x, top + plotH + 16);
  }}

  for (let i = 0; i < n; i++) {{
    const y = top + i * rowH + rowH * 0.15;
    const gW = (gptVals[i] / maxVal) * plotW;
    const cW = (claudeVals[i] / maxVal) * plotW;

    ctx.fillStyle = GPT_COLOR;
    ctx.fillRect(left, y, gW, barH);

    ctx.fillStyle = CLAUDE_COLOR;
    ctx.fillRect(left, y + barH + 1, cW, barH);

    ctx.fillStyle = TEXT; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(labels[i], left - 8, y + rowH * 0.45);
  }}
}}

// ── Draw charts ──────────────────────────────────────────

// 1. High-level paired bars
(function() {{
  const el = document.getElementById('highPaired');
  const cats = CATEGORY_NAMES;
  const maxVal = Math.max(...cats.map(c =>
    Math.max(...ALL_MODELS.map(m => D.high_proportions[m][c] || 0))
  ));

  function barPct(v) {{ return (v / maxVal * 100).toFixed(1); }}

  let html = `<table class="paired-table">
    <thead><tr>
      <th></th>
      <th style="text-align:left;padding-left:4px">
        ${{ALL_MODELS.map(m => `<span style="color:${{MODEL_COLORS[m]}}">${{MODEL_NAMES[m]}}</span>`).join(' <span style="color:var(--muted);padding:0 4px">/</span> ')}}
        <span style="color:var(--muted);font-weight:400;padding-left:8px">% of steps</span>
      </th>
    </tr></thead><tbody>`;

  let rowIdx = 0;
  for (const cat of cats) {{
    const vals = ALL_MODELS.map(m => (D.high_proportions[m][cat] || 0) * 100);
    const best = Math.max(...vals);
    const zebra = rowIdx % 2 === 1 ? ' zebra' : '';

    html += `<tr class="paired-row${{zebra}}">
      <td class="paired-name">${{cat}}</td>
      <td class="paired-bars">`;
    for (let mi = 0; mi < ALL_MODELS.length; mi++) {{
      const m = ALL_MODELS[mi];
      const v = vals[mi];
      const bold = v === best && best >= 0.3 ? 'font-weight:700' : '';
      html += `<div class="paired-bar-row">
          <div class="paired-bar" style="width:${{barPct(v / 100)}}%;background:${{MODEL_COLORS[m]}}"></div>
          <span class="paired-bar-val" style="color:${{MODEL_COLORS[m]}};${{bold}}">${{v.toFixed(1)}}</span>
        </div>`;
    }}
    html += `</td>
    </tr>`;
    rowIdx++;
  }}

  html += '</tbody></table>';
  el.innerHTML = html;
}})();

// 2–3. Paired horizontal bars
function renderIntentComparisonTable(containerId, data, threshold, countId) {{
  const el = document.getElementById(containerId);
  if (!el || !data) return;

  const intents = data.top_low_intents || [];
  const catMap = data.intent_to_category || {{}};
  const displayNames = data.intent_display_names || {{}};
  const lowProportions = data.low_proportions || {{}};
  const catOrder = ['read','search','reproduce','edit','verify','git','housekeeping','other'];

  const allRows = intents.map(intent => {{
    const vals = {{}};
    let maxV = 0;
    for (const m of ALL_MODELS) {{
      vals[m] = (((lowProportions[m] || {{}})[intent]) || 0) * 100;
      if (vals[m] > maxV) maxV = vals[m];
    }}
    const cat = catMap[intent] || '?';
    return {{ intent, vals, maxV, cat }};
  }});

  const rows = allRows.filter(r => r.maxV > threshold);
  const countEl = countId ? document.getElementById(countId) : null;
  if (countEl) countEl.textContent = `${{rows.length}}/${{allRows.length}} rows shown`;

  if (!rows.length) {{
    el.innerHTML = `<div class="chart-desc">No intent rows exceed ${{threshold.toFixed(1)}} per 100 steps.</div>`;
    return;
  }}

  const grouped = {{}};
  for (const r of rows) {{
    if (!grouped[r.cat]) grouped[r.cat] = [];
    grouped[r.cat].push(r);
  }}
  for (const cat of Object.keys(grouped)) {{
    grouped[cat].sort((a, b) => b.maxV - a.maxV);
  }}

  const maxVal = Math.max(...rows.map(r => r.maxV), 1);
  function barPct(v) {{ return (v / maxVal * 100).toFixed(1); }}

  let html = `<table class="paired-table"><tbody>`;
  let rowIdx = 0;
  for (const cat of catOrder) {{
    if (!grouped[cat] || grouped[cat].length === 0) continue;
    html += `<tr class="cat-header"><td colspan="2">${{cat}}</td></tr>`;

    for (const r of grouped[cat]) {{
      const displayName = displayNames[r.intent] || r.intent;
      const zebra = rowIdx % 2 === 1 ? ' zebra' : '';
      const best = Math.max(...ALL_MODELS.map(m => r.vals[m]));

      html += `<tr class="paired-row${{zebra}}">
        <td class="paired-name" title="${{r.intent}}">${{displayName}}</td>
        <td class="paired-bars">`;
      for (const m of ALL_MODELS) {{
        const v = r.vals[m];
        const bold = v === best && best >= 0.3 ? 'font-weight:700' : '';
        html += `<div class="paired-bar-row">
            <div class="paired-bar" style="width:${{barPct(v)}}%;background:${{MODEL_COLORS[m]}}"></div>
            <span class="paired-bar-val" style="color:${{MODEL_COLORS[m]}};${{bold}}">${{v.toFixed(1)}}</span>
          </div>`;
      }}
      html += `</td></tr>`;
      rowIdx++;
    }}
  }}

  html += '</tbody></table>';
  el.innerHTML = html;
}}

function bindIntentThresholdControl(sliderId, valueId, countId, containerId, data, defaultThreshold) {{
  const slider = document.getElementById(sliderId);
  const valueEl = document.getElementById(valueId);
  function render() {{
    const threshold = Number(slider?.value || defaultThreshold || 0);
    if (valueEl) valueEl.textContent = threshold.toFixed(1);
    renderIntentComparisonTable(containerId, data, threshold, countId);
  }}
  if (slider) slider.addEventListener('input', render);
  render();
}}

function renderIntentComparisonSummary(descId, data, scopeLabel) {{
  const el = document.getElementById(descId);
  if (!el || !data) return;

  const rawCounts = data.raw_single_model_counts || {{}};
  const analyzedCounts = data.analyzed_counts || data.num_trajs || {{}};
  const totalRaw = ALL_MODELS.reduce((sum, m) => sum + (rawCounts[m] || 0), 0);
  const totalAnalyzed = ALL_MODELS.reduce((sum, m) => sum + (analyzedCounts[m] || 0), 0);
  const perModel = ALL_MODELS
    .map(m => `${{MODEL_NAMES[m]}}: ${{rawCounts[m] || 0}} raw / ${{analyzedCounts[m] || 0}} analyzed`)
    .join(' · ');

  el.textContent = `${{scopeLabel}}: ${{totalRaw}} raw strict single-model sessions, ${{totalAnalyzed}} analyzed. ${{perModel}}.`;
}}

bindIntentThresholdControl('issueIntentMaxVSlider', 'issueIntentMaxVValue', 'issueIntentMaxVCount', 'heatTable', {{
  top_low_intents: D.top_low_intents,
  low_proportions: D.low_proportions,
  intent_to_category: D.intent_to_category,
  intent_display_names: D.intent_display_names,
}}, 0);
renderIntentComparisonSummary('issueIntentDesc', {{
  raw_single_model_counts: D.raw_single_model_counts,
  analyzed_counts: D.analyzed_counts,
  num_trajs: D.num_trajs,
}}, 'Issue sessions only');

bindIntentThresholdControl('allSingleIntentMaxVSlider', 'allSingleIntentMaxVValue', 'allSingleIntentMaxVCount', 'heatTableAllSingle', D.all_single_model_intents, 0);
renderIntentComparisonSummary('allSingleIntentDesc', D.all_single_model_intents, 'All sessions');

// 4. Step distribution — overlaid ECDFs
(function() {{
  const {{ ctx, w, h }} = getCtx('stepDistChart');
  const left = 56, top = 22, bot = 46;
  const xMax = 250;

  // Measure right-edge labels so the right margin always fits them
  ctx.font = '11px sans-serif';
  const sampleLabels = ALL_MODELS.map(m => `${{MODEL_NAMES[m]}} · 999 steps`);
  const maxLabelW = Math.max(...sampleLabels.map(s => ctx.measureText(s).width));
  const right = Math.ceil(maxLabelW) + 24;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const xPx = x => left + (x / xMax) * plotW;
  const yPx = y => top + (1 - y) * plotH;

  // Build ECDF points and censored fraction per model from binned step counts
  const cdfs = {{}};
  const censored = {{}};
  for (const m of ALL_MODELS) {{
    const entries = Object.entries(D.step_dist[m])
      .map(([k, v]) => [Number(k), v])
      .sort((a, b) => a[0] - b[0]);
    const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
    let cum = 0, beforeCap = 0;
    const pts = [[0, 0]];
    for (const [bin, count] of entries) {{
      if (bin < xMax) beforeCap += count;
      cum += count;
      pts.push([Math.min(bin, xMax), cum / total]);
    }}
    if (pts[pts.length - 1][1] < 1) pts.push([xMax, 1]);
    cdfs[m] = pts;
    censored[m] = 1 - beforeCap / total;
  }}

  // Faint guides at 25% and 75%; stronger rule at 50%
  ctx.strokeStyle = '#e8e8e8'; ctx.lineWidth = 0.5;
  for (const yv of [0.25, 0.75]) {{
    const py = yPx(yv);
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(left + plotW, py); ctx.stroke();
  }}
  ctx.strokeStyle = '#bdbdbd'; ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(left, yPx(0.5)); ctx.lineTo(left + plotW, yPx(0.5)); ctx.stroke();

  // Y-axis labels in percent
  ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'right';
  for (const yv of [0, 0.25, 0.5, 0.75, 1]) {{
    ctx.fillText((yv * 100).toFixed(0) + '%', left - 6, yPx(yv) + 3);
  }}

  // Baseline
  ctx.strokeStyle = '#cfcfcf'; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(left, yPx(0)); ctx.lineTo(left + plotW, yPx(0)); ctx.stroke();

  // X-axis ticks
  ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  for (let x = 0; x <= xMax; x += 50) {{
    ctx.fillText(x, xPx(x), top + plotH + 14);
  }}

  // Vertical dashed rule at the step cap; label sits at the bottom-right of the rule
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(xPx(xMax), top); ctx.lineTo(xPx(xMax), top + plotH); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = MUTED; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  ctx.fillText('step cap', xPx(xMax) - 4, top + plotH - 4);

  // Step curves (staircase — keep it sharp, do not smooth)
  for (const m of ALL_MODELS) {{
    const pts = cdfs[m];
    ctx.strokeStyle = MODEL_COLORS[m];
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    let prevY = yPx(0);
    ctx.moveTo(xPx(0), prevY);
    for (const [x, y] of pts) {{
      const px = xPx(x);
      ctx.lineTo(px, prevY);
      const py = yPx(y);
      ctx.lineTo(px, py);
      prevY = py;
    }}
    ctx.stroke();
  }}

  // Medians via linear interpolation across the cdf
  const medians = {{}};
  for (const m of ALL_MODELS) {{
    const pts = cdfs[m];
    let med = null;
    for (let i = 1; i < pts.length; i++) {{
      if (pts[i][1] >= 0.5) {{
        const [x0, y0] = pts[i - 1];
        const [x1, y1] = pts[i];
        med = y1 === y0 ? x1 : x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0);
        break;
      }}
    }}
    medians[m] = med;
  }}

  // Filled dot on each curve where it crosses y=50%
  for (const m of ALL_MODELS) {{
    if (medians[m] == null) continue;
    const px = xPx(medians[m]);
    const py = yPx(0.5);
    ctx.fillStyle = MODEL_COLORS[m];
    ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fffff8'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.stroke();
  }}

  // Censoring shelf annotation: small label hung below the plateau, near the cap
  ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  for (const m of ALL_MODELS) {{
    const pct = censored[m] * 100;
    if (pct < 1) continue;
    const pts = cdfs[m];
    let yShelf = 1;
    for (let i = pts.length - 1; i >= 0; i--) {{
      if (pts[i][0] < xMax) {{ yShelf = pts[i][1]; break; }}
    }}
    const px = xPx(xMax) - 6;
    const py = yPx(yShelf) + 14;
    ctx.fillStyle = MODEL_COLORS[m];
    ctx.fillText(`~${{pct.toFixed(0)}}% hit cap`, px, py);
  }}

  // Direct labels at right end, sorted by median ascending so top→bottom
  // mirrors the curves left→right at y=50%.
  const labels = ALL_MODELS
    .map(m => ({{ m, med: medians[m] }}))
    .filter(d => d.med != null)
    .sort((a, b) => a.med - b.med);
  ctx.textAlign = 'left';
  const lineH = 16;
  const labelX = left + plotW + 10;
  // Caption above the label stack — explains what the numbers mean once
  ctx.font = 'italic 10px sans-serif'; ctx.fillStyle = MUTED;
  ctx.fillText('half its runs finish in:', labelX, top + 4);
  ctx.font = '11px sans-serif';
  let py0 = top + 20;
  for (let i = 0; i < labels.length; i++) {{
    const item = labels[i];
    const py = py0 + i * lineH;
    ctx.fillStyle = MODEL_COLORS[item.m];
    ctx.fillText(
      `${{MODEL_NAMES[item.m]}} · ${{Math.round(item.med)}} steps`,
      labelX, py
    );
  }}

  // Axis labels
  ctx.fillStyle = MUTED; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('steps per trajectory', left + plotW / 2, h - 6);
  ctx.save();
  ctx.translate(14, top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('% of runs finished in ≤ N steps', 0, 0);
  ctx.restore();

  // Findings-led second subhead, generated from the data
  const desc = document.getElementById('stepDistDesc');
  if (desc && labels.length) {{
    const fastest = labels[0];
    const slowest = labels[labels.length - 1];
    const capHits = ALL_MODELS
      .map(m => ({{ m, pct: censored[m] * 100 }}))
      .filter(d => d.pct >= 1)
      .sort((a, b) => b.pct - a.pct);
    const parts = [
      `${{MODEL_NAMES[fastest.m]}} finishes fastest (median ${{Math.round(fastest.med)}} steps)`,
      `${{MODEL_NAMES[slowest.m]}} runs longest (median ${{Math.round(slowest.med)}})`,
    ];
    if (capHits.length) {{
      parts.push(
        capHits
          .map(d => `~${{d.pct.toFixed(0)}}% of ${{MODEL_NAMES[d.m]}} runs hit the 250-step cap`)
          .join('; ')
      );
    }}
    desc.textContent = parts.join('; ') + '.';
  }}
}})();

// 6. Stacked area charts — paired benchmark vs maintainer-guided panels
function drawStackedArea(canvasId, phaseData, opts = {{}}) {{
  const markers = opts.markers || [];
  const showMarkers = markers.length > 0;
  const includeGit = opts.includeGit !== false;
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 40, right = 18, top = 22, bot = showMarkers ? 28 : 12;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const bins = 20;

  const groups = STACKED_GROUPS.filter(g => includeGit || g.name !== 'git');

  const xPct = pct => left + (pct / 100) * plotW;
  const xAtBin = i => left + (i / (bins - 1)) * plotW;

  const groupVals = groups.map(g => {{
    const summed = new Array(bins).fill(0);
    for (const l of g.letters) {{
      const vals = phaseData?.[l] || [];
      for (let b = 0; b < bins; b++) summed[b] += vals[b] || 0;
    }}
    return summed;
  }});

  const stacked = [];
  let cumulative = new Array(bins).fill(0);
  for (let gi = 0; gi < groups.length; gi++) {{
    const layer = groupVals[gi].map((v, i) => cumulative[i] + v);
    stacked.push({{ group: groups[gi], bottom: [...cumulative], top: layer }});
    cumulative = layer;
  }}
  const maxes = cumulative;

  function yAt(v, binIdx) {{
    const norm = maxes[binIdx] > 0 ? v / maxes[binIdx] : 0;
    return top + plotH - norm * plotH;
  }}

  for (let s = stacked.length - 1; s >= 0; s--) {{
    const layer = stacked[s];
    ctx.fillStyle = layer.group.color;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (let i = 0; i < bins; i++) {{
      const x = xAtBin(i), y = yAt(layer.top[i], i);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    for (let i = bins - 1; i >= 0; i--) {{
      ctx.lineTo(xAtBin(i), yAt(layer.bottom[i], i));
    }}
    ctx.closePath();
    ctx.fill();
  }}
  ctx.globalAlpha = 1;

  ctx.fillStyle = MUTED;
  ctx.font = '9px Palatino, Georgia, serif';
  ctx.textAlign = 'center';
  ctx.strokeStyle = 'rgba(0,0,0,0.1)';
  ctx.lineWidth = 0.5;
  for (const pct of [0, 25, 50, 75, 100]) {{
    const x = xPct(pct);
    ctx.fillText(`${{pct}}%`, x, top - 8);
    ctx.beginPath();
    ctx.moveTo(x, top - 3);
    ctx.lineTo(x, top + 4);
    ctx.stroke();
  }}

  const halfX = xPct(50);
  ctx.strokeStyle = 'rgba(0,0,0,0.07)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(halfX, top);
  ctx.lineTo(halfX, top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = '#cfcfcf';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(left, top + plotH);
  ctx.lineTo(left + plotW, top + plotH);
  ctx.stroke();

  if (showMarkers) {{
    for (const marker of markers) {{
      if (marker?.median == null) continue;
      const xm = xPct(marker.median);
      ctx.strokeStyle = marker.color;
      ctx.lineWidth = marker.key === 'last_edit' ? 1.0 : 1.2;
      ctx.beginPath();
      ctx.moveTo(xm, top);
      ctx.lineTo(xm, top + plotH);
      ctx.stroke();
    }}

    const placed = [];
    for (const marker of markers) {{
      if (marker?.median == null) continue;
      const xm = xPct(marker.median);
      let y = top + plotH + 13;
      for (const prev of placed) {{
        if (Math.abs(prev.x - xm) < 16 && Math.abs(prev.y - y) < 8) y += 10;
      }}
      placed.push({{ x: xm, y }});
      ctx.fillStyle = marker.color;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(marker.symbol, xm, y);
    }}
  }}

  for (let s = 0; s < stacked.length; s++) {{
    const layer = stacked[s];
    const searchFrom = 2;
    const searchTo = bins - 3;
    let bestBin = Math.floor(bins / 2), bestH = 0;
    for (let i = searchFrom; i <= searchTo; i++) {{
      const h = Math.abs(yAt(layer.bottom[i], i) - yAt(layer.top[i], i));
      if (h > bestH) {{ bestH = h; bestBin = i; }}
    }}
    if (bestH < 16) continue;
    const midY = (yAt(layer.top[bestBin], bestBin) + yAt(layer.bottom[bestBin], bestBin)) / 2;
    ctx.fillStyle = '#fff';
    ctx.globalAlpha = 0.85;
    ctx.font = '10px Palatino, Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(layer.group.name, xAtBin(bestBin), midY + 4);
    ctx.globalAlpha = 1;
  }}
}}

function syncTrajectoryGitToggle() {{
  const toggle = document.getElementById('trajectoryGitToggle');
  const note = document.getElementById('trajectoryGitNote');
  if (toggle) {{
    for (const btn of toggle.querySelectorAll('button[data-include-git]')) {{
      const wantsGit = btn.dataset.includeGit === 'true';
      btn.classList.toggle('active', wantsGit === includeGitInStacked);
      btn.setAttribute('aria-pressed', wantsGit === includeGitInStacked ? 'true' : 'false');
    }}
  }}
  if (note) {{
    note.textContent = includeGitInStacked
      ? 'With git: shapes include the git layer.'
      : 'Without git: the git layer is removed and the remaining categories are re-normalized to 100% in each bin.';
  }}
}}

function renderStackedPanels() {{
  const container = document.getElementById('stackedPanels');
  if (!container) return;
  container.innerHTML = '';

  const interventionOrder = ['authorization', 'steering', 'closeout'];

  for (const m of ALL_MODELS) {{
    const wrap = document.createElement('div');
    wrap.className = 'chart-wrapper';
    const cls = tagClass[m] || '';
    const title = MODEL_NAMES[m];
    const guidedRate = D.resolve_rate[m];
    const rawN = D.raw_single_model_counts?.[m] ?? 0;
    const analyzedN = D.analyzed_counts?.[m] ?? D.num_trajs[m] ?? 0;
    const benchmarkModel = D.benchmark?.pair_for_pi_model?.[m] || null;
    const benchmarkRate = benchmarkModel ? D.benchmark?.resolve_rate?.[benchmarkModel] : null;
    const benchmarkN = benchmarkModel ? D.benchmark?.num_trajs?.[benchmarkModel] : null;
    const benchmarkName = benchmarkModel ? D.benchmark?.model_display_names?.[benchmarkModel] || benchmarkModel : null;

    const benchmarkSub = benchmarkModel && benchmarkRate != null
      ? `${{benchmarkN}} trajectories · ${{benchmarkRate.toFixed(1)}}% resolved · ${{benchmarkName}}`
      : 'benchmark baseline unavailable';
    const guidedSub = guidedRate != null
      ? `${{rawN}} single-model sessions · ${{analyzedN}} analyzed · ${{guidedRate.toFixed(1)}}% completed cleanly`
      : `${{rawN}} single-model sessions · ${{analyzedN}} analyzed`;

    wrap.innerHTML =
      `<div class="stacked-panel-header">` +
        `<span class="model-tag ${{cls}}">${{title}}</span>` +
      `</div>` +
      `<div class="stacked-pair-row">` +
        `<div class="side-label">benchmark (agent alone)<span class="panel-subhead">${{benchmarkSub}}</span></div>` +
        `<canvas id="stacked_benchmark_${{m}}" height="138"></canvas>` +
      `</div>` +
      `<div class="stacked-pair-row">` +
        `<div class="side-label">maintainer-guided<span class="panel-subhead">${{guidedSub}}</span></div>` +
        `<canvas id="stacked_guided_${{m}}" height="152"></canvas>` +
      `</div>`;
    container.appendChild(wrap);

    const benchmarkPhase = D.benchmark?.avg_phase?.[benchmarkModel];
    const benchmarkFirstEditMarker = D.benchmark?.first_edit_markers?.[benchmarkModel];
    const benchmarkLastEditMarker = D.benchmark?.last_edit_markers?.[benchmarkModel];
    if (benchmarkPhase) {{
      drawStackedArea(`stacked_benchmark_${{m}}`, benchmarkPhase, {{
        markers: [benchmarkFirstEditMarker, benchmarkLastEditMarker].filter(Boolean),
        includeGit: includeGitInStacked,
      }});
    }}

    const interventions = interventionOrder
      .map(key => D.intervention_markers?.[m]?.[key])
      .filter(Boolean);
    const guidedFirstEditMarker = D.first_edit_markers?.[m];
    const guidedLastEditMarker = D.last_edit_markers?.[m];
    drawStackedArea(`stacked_guided_${{m}}`, D.avg_phase[m], {{
      markers: [guidedFirstEditMarker, ...interventions, guidedLastEditMarker].filter(Boolean),
      includeGit: includeGitInStacked,
    }});
  }}
}}

(function() {{
  const toggle = document.getElementById('trajectoryGitToggle');
  if (toggle) {{
    toggle.addEventListener('click', (ev) => {{
      const btn = ev.target.closest('button[data-include-git]');
      if (!btn) return;
      includeGitInStacked = btn.dataset.includeGit === 'true';
      syncTrajectoryGitToggle();
      renderStackedPanels();
    }});
  }}
  syncTrajectoryGitToggle();
  renderStackedPanels();
}})();

</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/pi-mono")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_EXACT_MODELS))
    parser.add_argument(
        "--session-name-prefix",
        action="append",
        default=["Issue:"],
        help="Keep only sessions whose final non-empty session_info.name starts with this prefix. Repeatable.",
    )
    parser.add_argument("--benchmark-data-root", default="data")
    parser.add_argument("--output", "-o", default="docs/pi-analytics.html")
    parser.add_argument(
        "--merge-models",
        action="append",
        default=[],
        help=(
            "Merge several input model keys into one synthetic key for the "
            "trajectory-shape panels. Format: SRC1,SRC2=KEY:LABEL. Repeatable."
        ),
    )
    args = parser.parse_args()

    merge_specs = _parse_merge_specs(args.merge_models)

    data_root = Path(args.data_root)
    payload = build_payload(
        data_root,
        models=args.models,
        session_name_prefixes=args.session_name_prefix,
        benchmark_data_root=Path(args.benchmark_data_root),
        merge_specs=merge_specs,
    )
    html = render_html(payload)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    _write_sidecar_intent_csvs(out, payload)
    print(f"Wrote {out}")
    counts = " ".join(f"{m}={payload['num_trajs'][m]}" for m in payload["models"])
    print(counts)


if __name__ == "__main__":
    main()
