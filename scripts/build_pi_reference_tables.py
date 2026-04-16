#!/usr/bin/env python3
"""
Build a reference HTML for Pi transcript intent analysis.

This is the Pi-harness counterpart to the original reference tables page: same
intent taxonomy, different step parser and completion semantics.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_pi import aggregate
from analysis_pi.models import (
    INTENT_DESCRIPTIONS,
    INTENT_TO_HIGH_LEVEL,
    HIGH_LEVEL_COLORS,
    build_model_registry,
)
from analysis_pi.orchestrate import process_all
from analysis_pi.session_filter import DEFAULT_EXACT_MODELS, SessionFilter, collect_filtered_paths


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


def _html_table(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{h}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _fmt(v) -> str:
    if v is None:
        return "&mdash;"
    if isinstance(v, float):
        return f"{v:.1f}"
    return html.escape(str(v))


def _pct_bar(pct: float, color: str) -> str:
    return (
        '<div class="pct-wrap">'
        f'<div class="pct-bar" style="width:{pct:.1f}%;background:{color}"></div>'
        f'<span class="pct-label">{pct:.1f}%</span>'
        '</div>'
    )


def _intent_counters(results: dict[str, list], models: list[str]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    counts = {m: {} for m in models}
    totals = {m: 0 for m in models}
    for m in models:
        c = {}
        total = 0
        for row in results[m]:
            total += row.steps
            for intent, n in row.base_intent_counts.items():
                c[intent] = c.get(intent, 0) + n
        counts[m] = c
        totals[m] = total
    return counts, totals


def _render_detailed_classification_section(results: dict[str, list], models: list[str]) -> str:
    counts, totals = _intent_counters(results, models)
    category_order = ["read", "search", "reproduce", "edit", "verify", "git", "housekeeping", "failed", "other"]

    header = (
        '<table><thead><tr>'
        '<th>category</th><th>intent</th><th>description</th>' +
        ''.join(f'<th>{html.escape(m)}</th>' for m in models) +
        '</tr></thead><tbody>'
    )
    rows: list[str] = []
    for cat in category_order:
        intents = [
            intent for intent, high in INTENT_TO_HIGH_LEVEL.items()
            if high == cat and any(counts[m].get(intent, 0) > 0 for m in models)
        ]
        intents.sort(key=lambda intent: (-sum(counts[m].get(intent, 0) for m in models), intent))
        if not intents:
            continue
        rows.append(
            f'<tr class="cat-row"><td colspan="{3 + len(models)}">'
            f'<span class="cat-dot" style="background:{HIGH_LEVEL_COLORS.get(cat, "#999")}"></span>'
            f'{html.escape(cat)}</td></tr>'
        )
        for intent in intents:
            cells = []
            for m in models:
                n = counts[m].get(intent, 0)
                pct = (n / totals[m] * 100) if totals[m] else 0.0
                cells.append(f'<td>{n} <span class="muted-inline">({pct:.1f}%)</span></td>')
            rows.append(
                '<tr>'
                f'<td></td>'
                f'<td><code>{html.escape(intent)}</code></td>'
                f'<td>{html.escape(INTENT_DESCRIPTIONS.get(intent, ""))}</td>'
                + ''.join(cells) +
                '</tr>'
            )
    return (
        '<section><h2>3b. Detailed classification breakdown</h2>'
        '<p class="note">This is the detailed per-intent table from the Pi copy. Each cell shows <strong>count</strong> and <strong>share of all classified tool steps for that model</strong>. It is grouped by the same high-level taxonomy as the original reference tables, so you can inspect exactly what is inside categories like <em>cleanup</em> and <em>other</em>.</p>'
        + header + ''.join(rows) + '</tbody></table></section>'
    )


def _render_cleanup_decomposition_section(results: dict[str, list], models: list[str]) -> str:
    counts, totals = _intent_counters(results, models)
    cleanup_intents = [
        "git-status-log",
        "git-diff",
        "git-stash",
        "file-cleanup",
        "create-documentation",
        "start-service",
        "install-deps",
        "check-tool-exists",
    ]
    rows: list[list[str]] = []
    for intent in cleanup_intents:
        row = [html.escape(INTENT_TO_HIGH_LEVEL[intent]), f'<code>{html.escape(intent)}</code>', html.escape(INTENT_DESCRIPTIONS.get(intent, ""))]
        for m in models:
            n = counts[m].get(intent, 0)
            pct = (n / totals[m] * 100) if totals[m] else 0.0
            row.append(f'{n} <span class="muted-inline">({pct:.1f}%)</span>')
        rows.append(row)

    summary_rows = []
    for label, members in {
        'git total': [i for i in cleanup_intents if INTENT_TO_HIGH_LEVEL[i] == 'git'],
        'housekeeping total': [i for i in cleanup_intents if INTENT_TO_HIGH_LEVEL[i] == 'housekeeping'],
        'cleanup phase total': cleanup_intents,
    }.items():
        row = ['summary', html.escape(label), '']
        for m in models:
            n = sum(counts[m].get(i, 0) for i in members)
            pct = (n / totals[m] * 100) if totals[m] else 0.0
            row.append(f'{n} <span class="muted-inline">({pct:.1f}%)</span>')
        summary_rows.append(row)

    return (
        '<section><h2>4b. Cleanup decomposition</h2>'
        '<p class="note">In the inherited phase schema, <strong>cleanup = git + housekeeping</strong>. For Pi transcripts this phase is mostly git workflow, not literal cleanup. This table makes that explicit.</p>'
        + _html_table(["high-level", "intent", "description"] + [html.escape(m) for m in models], summary_rows + rows)
        + '</section>'
    )


def render_html(results: dict[str, list], raw_single_counts: dict[str, int]) -> str:
    models = sorted(
        results.keys(),
        key=lambda m: (-raw_single_counts.get(m, 0), -len(results.get(m, [])), m),
    )
    meta = aggregate.metadata_summary(results)
    base = aggregate.base_intent_frequencies(results)
    high = aggregate.high_level_frequencies(results)
    phases = aggregate.phase_frequencies(results)
    verify = aggregate.verify_outcomes(results)
    seq = aggregate.sequence_labels(results)
    markers = aggregate.structural_markers(results)
    step_dist = aggregate.step_distribution(results)
    wd = aggregate.work_done_vs_completed(results)
    model_meta = build_model_registry(models)

    model_tags = " ".join(
        f'<span class="tag" style="border-color:{model_meta[m]["color"]};color:{model_meta[m]["color"]}">{html.escape(model_meta[m]["label"])}</span>'
        for m in models
    )

    sections: list[str] = []

    # 1. Metadata summary
    rows = []
    for m in models:
        mm = meta[m]
        rows.append([
            html.escape(model_meta[m]["label"]),
            _fmt(raw_single_counts.get(m, 0)),
            _fmt(mm["n"]),
            _fmt(mm["avg"]),
            _fmt(mm["median"]),
            _fmt(mm["p25"]),
            _fmt(mm["p75"]),
            _fmt(mm["min"]),
            _fmt(mm["max"]),
            _fmt(mm["completed"]),
            html.escape(mm["completion_rate"]),
        ])
    sections.append(
        '<section><h2>1. Session metadata</h2>'
        '<p class="note">Strict single-model purity is determined from both <code>assistant.message.model</code> and <code>model_change.modelId</code>. “single-model sessions” is the raw eligible count; “analyzed sessions” is the subset with tool-call trajectories that can actually be classified.</p>'
        + _html_table(
            ["model", "single-model sessions", "analyzed sessions", "avg steps", "median", "p25", "p75", "min", "max", "completed", "completion rate"],
            rows,
        )
        + '</section>'
    )

    # 2. Exit statuses
    exit_rows = []
    for m in models:
        exits = meta[m]["exits"]
        summary = "<br>".join(f"<code>{html.escape(k)}</code>: {v}" for k, v in exits.items())
        exit_rows.append([html.escape(model_meta[m]["label"]), summary])
    sections.append(
        '<section><h2>2. End states</h2>'
        '<p class="note">These are transcript-level end states, derived from the final assistant stop reason recorded in the session.</p>'
        + _html_table(["model", "end-state counts"], exit_rows)
        + '</section>'
    )

    # 3. High-level action frequencies
    headers = ["model"] + ["read", "search", "reproduce", "edit", "verify", "git", "housekeeping", "failed", "other"]
    rows = []
    for m in models:
        props = high["proportions"].get(m, {})
        rows.append([html.escape(model_meta[m]["label"])] + [f"{props.get(k, 0) * 100:.1f}%" for k in headers[1:]])
    sections.append(
        '<section><h2>3. High-level category mix</h2>'
        '<p class="note">Shares of all classified tool steps by high-level category.</p>'
        + _html_table(headers, rows)
        + '</section>'
    )

    sections.append(_render_detailed_classification_section(results, models))
    sections.append(_render_cleanup_decomposition_section(results, models))

    # 4. Phase mix
    phase_headers = ["model"] + list(next(iter(phases.values())).keys()) if phases else ["model"]
    phase_rows = []
    for m in models:
        phase_rows.append([html.escape(model_meta[m]["label"])] + [f"{phases[m].get(p, 0):.1f}%" for p in phase_headers[1:]])
    sections.append(
        '<section><h2>4. Phase mix</h2>'
        '<p class="note">Phase grouping reused from the original analysis: understand = read + search, cleanup = git + housekeeping.</p>'
        + _html_table(phase_headers, phase_rows)
        + '</section>'
    )

    # 5. Verify outcomes
    verify_rows = []
    for m in models:
        v = verify[m]
        verify_rows.append([
            html.escape(model_meta[m]["label"]),
            _fmt(v["pass"]),
            _fmt(v["fail"]),
            _fmt(v["unknown"]),
            _fmt(v["total"]),
            html.escape(v["pass_rate"]),
        ])
    sections.append(
        '<section><h2>5. Verify outcomes</h2>'
        '<p class="note">Verify pass/fail uses the original deterministic parser over bash observations. This mainly applies to test/build commands run through Pi’s <code>bash</code> tool.</p>'
        + _html_table(["model", "pass", "fail", "unknown", "total verify steps", "pass rate"], verify_rows)
        + '</section>'
    )

    # 6. Most common base intents
    top_intents = base["top_intents"][:18]
    rows = []
    for intent in top_intents:
        row = [f'<code>{html.escape(intent)}</code>']
        for m in models:
            pct = base["proportions"].get(m, {}).get(intent, 0) * 100
            row.append(f"{pct:.1f}%")
        rows.append(row)
    sections.append(
        '<section><h2>6. Most common base intents</h2>'
        '<p class="note">Same base-intent taxonomy as the SWE-Agent analysis, but applied to Pi tool calls by mapping <code>read</code>/<code>edit</code>/<code>write</code>/<code>bash</code> into equivalent intent semantics.</p>'
        + _html_table(["intent"] + [html.escape(model_meta[m]["label"]) for m in models], rows)
        + '</section>'
    )

    # 7. Sequence labels
    seq_rows = []
    for label in seq["all_labels"][:20]:
        row = [f'<code>{html.escape(label)}</code>']
        for m in models:
            row.append(_fmt(seq["counts"].get(m, {}).get(label, 0)))
        seq_rows.append(row)
    sections.append(
        '<section><h2>7. Sequence-layer labels</h2>'
        '<p class="note">These are second-pass labels derived from nearby history, such as verify-after-edit or first-all-pass after the last source edit.</p>'
        + _html_table(["label"] + [html.escape(model_meta[m]["label"]) for m in models], seq_rows)
        + '</section>'
    )

    # 8. Structural markers
    marker_order = [
        ("first_edit", "first edit"),
        ("last_edit", "last edit"),
        ("first_verify", "first verify"),
        ("first_verify_pass", "first verify pass"),
        ("submit", "finish / submit"),
    ]
    rows = []
    for key, label in marker_order:
        for m in models:
            info = markers[key][m]
            rows.append([
                html.escape(model_meta[m]["label"]),
                label,
                _fmt(info["median"]),
                _fmt(info["p25"]),
                _fmt(info["p75"]),
                _fmt(info["completed_median"]),
                _fmt(info["incomplete_median"]),
            ])
    sections.append(
        '<section><h2>8. Structural markers</h2>'
        '<p class="note">Marker positions are measured as a percentage of session length. “Completed” and “incomplete” split by clean session completion, not correctness.</p>'
        + _html_table(
            ["model", "marker", "median %", "p25", "p75", "completed median", "incomplete median"],
            rows,
        )
        + '</section>'
    )

    # 9. Work done vs completion
    wd_rows = []
    for m in models:
        info = wd[m]
        n = max(info["n"], 1)
        wd_rows.append([
            html.escape(model_meta[m]["label"]),
            _fmt(info["wd_completed"]),
            _fmt(info["wd_incomplete"]),
            _fmt(info["no_wd_completed"]),
            _fmt(info["no_wd_incomplete"]),
            f"{info['wd_completed'] / n * 100:.1f}%",
        ])
    sections.append(
        '<section><h2>9. Work done vs completion</h2>'
        '<p class="note"><strong>work done</strong> keeps the original meaning: the transcript reaches a verify pass after its last source edit. Here we compare that against whether the session ended cleanly.</p>'
        + _html_table(
            ["model", "wd + completed", "wd + incomplete", "no wd + completed", "no wd + incomplete", "wd+completed rate"],
            wd_rows,
        )
        + '</section>'
    )

    # 10. Step distribution
    dist_rows = []
    for m in models:
        bins = step_dist[m]
        dist_rows.append([
            html.escape(model_meta[m]["label"]),
            "<br>".join(f"{k}-{k+4}: {v}" for k, v in bins.items()),
        ])
    sections.append(
        '<section><h2>10. Step-count distribution</h2>'
        + _html_table(["model", "5-step bins"], dist_rows)
        + '</section>'
    )

    # Sidebar summary cards
    cards = []
    for m in models:
        mm = meta[m]
        pct = float(mm["completion_rate"].rstrip("%")) if mm["completion_rate"].endswith("%") else 0.0
        cards.append(
            '<div class="card">'
            f'<div class="card-title" style="color:{model_meta[m]["color"]}">{html.escape(model_meta[m]["label"])}</div>'
            f'<div class="card-stat">{mm["n"]} sessions</div>'
            f'{_pct_bar(pct, model_meta[m]["color"])}'
            f'<div class="card-sub">median {mm["median"]} tool steps</div>'
            '</div>'
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Pi transcript reference tables</title>
  <style>
    :root {{
      --bg: #fffff8;
      --text: #333;
      --muted: #666;
      --border: #ddd;
      --panel: #fffef7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Palatino Linotype', Palatino, Georgia, serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 24px 22px 60px; }}
    h1 {{ font-size: 28px; font-weight: 400; margin: 0 0 4px 0; }}
    .subtitle {{ color: var(--muted); font-style: italic; margin-bottom: 18px; }}
    .lede {{ color: var(--text); max-width: 80ch; margin: 0 0 22px 0; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
    .tag {{ border: 1px solid var(--border); border-radius: 999px; padding: 4px 10px; font-size: 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 28px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); padding: 12px 14px; border-radius: 8px; }}
    .card-title {{ font-style: italic; margin-bottom: 6px; }}
    .card-stat {{ font-size: 22px; margin-bottom: 8px; }}
    .card-sub {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
    .pct-wrap {{ position: relative; height: 20px; background: #f1efe6; border-radius: 999px; overflow: hidden; }}
    .pct-bar {{ position: absolute; left: 0; top: 0; bottom: 0; opacity: 0.85; }}
    .pct-label {{ position: absolute; right: 8px; top: 1px; font-size: 12px; color: #444; }}
    .muted-inline {{ color: var(--muted); font-size: 11px; }}
    .cat-row td {{ background: #faf7ed; font-style: italic; color: #555; }}
    .cat-dot {{ display:inline-block; width:8px; height:8px; border-radius:1px; margin-right:8px; vertical-align:middle; }}
    section {{ margin: 34px 0 0 0; padding-top: 18px; border-top: 1px solid #e6e0d5; }}
    h2 {{ font-size: 18px; font-style: italic; font-weight: 400; margin: 0 0 10px 0; }}
    .note {{ color: var(--muted); font-size: 13px; max-width: 90ch; margin: 0 0 14px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: var(--panel); }}
    caption {{ text-align: left; color: var(--muted); margin-bottom: 8px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #ece7dc; vertical-align: top; text-align: left; }}
    th {{ font-size: 12px; color: var(--muted); font-style: italic; font-weight: 400; position: sticky; top: 0; background: #fbf8ee; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; }}
    a {{ color: inherit; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Pi transcript reference tables</h1>
    <div class="subtitle">Same intent taxonomy as the SWE-Agent analysis, adapted to Pi tool calls and currently filtered to strict single-model issue sessions</div>
    <p class="lede">This page is the Pi-session analogue of the original reference tables. It keeps the same read/search/reproduce/edit/verify taxonomy, but classifies Pi’s <code>read</code>, <code>edit</code>, <code>write</code>, <code>bash</code>, and auxiliary tools into that shared scheme.</p>
    <div class="tags">{model_tags}</div>
    <div class="cards">{''.join(cards)}</div>
    {''.join(sections)}
  </div>
</body>
</html>"""


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
    parser.add_argument("--output", "-o", default="docs/pi-references.html")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    session_filter = SessionFilter(
        allowed_models=args.models,
        require_single_model=True,
        session_name_prefixes=args.session_name_prefix,
    )
    allowed_paths, raw_counts, _ = collect_filtered_paths(data_root, session_filter)
    results = process_all(data_root, models=args.models)
    results = _filter_results_to_paths(results, allowed_paths)
    for model in args.models:
        print(f"  {model}: {raw_counts.get(model, 0)} strict single-model sessions, {len(results.get(model, []))} analyzed")

    html_out = render_html(results, {m: raw_counts.get(m, 0) for m in args.models})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
