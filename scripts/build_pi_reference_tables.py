#!/usr/bin/env python3
"""
Build a reference HTML for Pi transcript intent analysis.

This is the Pi-harness counterpart to the original reference tables page: same
intent taxonomy, different step parser and completion semantics.
"""

from __future__ import annotations

import argparse
import html
import re
import statistics
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
from analysis_pi.resolved import ModelResolutionStats, compute_resolution_by_model
from analysis_pi.session_filter import DEFAULT_EXACT_MODELS, SessionFilter, collect_filtered_paths
from analysis_pi.user_messages import CLASS_ORDER, analyze_user_messages


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


def _heat_strip(values: list[float], color: str) -> str:
    cells = []
    for i, pct in enumerate(values):
        alpha = max(0.06, min(0.92, pct / 100.0))
        title = f"{i * 5}-{i * 5 + 5}% of trajectory: {pct:.1f}% of sessions"
        cells.append(
            f'<span class="heat-cell" title="{html.escape(title)}" '
            f'style="background:{color};opacity:{alpha:.3f}"></span>'
        )
    return '<div class="heat-strip">' + ''.join(cells) + '</div>'


def _render_intervention_macro_section(user_data: dict, models: list[str], model_meta: dict[str, dict[str, str]]) -> str:
    macro_defs = [
        {
            "key": "analysis_start",
            "label": "analysis starts",
            "members": ["task_brief"],
            "color": "#5a7d9a",
            "desc": "first task brief / issue-analysis framing message",
        },
        {
            "key": "work_start",
            "label": "work starts (authorized)",
            "members": ["authorize_work"],
            "color": "#4a8a5a",
            "desc": "first explicit authorization to implement or fix",
        },
        {
            "key": "solution_steering",
            "label": "solution steering",
            "members": ["solution_steer", "evidence_or_repro", "qa_or_critique", "validation_request"],
            "color": "#b56a50",
            "desc": "first substantive maintainer steering / evidence / critique / validation turn after the brief",
        },
        {
            "key": "workflow_closeout",
            "label": "workflow closeout",
            "members": ["workflow_closeout"],
            "color": "#8a6a9a",
            "desc": "first commit / push / changelog / comment / close-issue style instruction",
        },
    ]

    def collect_scope_messages(scope: str | None) -> tuple[list[dict], int]:
        if scope is None:
            msgs = []
            for model in models:
                for label in CLASS_ORDER:
                    msgs.extend(user_data["per_model"][model]["classes"][label]["messages"])
            return msgs, user_data["total_sessions"]
        pdata = user_data["per_model"][scope]
        msgs = []
        for label in CLASS_ORDER:
            msgs.extend(pdata["classes"][label]["messages"])
        return msgs, pdata["num_sessions"]

    def scope_stats(scope: str | None) -> dict[str, dict]:
        rows = {}
        _, num_sessions = collect_scope_messages(scope)
        by_label = {}
        for label in CLASS_ORDER:
            if scope is None:
                pool = []
                for model in models:
                    pool.extend(user_data["per_model"][model]["classes"][label]["messages"])
            else:
                pool = list(user_data["per_model"][scope]["classes"][label]["messages"])
            by_label[label] = pool
        for macro in macro_defs:
            recs = []
            for label in macro["members"]:
                recs.extend(by_label[label])
            per_path: dict[str, list[dict]] = {}
            for rec in recs:
                per_path.setdefault(rec["path"], []).append(rec)
            firsts = []
            for path, msgs in per_path.items():
                msgs = sorted(msgs, key=lambda m: (m["message_index"], m["progress_pct"]))
                firsts.append(msgs[0]["progress_pct"])
            firsts.sort()
            if firsts:
                med = round(statistics.median(firsts), 1)
                if len(firsts) >= 2:
                    p25 = round(statistics.quantiles(firsts, n=4)[0], 1)
                    p75 = round(statistics.quantiles(firsts, n=4)[2], 1)
                else:
                    p25 = p75 = round(firsts[0], 1)
            else:
                med = p25 = p75 = None
            rows[macro["key"]] = {
                "session_count": len(per_path),
                "session_pct": round(len(per_path) / num_sessions * 100, 1) if num_sessions else 0.0,
                "median": med,
                "p25": p25,
                "p75": p75,
            }
        return rows

    def chart(scope_label: str, color: str, stats: dict[str, dict]) -> str:
        row_html = []
        for macro in macro_defs:
            s = stats[macro["key"]]
            if s["median"] is None:
                track = '<div class="marker-track"></div>'
                summary = '&mdash;'
            else:
                width = max(0.0, (s["p75"] or 0) - (s["p25"] or 0))
                track = (
                    '<div class="marker-track">'
                    '<span class="marker-grid" style="left:0%"></span>'
                    '<span class="marker-grid" style="left:25%"></span>'
                    '<span class="marker-grid" style="left:50%"></span>'
                    '<span class="marker-grid" style="left:75%"></span>'
                    '<span class="marker-grid" style="left:100%"></span>'
                    f'<span class="marker-iqr" style="left:{s["p25"]}%;width:{width}%;background:{macro["color"]}"></span>'
                    f'<span class="marker-dot" style="left:{s["median"]}%;background:{macro["color"]}"></span>'
                    '</div>'
                )
                summary = f'{s["session_pct"]:.1f}% sessions · first @ {s["median"]:.1f}%'
            row_html.append(
                '<div class="marker-row">'
                f'<div class="marker-label"><code>{html.escape(macro["label"])}</code><div class="muted-inline">{html.escape(macro["desc"])}</div></div>'
                + track +
                f'<div class="marker-summary">{summary}</div>'
                '</div>'
            )
        return (
            '<div class="marker-card">'
            f'<div class="card-title" style="color:{color}">{html.escape(scope_label)}</div>'
            '<div class="marker-axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>'
            + ''.join(row_html) +
            '</div>'
        )

    parts = [
        '<section><h2>11. Maintainer intervention markers</h2>'
        '<p class="note">This is the higher-level phase view derived from the 7 user-message classes. '
        'For each model, the marker shows the <strong>median first occurrence</strong> of that intervention type as a percentage of trajectory progress; '
        'the horizontal bar shows the interquartile range (p25–p75). '
        'This is the user-message analogue of structural markers like first edit / last edit.</p>'
        '<p class="note"><strong>solution steering</strong> here intentionally lumps together <code>solution_steer</code>, '
        '<code>evidence_or_repro</code>, <code>qa_or_critique</code>, and <code>validation_request</code>.</p>'
    ]
    for model in models:
        parts.append(chart(model_meta[model]['label'], model_meta[model]['color'], scope_stats(model)))

    headers = ['scope'] + [m['label'] for m in macro_defs]
    table_rows = []
    def cell(s: dict) -> str:
        if s['median'] is None:
            return '&mdash;'
        return f"{s['session_pct']:.1f}% sessions<br><span class='muted-inline'>median {s['median']:.1f}% · [{s['p25']:.1f},{s['p75']:.1f}]</span>"
    for model in models:
        sstats = scope_stats(model)
        table_rows.append([html.escape(model_meta[model]['label'])] + [cell(sstats[m['key']]) for m in macro_defs])
    parts.append(
        _html_table(headers, table_rows)
    )
    parts.append('</section>')
    return ''.join(parts)


def _render_user_message_sections(user_data: dict, models: list[str], model_meta: dict[str, dict[str, str]]) -> str:
    total_sessions = user_data["total_sessions"]
    total_messages = user_data["total_messages"]
    class_desc = user_data["class_descriptions"]

    # Overall summary table.
    rows = []
    for label in CLASS_ORDER:
        stats = user_data["overall"][label]
        rows.append([
            f'<code>{html.escape(label)}</code>',
            html.escape(class_desc[label]),
            _fmt(stats["message_count"]),
            f'{stats["message_pct"]:.1f}%',
            _fmt(stats["session_count"]),
            f'{stats["session_pct"]:.1f}%',
            _fmt(stats["first_progress_median"]),
            _fmt(stats["first_progress_p25"]),
            _fmt(stats["first_progress_p75"]),
        ])
    summary = (
        '<section><h2>12. User message classes</h2>'
        '<p class="note">These counts are computed over the raw filtered issue sessions, not just the classified tool-step subset. '
        'Messages are assigned a single primary class using a deterministic, dataset-tuned rule set. '
        'The timing columns use the same trajectory-normalised 0-100% progress scale as the stacked trajectory-shape charts: '
        'for each user message, we count how many assistant tool calls have already happened in that session.</p>'
        f'<p class="note"><strong>{total_messages}</strong> user messages across <strong>{total_sessions}</strong> strict single-model issue sessions.</p>'
        + _html_table(
            [
                'class', 'description', 'messages', '% of messages', 'sessions', '% of sessions',
                'median first %', 'p25', 'p75',
            ],
            rows,
        )
        + '</section>'
    )

    # Per-model timing strips.
    per_model_parts = [
        '<section><h2>13. User intervention timing by model</h2>'
        '<p class="note">Each heat strip is aligned to the same 20-bin trajectory-normalised timeline used by the analytics stacked-area charts. '
        'A darker cell means more sessions for that model had at least one message of that class in that 5% trajectory bin. '
        'The summary percentages at right describe class prevalence and median first-occurrence position.</p>'
    ]
    for model in models:
        pdata = user_data["per_model"].get(model, {})
        per_model_parts.append(
            f'<h3 style="margin:22px 0 8px 0;color:{model_meta[model]["color"]};font-style:italic;font-weight:400">'
            f'{html.escape(model_meta[model]["label"])} '
            f'<span class="muted-inline">({pdata.get("num_messages", 0)} user messages across {pdata.get("num_sessions", 0)} sessions)</span>'
            '</h3>'
        )
        rows_html = []
        for label in CLASS_ORDER:
            stats = pdata["classes"][label]
            rows_html.append(
                '<tr>'
                f'<td><code>{html.escape(label)}</code></td>'
                f'<td>{html.escape(class_desc[label])}</td>'
                f'<td>{stats["message_count"]} <span class="muted-inline">({stats["message_pct"]:.1f}%)</span></td>'
                f'<td>{stats["session_count"]} <span class="muted-inline">({stats["session_pct"]:.1f}%)</span></td>'
                f'<td>{_fmt(stats["first_progress_median"])} <span class="muted-inline">[{_fmt(stats["first_progress_p25"])}–{_fmt(stats["first_progress_p75"])}]</span></td>'
                f'<td>{_heat_strip(stats["bin_session_pct"], model_meta[model]["color"])}</td>'
                '</tr>'
            )
        per_model_parts.append(
            '<table><thead><tr>'
            '<th>class</th><th>description</th><th>messages</th><th>sessions</th><th>first occurrence</th><th>20-bin prevalence</th>'
            '</tr></thead><tbody>'
            + ''.join(rows_html) +
            '</tbody></table>'
        )
    per_model_parts.append('</section>')

    # All raw messages grouped by class and then model.
    detail_parts = [
        '<section><h2>14. All user messages by class</h2>'
        '<p class="note">Every classified user message in the filtered issue subset. '
        'Each entry records the session, user-turn index, and trajectory progress when that interruption happened.</p>'
    ]
    for label in CLASS_ORDER:
        overall = user_data["overall"][label]
        detail_parts.append(
            '<details class="msg-group" open>'
            f'<summary><code>{html.escape(label)}</code> — {overall["message_count"]} messages across {overall["session_count"]} sessions '
            f'({overall["session_pct"]:.1f}% of sessions)</summary>'
            f'<p class="note" style="margin-top:8px">{html.escape(class_desc[label])}</p>'
        )
        for model in models:
            stats = user_data["per_model"][model]["classes"][label]
            if not stats["messages"]:
                continue
            detail_parts.append(
                f'<details class="msg-subgroup"><summary style="color:{model_meta[model]["color"]}">'
                f'{html.escape(model_meta[model]["label"])} — {stats["message_count"]} messages in {stats["session_count"]} sessions'
                '</summary><ul class="msg-list">'
            )
            for msg in stats["messages"]:
                detail_parts.append(
                    '<li>'
                    f'<div class="msg-meta"><strong>{html.escape(msg["session_name"] or Path(msg["path"]).name)}</strong> '
                    f'<span class="muted-inline">turn {msg["message_index"]} · {msg["progress_pct"]:.1f}% through trajectory · {html.escape(Path(msg["path"]).name)}</span></div>'
                    f'<div class="msg-text">{html.escape(msg["text"] or "<empty>")}</div>'
                    '</li>'
                )
            detail_parts.append('</ul></details>')
        detail_parts.append('</details>')
    detail_parts.append('</section>')

    return summary + ''.join(per_model_parts) + ''.join(detail_parts)


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


_KIND_LABELS = {
    "push": "git push",
    "gh_close": "gh issue close",
    "gh_merge": "gh pr merge",
    "gh_comment": "triage comment",
    "user_close": "maintainer close",
}

_KIND_DESCRIPTIONS = {
    "push": "agent pushed the fix to the remote",
    "gh_close": "agent ran <code>gh issue close</code>",
    "gh_merge": "agent ran <code>gh pr merge</code>",
    "gh_comment": "agent posted a triage comment after the maintainer asked to comment/close",
    "user_close": "maintainer gave a terminal close/triage instruction; the agent didn't ship a shell action but the task was decided",
}


def _render_resolution_section(stats: dict, models: list[str], model_meta: dict[str, dict[str, str]]) -> str:
    """Scheme-3 per-issue resolution rate with a broadened success rule."""
    rows: list[list[str]] = []
    max_rate = 0.0
    for m in models:
        s = stats.get(m)
        if s is None:
            continue
        if s.resolve_rate > max_rate:
            max_rate = s.resolve_rate

    # Bar chart rows.
    bars: list[str] = []
    for m in models:
        s = stats.get(m)
        if s is None:
            continue
        color = model_meta[m]["color"]
        width = (s.resolve_rate / 100.0) * 100.0
        bars.append(
            '<div class="resolve-row">'
            f'<div class="resolve-label" style="color:{color}">{html.escape(model_meta[m]["label"])}</div>'
            '<div class="resolve-track">'
            f'<div class="resolve-bar" style="width:{width:.1f}%;background:{color}"></div>'
            '</div>'
            f'<div class="resolve-summary">'
            f'<strong>{s.resolve_rate:.1f}%</strong> '
            f'<span class="muted-inline">({s.n_issues_resolved}/{s.n_issues_attempted} issues · {s.n_sessions} sessions)</span>'
            '</div>'
            '</div>'
        )

    # Table with kind breakdown.
    kinds_order = ["push", "gh_close", "gh_merge", "gh_comment", "user_close"]
    headers = [
        "model",
        "sessions (filtered)",
        "distinct issues",
        "resolved",
        "resolve rate",
    ] + [_KIND_LABELS[k] for k in kinds_order]
    for m in models:
        s = stats.get(m)
        if s is None:
            continue
        kc = s.kind_counts
        row = [
            f'<span style="color:{model_meta[m]["color"]}">{html.escape(model_meta[m]["label"])}</span>',
            str(s.n_sessions),
            str(s.n_issues_attempted),
            str(s.n_issues_resolved),
            _pct_bar(s.resolve_rate, model_meta[m]["color"]),
        ] + [str(kc.get(k, 0)) for k in kinds_order]
        rows.append(row)

    # Totals.
    total_sessions = sum(s.n_sessions for s in stats.values())
    total_issues = sum(s.n_issues_attempted for s in stats.values())
    total_resolved = sum(s.n_issues_resolved for s in stats.values())
    total_rate = (total_resolved / total_issues * 100.0) if total_issues else 0.0
    total_kinds = {k: sum(s.kind_counts.get(k, 0) for s in stats.values()) for k in kinds_order}
    rows.append(
        [
            "<strong>total</strong>",
            f"<strong>{total_sessions}</strong>",
            f"<strong>{total_issues}</strong>",
            f"<strong>{total_resolved}</strong>",
            f"<strong>{total_rate:.1f}%</strong>",
        ]
        + [f"<strong>{total_kinds[k]}</strong>" for k in kinds_order]
    )

    kind_legend = "".join(
        f'<li><code>{_KIND_LABELS[k]}</code> — {_KIND_DESCRIPTIONS[k]}</li>'
        for k in kinds_order
    )

    return (
        '<section><h2>0. Task resolution rate (per issue)</h2>'
        '<p class="note">Per-model resolve rate using SWE-bench-style unit of analysis: '
        'one <em>(model, issue#)</em> pair per attempt, resolved if <strong>any</strong> same-model session on that issue reached a terminal action. '
        'Issues are joined across sessions by the GitHub issue/PR number that appears in the session name or in the first user message. '
        'This counts all legitimate maintainer completion mechanisms (ship, triage-close, duplicate-close, won\'t-fix), not just code-shipped resolutions.</p>'
        '<div class="resolve-chart">' + ''.join(bars) + '</div>'
        + _html_table(headers, rows)
        + '<details class="resolve-legend"><summary>What each resolution kind means</summary>'
        f'<ul>{kind_legend}</ul>'
        '<p class="note" style="margin-top:8px"><strong>Precedence:</strong> <code>push</code> > <code>gh_merge</code> > <code>gh_close</code> > <code>gh_comment</code> > <code>user_close</code>. '
        'A session with both a push and a triage comment is counted as <code>push</code>.</p>'
        '</details>'
        '</section>'
    )


def _render_trajectory_metadata_section(
    meta: dict,
    raw_single_counts: dict[str, int],
    models: list[str],
    model_meta: dict[str, dict[str, str]],
) -> str:
    present_models = [m for m in models if m in meta]
    if not present_models:
        return ""

    lo_model = min(present_models, key=lambda m: float(meta[m]["median"] or 0))
    hi_model = max(present_models, key=lambda m: float(meta[m]["median"] or 0))
    lo_med = float(meta[lo_model]["median"] or 0)
    hi_med = float(meta[hi_model]["median"] or 0)
    ratio_text = f" The longer trajectories take ~{(hi_med / lo_med):.1f}x more steps per session." if lo_med > 0 else ""
    intro = (
        f'Median trajectory length varies from {_fmt(meta[lo_model]["median"])} to {_fmt(meta[hi_model]["median"])} '
        f'tool steps ({html.escape(model_meta[lo_model]["label"])} vs. {html.escape(model_meta[hi_model]["label"])}).'
        f'{ratio_text}'
    )

    axis_limit = max(int(meta[m]["max"] or 0) for m in present_models)
    axis_limit = max(1, ((axis_limit + 9) // 10) * 10)

    chart_rows: list[str] = []
    for m in present_models:
        mm = meta[m]
        color = model_meta[m]["color"]
        p25 = float(mm["p25"] or 0)
        p75 = float(mm["p75"] or 0)
        median = float(mm["median"] or 0)
        left = p25 / axis_limit * 100.0
        width = max((p75 - p25) / axis_limit * 100.0, 1.0)
        dot = median / axis_limit * 100.0
        chart_rows.append(
            '<div class="marker-row">'
            f'<div class="marker-label" style="color:{color}">{html.escape(model_meta[m]["label"])}</div>'
            '<div class="marker-track">'
            '<span class="marker-grid" style="left:25%"></span>'
            '<span class="marker-grid" style="left:50%"></span>'
            '<span class="marker-grid" style="left:75%"></span>'
            f'<span class="marker-iqr" style="left:{left:.1f}%;width:{width:.1f}%;background:{color}"></span>'
            f'<span class="marker-dot" style="left:{dot:.1f}%;background:{color}"></span>'
            f'<span style="position:absolute;left:{dot:.1f}%;top:-18px;transform:translateX(-50%);font-size:10px;color:{color}">{_fmt(mm["median"])}</span>'
            '</div>'
            f'<div class="marker-summary">{_fmt(mm["p25"])}&ndash;{_fmt(mm["p75"])} IQR · {_fmt(mm["min"])}&ndash;{_fmt(mm["max"])} range</div>'
            '</div>'
        )

    rows: list[list[str]] = []
    for m in present_models:
        mm = meta[m]
        rows.append([
            html.escape(model_meta[m]["label"]),
            _fmt(raw_single_counts.get(m, 0)),
            _fmt(mm["n"]),
            _fmt(mm["total_steps"]),
            _fmt(mm["avg"]),
            _fmt(mm["median"]),
            _fmt(mm["p25"]),
            _fmt(mm["p75"]),
            _fmt(mm["min"]),
            _fmt(mm["max"]),
            _fmt(mm["completed"]),
            html.escape(mm["completion_rate"]),
        ])

    axis_labels = (
        '<div class="marker-axis">'
        '<span>0</span>'
        f'<span>{axis_limit // 2}</span>'
        f'<span>{axis_limit} steps</span>'
        '</div>'
    )

    return (
        '<section><h2>1. Trajectory Metadata</h2>'
        f'<p class="note">{intro}</p>'
        '<div class="marker-card">'
        + axis_labels
        + ''.join(chart_rows)
        + '</div>'
        + _html_table(
            [
                "model",
                "single-model sessions",
                "analyzed sessions",
                "total_steps",
                "avg",
                "median",
                "p25",
                "p75",
                "min",
                "max",
                "completed",
                "completion rate",
            ],
            rows,
        )
        + '<p class="note">Summary statistics for Pi tool-call trajectory length, measured in classified tool steps per session. '
        '<strong>single-model sessions</strong> is the raw eligible pool after purity filtering; '
        '<strong>analyzed sessions</strong> is the subset with a usable tool trajectory; '
        '<strong>completed</strong> means the transcript reached a terminal assistant stop.</p>'
        '</section>'
    )


def _parse_merge_specs(specs: list[str] | None) -> list[dict]:
    """Parse ``--merge-models`` flags of the form ``SRC1,SRC2=KEY:LABEL``."""
    out: list[dict] = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"invalid --merge-models spec (missing '='): {spec!r}")
        src_side, dst_side = spec.split("=", 1)
        sources = [s.strip() for s in src_side.split(",") if s.strip()]
        if not sources:
            raise ValueError(f"invalid --merge-models spec (no sources): {spec!r}")
        if ":" in dst_side:
            key, label = dst_side.split(":", 1)
        else:
            key, label = dst_side, dst_side
        key = key.strip()
        label = label.strip()
        if not key:
            raise ValueError(f"invalid --merge-models spec (empty target key): {spec!r}")
        out.append({"sources": sources, "key": key, "label": label or key})
    return out


def _apply_model_merges(
    merges: list[dict],
    results: dict[str, list],
    allowed_paths: dict[str, set[str]],
    raw_counts: dict[str, int],
) -> None:
    """Merge multiple source model keys into a single synthetic key in place."""
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



def _merge_resolution_stats(
    merges: list[dict],
    resolution_stats: dict[str, ModelResolutionStats],
) -> dict[str, ModelResolutionStats]:
    merged = dict(resolution_stats)
    for merge in merges:
        key = merge["key"]
        n_sessions = 0
        kind_counts: dict[str, int] = {}
        sessions = []
        issues_attempted: set[str] = set()
        issues_resolved: set[str] = set()
        saw_any = False

        for src in merge["sources"]:
            stat = merged.pop(src, None)
            if stat is None:
                continue
            saw_any = True
            n_sessions += stat.n_sessions
            sessions.extend(stat.sessions)
            issues_attempted.update(stat.issues_attempted)
            issues_resolved.update(stat.issues_resolved)
            for kind, count in stat.kind_counts.items():
                kind_counts[kind] = kind_counts.get(kind, 0) + count

        if saw_any:
            merged[key] = ModelResolutionStats(
                model=key,
                n_sessions=n_sessions,
                n_issues_attempted=len(issues_attempted),
                n_issues_resolved=len(issues_resolved),
                kind_counts=kind_counts,
                sessions=sessions,
                issues_resolved=issues_resolved,
                issues_attempted=issues_attempted,
            )
    return merged



def _render_detailed_classification_section(results: dict[str, list], models: list[str], model_meta: dict[str, dict[str, str]]) -> str:
    counts, totals = _intent_counters(results, models)
    category_order = ["read", "search", "reproduce", "edit", "verify", "git", "housekeeping", "failed", "other"]

    header = (
        '<table><thead><tr>'
        '<th>category</th><th>intent</th><th>description</th>' +
        ''.join(f'<th>{html.escape(model_meta[m]["label"])}</th>' for m in models) +
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


def _render_cleanup_decomposition_section(results: dict[str, list], models: list[str], model_meta: dict[str, dict[str, str]]) -> str:
    counts, totals = _intent_counters(results, models)
    preferred_order = [
        "git-github-context",
        "git-repo-inspect",
        "git-diff-review",
        "git-sync-integrate",
        "git-local-state-change",
        "git-publish",
        "file-cleanup",
        "create-documentation",
        "start-service",
        "install-deps",
        "check-tool-exists",
    ]
    cleanup_intents = [
        intent
        for intent in preferred_order
        if any(counts[m].get(intent, 0) > 0 for m in models)
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
        '<p class="note">In the inherited phase schema, <strong>cleanup = git + housekeeping</strong>. For Pi transcripts this phase is mostly repo workflow, not literal cleanup. This table makes the git side explicit.</p>'
        + _html_table(["high-level", "intent", "description"] + [html.escape(model_meta[m]["label"]) for m in models], summary_rows + rows)
        + '</section>'
    )


def _slugify_heading(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "section"


def _add_h2_ids(html_text: str) -> str:
    seen: dict[str, int] = {}

    def repl(match: re.Match[str]) -> str:
        attrs = match.group(1) or ""
        inner = match.group(2)
        if "id=" in attrs:
            return match.group(0)
        slug = _slugify_heading(inner)
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        return f'<h2{attrs} id="{slug}">{inner}</h2>'

    return re.sub(r"<h2([^>]*)>(.*?)</h2>", repl, html_text, flags=re.S)


def render_html(
    results: dict[str, list],
    raw_single_counts: dict[str, int],
    user_data: dict,
    resolution_stats: dict,
    display_labels: dict[str, str] | None = None,
) -> str:
    models = sorted(
        results.keys(),
        key=lambda m: (
            -(resolution_stats.get(m).resolve_rate if resolution_stats.get(m) is not None else -1.0),
            -raw_single_counts.get(m, 0),
            -len(results.get(m, [])),
            m,
        ),
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
    for model, label in (display_labels or {}).items():
        if model in model_meta:
            model_meta[model]["label"] = label

    model_tags = " ".join(
        f'<span class="tag" style="border-color:{model_meta[m]["color"]};color:{model_meta[m]["color"]}">{html.escape(model_meta[m]["label"])}</span>'
        for m in models
    )

    sections: list[str] = []

    # 0. Task resolution rate
    sections.append(_render_resolution_section(resolution_stats, models, model_meta))

    # 1. Trajectory metadata
    sections.append(_render_trajectory_metadata_section(meta, raw_single_counts, models, model_meta))

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

    sections.append(_render_detailed_classification_section(results, models, model_meta))
    sections.append(_render_cleanup_decomposition_section(results, models, model_meta))

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

    sections.append(_render_intervention_macro_section(user_data, models, model_meta))
    sections.append(_render_user_message_sections(user_data, models, model_meta))

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

    html_out = f"""<!doctype html>
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
    .heat-strip {{ display:grid; grid-template-columns: repeat(20, minmax(10px, 1fr)); gap:2px; min-width: 300px; }}
    .heat-cell {{ display:block; height:14px; border-radius:2px; background:#999; }}
    .marker-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin: 12px 0; }}
    .marker-axis {{ margin: 8px 0 10px 220px; display:flex; justify-content:space-between; color: var(--muted); font-size: 12px; max-width: 560px; }}
    .marker-row {{ display:grid; grid-template-columns: 210px minmax(360px, 560px) 170px; gap: 10px; align-items:center; margin: 8px 0; }}
    .marker-label {{ font-size: 13px; }}
    .marker-summary {{ font-size: 12px; color: var(--muted); }}
    .marker-track {{ position: relative; height: 18px; background: #f1efe6; border-radius: 999px; overflow: visible; }}
    .marker-grid {{ position:absolute; top:-3px; bottom:-3px; width:1px; background: rgba(0,0,0,0.08); transform: translateX(-0.5px); }}
    .marker-iqr {{ position:absolute; top:3px; height:12px; border-radius: 999px; opacity: 0.45; }}
    .marker-dot {{ position:absolute; top:1px; width:16px; height:16px; border-radius: 999px; transform: translateX(-50%); border: 2px solid rgba(255,255,255,0.95); box-shadow: 0 0 0 1px rgba(0,0,0,0.15); }}
    .resolve-chart {{ margin: 6px 0 18px 0; }}
    .resolve-row {{ display:grid; grid-template-columns: 160px minmax(240px, 1fr) 260px; gap: 12px; align-items:center; margin: 6px 0; }}
    .resolve-label {{ font-style: italic; font-size: 13px; }}
    .resolve-track {{ position: relative; height: 18px; background: #f1efe6; border-radius: 999px; overflow: hidden; }}
    .resolve-bar {{ position: absolute; left: 0; top: 0; bottom: 0; opacity: 0.78; }}
    .resolve-summary {{ font-size: 12px; color: var(--muted); }}
    .resolve-summary strong {{ color: var(--text); font-size: 14px; font-style: italic; font-weight: 400; }}
    details.resolve-legend {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin: 14px 0 0 0; }}
    details.resolve-legend ul {{ margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; }}
    details.resolve-legend li {{ margin: 3px 0; color: var(--text); }}
    details.msg-group, details.msg-subgroup {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; margin: 10px 0; }}
    details.msg-subgroup {{ margin: 10px 0 12px 0; }}
    summary {{ cursor:pointer; }}
    .msg-list {{ margin: 10px 0 0 0; padding-left: 18px; }}
    .msg-list li {{ margin: 0 0 10px 0; }}
    .msg-meta {{ margin-bottom: 2px; }}
    .msg-text {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Pi transcript reference tables</h1>
    <div class="subtitle">Same high-level taxonomy as the SWE-Agent analysis, adapted to Pi tool calls and currently filtered to strict single-model issue sessions</div>
    <p class="lede">This page is the Pi-session analogue of the original reference tables. It keeps the same high-level read/search/reproduce/edit/verify/git structure, but gives Pi a more semantic low-level git decomposition while classifying Pi’s <code>read</code>, <code>edit</code>, <code>write</code>, <code>bash</code>, and auxiliary tools into that shared scheme.</p>
    <div class="tags">{model_tags}</div>
    <div class="cards">{''.join(cards)}</div>
    {''.join(sections)}
  </div>
</body>
</html>"""
    return _add_h2_ids(html_out)


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
    parser.add_argument(
        "--merge-models",
        action="append",
        default=[],
        help="Merge multiple models after filtering, e.g. claude-opus-4-5,claude-opus-4-6=claude-opus-4-5-6:Opus 4.5/4.6",
    )
    parser.add_argument("--output", "-o", default="docs/pi-references.html")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    session_filter = SessionFilter(
        allowed_models=args.models,
        require_single_model=True,
        session_name_prefixes=args.session_name_prefix,
    )
    merge_specs = _parse_merge_specs(args.merge_models)

    allowed_paths, raw_counts, _ = collect_filtered_paths(data_root, session_filter)
    results = process_all(data_root, models=args.models)
    results = _filter_results_to_paths(results, allowed_paths)
    resolution_stats = compute_resolution_by_model(data_root, session_filter)

    if merge_specs:
        _apply_model_merges(merge_specs, results, allowed_paths, raw_counts)
        resolution_stats = _merge_resolution_stats(merge_specs, resolution_stats)

    for model in sorted(results):
        print(f"  {model}: {raw_counts.get(model, 0)} strict single-model sessions, {len(results.get(model, []))} analyzed")

    user_data = analyze_user_messages(allowed_paths)
    display_labels = {spec["key"]: spec["label"] for spec in merge_specs}
    html_out = render_html(
        results,
        raw_counts,
        user_data,
        resolution_stats,
        display_labels=display_labels,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
