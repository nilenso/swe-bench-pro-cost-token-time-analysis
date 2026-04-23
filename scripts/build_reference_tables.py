#!/usr/bin/env python3
"""
Build a reference HTML with all raw data tables for the analytics report.

Uses the analysis/ package for data. Supports N models automatically.

Usage:
  python scripts/build_reference_tables.py --data-root data -o docs/reference.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.orchestrate import process_all
from analysis.models import (
    MODELS,
    INTENT_TO_HIGH_LEVEL,
    INTENT_DESCRIPTIONS,
    HIGH_LEVEL_COLORS,
)
from analysis.failure_modes import FAMILY_COLOR
from analysis.classify import SEQUENCE_VERIFY_INTENTS

# ── Helpers ──────────────────────────────────────────────

def _pct(n, total):
    return f"{n / total * 100:.1f}%" if total > 0 else "0%"

def _median_safe(vals):
    return round(statistics.median(vals), 1) if vals else None

def _p25(vals):
    s = sorted(vals)
    return round(s[len(s) // 4], 1) if s else None

def _p75(vals):
    s = sorted(vals)
    return round(s[len(s) * 3 // 4], 1) if s else None


# ── Palette helpers ───────────────────────────────────────

# Muted sub-intent colors (Tufte: desaturated, distinguishable)
_VERIFY_SUB_COLORS = {
    "run-test-suite":       "#8faabc",
    "run-test-specific":    "#6b8fa8",
    "run-verify-script":    "#b0956a",
    "create-test-script":   "#9ab07a",
    "run-inline-verify":    "#c4957a",
    "compile-build":        "#7aaa8a",
    "edit-test-or-repro":   "#aaa07a",
    "run-custom-script":    "#9a8aaa",
    "create-verify-script": "#ba9a7a",
    "syntax-check":         "#8a9a9a",
}

def _model_color(model: str) -> str:
    return MODELS.get(model, {}).get("color", "#888")

def _model_label(model: str) -> str:
    return MODELS.get(model, {}).get("label", model)


# ── HTML rendering ───────────────────────────────────────

def _html_table(headers: list[str], rows: list[list], caption: str = "") -> str:
    h = "".join(f"<th>{hdr}</th>" for hdr in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n"
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def render_structure_sections(results, models) -> list[tuple[str, str, str]]:
    """Sections 9 (Work-Done vs Resolved) and 10 (Structural Markers).

    Returns list of (title_html, content_html, notes_html) tuples with
    Tufte-style inline visualizations alongside the data tables.
    """
    sections: list[tuple[str, str, str]] = []

    # ── Bucket colors (muted, Tufte-appropriate) ─────────────
    _BUCKET_COLORS = {
        "wr": "#6a9a6a",   # wd+resolved: muted green (good outcome)
        "wu": "#b0956a",   # wd+unresolved: muted amber (false positive)
        "nr": "#6a8da8",   # no_wd+resolved: muted blue (lucky submit)
        "nu": "#bbb",      # no_wd+unresolved: gray (nothing worked)
    }
    _BUCKET_LABELS = {
        "wr": "wd + resolved",
        "wu": "wd + unresolved",
        "nr": "no wd + resolved",
        "nu": "no wd + unresolved",
    }

    # ── Section 9: Work-Done vs Resolved ─────────────────────
    # Compute buckets per model
    buckets = {}  # model -> {wr, wu, nr, nu}
    for model in models:
        b = {"wr": 0, "wu": 0, "nr": 0, "nu": 0}
        for r in results[model]:
            if r.work_done and r.resolved:
                b["wr"] += 1
            elif r.work_done:
                b["wu"] += 1
            elif r.resolved:
                b["nr"] += 1
            else:
                b["nu"] += 1
        buckets[model] = b

    # Data table
    headers = ["model", "wd+resolved", "wd+unresolved", "no_wd+resolved", "no_wd+unresolved", "total"]
    tbl_rows = []
    for model in models:
        b = buckets[model]
        tbl_rows.append([model, b["wr"], b["wu"], b["nr"], b["nu"], len(results[model])])
    table_html = _html_table(headers, tbl_rows)

    # Stacked bar visualization
    bar_width = 600  # px, total bar width
    bar_height = 22
    bar_html = '<div style="margin:20px 0 12px 0">'

    # Legend
    bar_html += '<div style="display:flex;gap:18px;margin-bottom:10px;font-size:11.5px;color:#555">'
    for key in ("wr", "wu", "nr", "nu"):
        bar_html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{_BUCKET_COLORS[key]};border-radius:2px"></span>'
            f'{_BUCKET_LABELS[key]}</span>'
        )
    bar_html += '</div>'

    for model in models:
        b = buckets[model]
        n = len(results[model])
        color = _model_color(model)
        label = _model_label(model)

        bar_html += (
            f'<div style="display:grid;grid-template-columns:100px {bar_width}px 1fr;'
            f'align-items:center;gap:8px;margin-bottom:6px">'
            f'<div style="text-align:right;font-size:12px;color:{color};'
            f'font-style:italic">{label}</div>'
            f'<div style="display:flex;height:{bar_height}px;border-radius:3px;overflow:hidden">'
        )

        for key in ("wr", "wu", "nr", "nu"):
            count = b[key]
            pct = count / n * 100 if n > 0 else 0
            if pct < 0.5:
                # Too narrow to render meaningfully
                bar_html += (
                    f'<div style="width:{pct:.2f}%;background:{_BUCKET_COLORS[key]}" '
                    f'title="{_BUCKET_LABELS[key]}: {count} ({pct:.1f}%)"></div>'
                )
            else:
                # Show count label if segment is wide enough
                show_label = pct > 5
                bar_html += (
                    f'<div style="width:{pct:.2f}%;background:{_BUCKET_COLORS[key]};'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:10px;color:{"rgba(255,255,255,0.85)" if show_label else "transparent"}'
                    f'" title="{_BUCKET_LABELS[key]}: {count} ({pct:.1f}%)">'
                    f'{"" + str(count) if show_label else ""}</div>'
                )

        bar_html += '</div>'

        # Annotation: wd+resolved percentage
        wr_pct = b["wr"] / n * 100 if n > 0 else 0
        bar_html += (
            f'<div style="font-size:11px;color:#666">'
            f'{wr_pct:.1f}% wd+resolved</div>'
        )
        bar_html += '</div>'

    bar_html += '</div>'

    content_9 = bar_html + table_html

    notes_9 = (
        "<p>A confusion matrix crossing two signals: whether the agent reached "
        "'work-done' and whether the benchmark evaluated the patch as correct.</p>"
        "<p><strong>work-done</strong>: the trajectory contains a seq-first-all-pass label. "
        "Specifically: we find the last step classified as a source edit (edit-source, insert-source, "
        "apply-patch, edit-via-inline-script). Then we check if any verify step after that point "
        "has a 'pass' outcome (all tests passed, per the rules in Section 7). If yes, the trajectory "
        "is 'work-done'. This is a stronger signal than first_verify_pass (Section 10), which can fire "
        "before any edits because existing tests pass on unmodified code.</p>"
        "<p><strong>resolved</strong>: the submitted patch actually fixes the failing tests, "
        "as judged by the SWE-Bench Pro benchmark evaluation (from <code>agent_runs_data.csv</code>). "
        "This is different from 'submitted', which only means the agent produced a patch.</p>"
        "<p><strong>wd+resolved</strong>: the agent's tests passed after its last edit, "
        "and the benchmark confirmed the patch is correct. The best case.</p>"
        "<p><strong>wd+unresolved</strong>: tests passed but the patch was wrong. "
        "The agent's own verification was a false positive.</p>"
        "<p><strong>no_wd+resolved</strong>: the agent never reached a clean test pass "
        "after its final edit, yet the benchmark accepted the patch. "
        "The agent submitted without confirmation that its code works.</p>"
        "<p><strong>no_wd+unresolved</strong>: the agent neither achieved passing tests "
        "nor produced a correct patch.</p>"
        "<p>The stacked bars show how each model's 730 trajectories split across these four "
        "outcomes. The no_wd+resolved column is notably large across all models, "
        "meaning the agent's own test-passing signal is not a reliable predictor of "
        "benchmark resolution.</p>"
        "<p>Method: 'work-done' = find the last step in SOURCE_EDIT_INTENTS (edit-source, "
        "insert-source, apply-patch, edit-via-inline-script), then scan forward for any step where "
        "classify_verify_outcome() returns 'pass'. If found, work-done is true. "
        "'resolved' comes from the benchmark CSV (<code>agent_runs_data.csv</code>, "
        "field <code>metadata.resolved</code>), which records whether the submitted patch "
        "actually made the failing tests pass when evaluated by the benchmark harness.</p>"
        "<p>Caveat: work-done can be a false positive. The agent's tests might pass because "
        "the test suite doesn't cover the specific failure the task requires fixing. The agent "
        "thinks it's done (tests pass), but the benchmark's evaluation finds the bug isn't actually fixed.</p>"
    )

    sections.append((
        "<h2>9. Work-Done vs Resolved</h2>",
        content_9,
        notes_9,
    ))

    # ── Section 10: Structural Markers ───────────────────────
    marker_styles = {
        "first_edit":        "width:8px;height:8px;border-radius:50%",                                # circle
        "last_edit":         "width:8px;height:8px;border-radius:50%;border:2px solid {color};background:#fffff8",  # hollow circle
        "first_verify":      "width:7px;height:7px;border-radius:1px;transform:rotate(45deg)",        # diamond
        "first_verify_pass": "width:8px;height:8px;border-radius:1px",                                # filled square
        "last_verify":       "width:8px;height:8px;border-radius:1px;border:2px solid {color};background:#fffff8",  # hollow square
        "submit":            "width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-bottom:9px solid {color}", # triangle
    }

    def render_marker_variant(marker_keys, marker_display, intro_html="", marker_style_overrides=None):
        def marker_position(r, mk):
            pos = r.positions.get(mk)
            if pos is not None:
                return pos
            if mk != "last_verify":
                return None
            n = len(r.base_intents)
            idx = next((i for i in range(n - 1, -1, -1) if r.base_intents[i] in SEQUENCE_VERIFY_INTENTS), -1)
            return round(idx / max(n - 1, 1) * 100, 1) if idx >= 0 else None

        style_map = dict(marker_styles)
        if marker_style_overrides:
            style_map.update(marker_style_overrides)

        marker_data = {}
        for model in models:
            marker_data[model] = {}
            for mk in marker_keys:
                vals = [pos for r in results[model] if (pos := marker_position(r, mk)) is not None]
                marker_data[model][mk] = {
                    "med": _median_safe(vals),
                    "p25": _p25(vals),
                    "p75": _p75(vals),
                    "n": len(vals),
                }

        headers = ["marker"]
        for m in models:
            headers.extend([f"{m}_med", f"{m}_p25", f"{m}_p75", f"{m}_n"])
        tbl_rows = []
        for mk in marker_keys:
            row = [mk]
            for model in models:
                d = marker_data[model][mk]
                row.extend([d["med"], d["p25"], d["p75"], d["n"]])
            tbl_rows.append(row)
        table_html = _html_table(headers, tbl_rows)

        track_width = 560
        timeline_html = '<div style="margin:20px 0 16px 0">'
        timeline_html += '<div style="display:flex;gap:16px;margin-bottom:14px;font-size:11px;color:#555">'
        for mk in marker_keys:
            style = style_map[mk].replace("{color}", "#888")
            is_hollow = "border:2px solid" in style
            if is_hollow:
                swatch = f'<span style="display:inline-block;{style};box-sizing:border-box"></span>'
            elif mk == "submit":
                swatch = f'<span style="display:inline-block;{style}"></span>'
            else:
                swatch = f'<span style="display:inline-block;{style};background:#888"></span>'
            timeline_html += (
                f'<span style="display:inline-flex;align-items:center;gap:4px">'
                f'{swatch} {marker_display[mk]}</span>'
            )
        timeline_html += '</div>'

        timeline_html += (
            f'<div style="display:grid;grid-template-columns:100px {track_width}px;gap:8px;margin-bottom:2px">'
            f'<div></div>'
            f'<div style="position:relative;height:14px;font-size:9px;color:#999">'
        )
        for tick in (0, 25, 50, 75, 100):
            left = tick / 100 * track_width
            timeline_html += (
                f'<span style="position:absolute;left:{left:.0f}px;transform:translateX(-50%)">{tick}%</span>'
            )
        timeline_html += '</div></div>'

        for model in models:
            color = _model_color(model)
            label = _model_label(model)
            timeline_html += (
                f'<div style="display:grid;grid-template-columns:100px {track_width}px;gap:8px;align-items:center;margin-bottom:10px">'
                f'<div style="text-align:right;font-size:12px;color:{color};font-style:italic">{label}</div>'
                f'<div style="position:relative;height:26px">'
            )
            timeline_html += (
                f'<div style="position:absolute;top:12px;left:0;width:100%;height:1px;background:#ddd"></div>'
            )

            for mk in marker_keys:
                d = marker_data[model][mk]
                if d["med"] is None:
                    continue
                p25 = d["p25"] if d["p25"] is not None else d["med"]
                p75 = d["p75"] if d["p75"] is not None else d["med"]
                left_pct = p25
                width_pct = max(p75 - p25, 0.3)
                timeline_html += (
                    f'<div style="position:absolute;top:10px;height:5px;left:{left_pct:.2f}%;width:{width_pct:.2f}%;background:{color};opacity:0.18;border-radius:2px" '
                    f'title="{marker_display[mk]} IQR: {p25:.1f}%-{p75:.1f}%"></div>'
                )

            for mk in marker_keys:
                d = marker_data[model][mk]
                if d["med"] is None:
                    continue
                left_pct = d["med"]
                mk_style = style_map[mk].replace("{color}", color)
                if "transform:rotate(45deg)" in mk_style:
                    mk_style = mk_style.replace("transform:rotate(45deg)", "transform:translateX(-50%) rotate(45deg)")
                    base_transform = ""
                else:
                    base_transform = "transform:translateX(-50%);"
                is_hollow = "border:2px solid" in mk_style
                if is_hollow:
                    marker_el = (
                        f'<div style="position:absolute;left:{left_pct:.2f}%;top:5px;{base_transform}{mk_style};box-sizing:border-box" '
                        f'title="{marker_display[mk]}: {d["med"]:.1f}%"></div>'
                    )
                elif mk == "submit":
                    marker_el = (
                        f'<div style="position:absolute;left:{left_pct:.2f}%;top:4px;transform:translateX(-50%);{mk_style}" '
                        f'title="{marker_display[mk]}: {d["med"]:.1f}%"></div>'
                    )
                else:
                    marker_el = (
                        f'<div style="position:absolute;left:{left_pct:.2f}%;top:5px;{base_transform}{mk_style};background:{color}" '
                        f'title="{marker_display[mk]}: {d["med"]:.1f}%"></div>'
                    )
                timeline_html += marker_el

            timeline_html += '</div></div>'

        timeline_html += '</div>'
        return intro_html + timeline_html + table_html

    content_10 = render_marker_variant(
        ["first_edit", "last_edit", "first_verify", "first_verify_pass", "submit"],
        {
            "first_edit": "first edit",
            "last_edit": "last edit",
            "first_verify": "first verify",
            "first_verify_pass": "first pass",
            "submit": "submit",
        },
    )

    content_10 += render_marker_variant(
        ["first_edit", "last_edit", "first_verify", "last_verify", "submit"],
        {
            "first_edit": "first edit",
            "last_edit": "last edit",
            "first_verify": "first verify",
            "last_verify": "last verify",
            "submit": "submit",
        },
        intro_html=(
            '<h3 style="font-size:14px;font-style:italic;font-weight:400;margin:18px 0 6px 0">'
            'Alternative view: replace first pass with last verify</h3>'
            '<p style="font-size:12px;color:#666;margin:0 0 8px 0">'
            'This version drops first pass, which can happen on baseline tests before any edits, '
            'and instead shows the last verify step so the post-edit verification tail is easier to see.'
            '</p>'
        ),
        marker_style_overrides={
            "first_edit": "width:8px;height:8px;border-radius:50%;border:2px solid {color};background:#fffff8",
            "last_edit": "width:8px;height:8px;border-radius:50%",
            "first_verify": "width:8px;height:8px;border-radius:1px;transform:rotate(45deg);border:2px solid {color};background:#fffff8",
            "last_verify": "width:8px;height:8px;border-radius:1px;transform:rotate(45deg)",
        },
    )

    notes_10 = (
        "<p>Key events in each trajectory, expressed as a percentage of the way through "
        "(0% = first step, 100% = last step). Aggregated across all trajectories per model.</p>"
        "<p>The timeline shows median positions as shaped markers, with faint bands for the "
        "interquartile range (p25-p75). Hover over markers for exact values.</p>"
        "<p><strong>first_edit</strong>: the first step whose base intent is one of: "
        "edit-source (str_replace on a source file), insert-source (str_replace_editor insert), "
        "apply-patch, or edit-via-inline-script. Does not include create-file or edit-test-or-repro, "
        "which are classified differently.</p>"
        "<p><strong>last_edit</strong>: the last step matching those same intents. "
        "The gap between last_edit and submit is the 'tail' where the agent is verifying, "
        "cleaning up, or submitting but no longer changing source code.</p>"
        "<p><strong>first_verify</strong>: the first step whose intent is in SEQUENCE_VERIFY_INTENTS: "
        "run-test-suite, run-test-specific, run-verify-script, run-custom-script, compile-build, "
        "syntax-check, run-inline-verify.</p>"
        "<p><strong>first_verify_pass</strong>: the first step where classify_verify_outcome() returns 'pass' "
        "(see Section 7 for what 'pass' means). This does NOT mean 'the agent's fix worked'. It means "
        "'the first time a test/build command produced output where all tests passed'. Because SWE-Bench Pro "
        "tasks have existing test suites that mostly pass on unmodified code, an agent that runs pytest before "
        "making any edits will often get a 'pass' here. This is why first_verify_pass can appear before first_edit "
        "for Claude (median 28.8% vs 34.6%): Claude runs the existing test suite early as a diagnostic baseline. "
        "For a marker that means 'the fix works', see work_done in Section 9, which requires a verify pass "
        "after the last source edit.</p>"
        "<p><strong>last_verify</strong>: the last step whose intent is in SEQUENCE_VERIFY_INTENTS. "
        "The alternate view replaces first pass with last verify to show where verification actually finishes, "
        "which makes the late verification tail clearer for Claude.</p>"
        "<p><strong>submit</strong>: the first step with intent 'submit'.</p>"
        "<p><strong>_med / _p25 / _p75</strong>: median, 25th percentile, and 75th "
        "percentile across trajectories where the event occurred.</p>"
        "<p><strong>_n</strong>: number of trajectories where this event occurred. "
        "Claude has 639 for first_verify_pass (out of 730) meaning 91 trajectories never had "
        "a fully-passing test run. GPT has only 130, meaning most GPT trajectories either never "
        "ran tests or never achieved a clean pass.</p>"
        "<p>Method: for each trajectory, scan for the first (or last) step matching the relevant "
        "intent set, compute step_index / (total_steps - 1) * 100 to get a percentage position, "
        "then take the median across all trajectories where the event occurred.</p>"
    )

    sections.append((
        "<h2>10. Structural Markers (% of trajectory)</h2>",
        content_10,
        notes_10,
    ))

    return sections


def render_verify_sections(results, models) -> list[tuple[str, str, str]]:
    """Return (title_html, content_html, notes_html) for sections 5, 7, 8.

    Each section: phenomenon line, inline visualization, data table, notes.
    """
    sections = []

    # -- Section 5: Verify Sub-Intent Breakdown --

    verify_intents = [i for i, h in INTENT_TO_HIGH_LEVEL.items() if h == "verify"]

    # Aggregate counts per model per sub-intent
    sub_totals = {}  # {model: {intent: count}}
    for model in models:
        c = Counter()
        for r in results[model]:
            for intent in verify_intents:
                c[intent] += r.base_intent_counts.get(intent, 0)
        sub_totals[model] = c

    # Sort sub-intents by total across models (most common first)
    sorted_subs = sorted(
        verify_intents,
        key=lambda i: -sum(sub_totals[m].get(i, 0) for m in models))
    sorted_subs = [s for s in sorted_subs
                   if sum(sub_totals[m].get(s, 0) for m in models) > 0]

    # Stacked horizontal bars showing composition of verify per model
    sec5_viz = '<div style="margin:16px 0 20px 0">\n'

    # Legend
    sec5_viz += ('<div style="display:flex;flex-wrap:wrap;gap:12px;'
                 'margin-bottom:12px;font-size:11px;color:#777">\n')
    for si in sorted_subs:
        color = _VERIFY_SUB_COLORS.get(si, "#bbb")
        short = (si.replace("run-", "").replace("create-", "c-")
                 .replace("-or-repro", ""))
        sec5_viz += (
            f'<span style="display:inline-flex;align-items:center;gap:4px">'
            f'<span style="width:10px;height:10px;border-radius:2px;'
            f'background:{color};display:inline-block"></span>'
            f'{short}</span>\n')
    sec5_viz += '</div>\n'

    for model in models:
        total = sum(sub_totals[model].values())
        if total == 0:
            continue
        label = _model_label(model)
        sec5_viz += (
            f'<div style="display:grid;grid-template-columns:110px 1fr 40px;'
            f'gap:8px;align-items:center;margin-bottom:6px">\n')
        sec5_viz += (
            f'<span style="text-align:right;font-size:12px;'
            f'color:{_model_color(model)}">{label}</span>\n')
        sec5_viz += ('<div style="display:flex;height:16px;border-radius:2px;'
                     'overflow:hidden">\n')
        for si in sorted_subs:
            n = sub_totals[model].get(si, 0)
            if n == 0:
                continue
            pct = n / total * 100
            color = _VERIFY_SUB_COLORS.get(si, "#bbb")
            title = f"{si}: {n} ({pct:.1f}%)"
            sec5_viz += (
                f'<div style="width:{pct:.2f}%;background:{color};'
                f'opacity:0.85" title="{title}"></div>\n')
        sec5_viz += '</div>\n'
        sec5_viz += f'<span style="font-size:10px;color:#999">{total:,}</span>\n'
        sec5_viz += '</div>\n'
    sec5_viz += '</div>\n'

    # Table
    headers = ["intent"] + [f"{m}_n" for m in models]
    tbl_rows = []
    for intent in sorted(verify_intents):
        row = [intent]
        for model in models:
            row.append(sum(r.base_intent_counts.get(intent, 0)
                           for r in results[model]))
        tbl_rows.append(row)
    tbl_rows.sort(key=lambda r: -sum(r[1:]))

    sec5_phenom = (
        '<p style="font-size:13px;color:#555;margin-bottom:8px">'
        'How models verify differs in kind, not just amount. '
        'Claude and GLM lean on broad test suites; '
        'Gemini and GPT use more targeted runs and custom scripts.</p>')

    sec5_notes = (
        "<p>The 'verify' category contains ~10 sub-intents. "
        "This table shows where each model's verification volume "
        "comes from.</p>"
        "<p><strong>run-test-suite</strong>: broad test runs "
        "(pytest, go test, npm test, mocha) without targeting "
        "specific tests.</p>"
        "<p><strong>run-test-specific</strong>: targeted test runs "
        "using pytest -k or :: to run specific test functions.</p>"
        "<p><strong>run-verify-script</strong>: running a script named "
        "verify*, check*, validate*, or edge_case*.</p>"
        "<p><strong>create-test-script</strong>: creating a new test "
        "file (test_*, *test.py, etc.).</p>"
        "<p><strong>run-inline-verify</strong>: an inline python -c / "
        "node -e snippet that imports project code or runs "
        "assertions.</p>"
        "<p><strong>compile-build</strong>: go build, go vet, make, "
        "npx tsc. Compilation as a verification step.</p>"
        "<p><strong>edit-test-or-repro</strong>: editing an existing "
        "test or repro file (str_replace on test_* or "
        "repro* files).</p>"
        "<p><strong>run-custom-script</strong>: running a named script "
        "that doesn't match repro/test/verify naming "
        "patterns.</p>"
        "<p><strong>create-verify-script</strong>: creating a new file "
        "named verify*, check*, validate*.</p>"
        "<p><strong>syntax-check</strong>: py_compile, compileall, "
        "node -c. Quick syntax validation.</p>")

    sections.append((
        "<h2>5. Verify Sub-Intent Breakdown</h2>",
        sec5_phenom + sec5_viz + _html_table(headers, tbl_rows),
        sec5_notes))

    # -- Section 7: Verify Outcomes --

    outcome_data = {}
    headers = ["model", "pass", "fail", "unknown", "total", "pass_rate"]
    tbl_rows = []
    for model in models:
        c = Counter()
        for r in results[model]:
            c.update(r.verify_outcome_counts)
        p, f_, u = c.get("pass", 0), c.get("fail", 0), c.get("", 0)
        det = p + f_
        total = p + f_ + u
        pr_str = f"{p / det * 100:.1f}%" if det else "n/a"
        outcome_data[model] = (p, f_, u, total, pr_str)
        tbl_rows.append([model, p, f_, u, total, pr_str])

    # Proportion bars: green=pass, red=fail, grey=unknown
    sec7_viz = '<div style="margin:16px 0 20px 0">\n'

    sec7_viz += (
        '<div style="display:flex;gap:16px;margin-bottom:10px;'
        'font-size:11px;color:#777">'
        '<span style="display:inline-flex;align-items:center;gap:4px">'
        '<span style="width:10px;height:10px;border-radius:2px;'
        'background:#6a9a6a;display:inline-block"></span>pass</span>'
        '<span style="display:inline-flex;align-items:center;gap:4px">'
        '<span style="width:10px;height:10px;border-radius:2px;'
        'background:#b05050;display:inline-block"></span>fail</span>'
        '<span style="display:inline-flex;align-items:center;gap:4px">'
        '<span style="width:10px;height:10px;border-radius:2px;'
        'background:#c8c8c0;display:inline-block"></span>unknown</span>'
        '</div>\n')

    for model in models:
        p, f_, u, total, pr_str = outcome_data[model]
        if total == 0:
            continue
        label = _model_label(model)
        pct_p = p / total * 100
        pct_f = f_ / total * 100
        pct_u = u / total * 100

        sec7_viz += (
            f'<div style="display:grid;grid-template-columns:110px 1fr 70px;'
            f'gap:8px;align-items:center;margin-bottom:6px">\n')
        sec7_viz += (
            f'<span style="text-align:right;font-size:12px;'
            f'color:{_model_color(model)}">{label}</span>\n')
        sec7_viz += ('<div style="display:flex;height:16px;border-radius:2px;'
                     'overflow:hidden">\n')
        if pct_p > 0:
            sec7_viz += (
                f'<div style="width:{pct_p:.2f}%;background:#6a9a6a;'
                f'opacity:0.8" title="pass: {p} ({pct_p:.1f}%)"></div>\n')
        if pct_f > 0:
            sec7_viz += (
                f'<div style="width:{pct_f:.2f}%;background:#b05050;'
                f'opacity:0.8" title="fail: {f_} ({pct_f:.1f}%)"></div>\n')
        if pct_u > 0:
            sec7_viz += (
                f'<div style="width:{pct_u:.2f}%;background:#c8c8c0;'
                f'opacity:0.7" title="unknown: {u} ({pct_u:.1f}%)">'
                '</div>\n')
        sec7_viz += '</div>\n'
        sec7_viz += (
            f'<span style="font-size:10px;color:#999">'
            f'pass rate {pr_str}</span>\n')
        sec7_viz += '</div>\n'

    sec7_viz += '</div>\n'

    sec7_phenom = (
        '<p style="font-size:13px;color:#555;margin-bottom:8px">'
        'About half of all verify steps yield a pass. '
        'Claude and GLM have nearly identical pass rates (~51%), '
        'despite Claude running 3x more verify steps.</p>')

    sec7_notes = (
        "<p>Only steps classified as one of these intents are evaluated for outcome: "
        "run-test-suite, run-test-specific, run-verify-script, run-custom-script, "
        "run-inline-verify, compile-build, syntax-check. All other steps get outcome ''.</p>"
        "<p><strong>pass</strong>: the observation's last 2000 characters match a framework-specific all-pass pattern. "
        "For pytest: the summary line (e.g. '200 passed in 12.3s') must contain 'passed' and must NOT contain 'failed' or 'error'. "
        "If even one test fails ('195 passed, 5 failed'), the outcome is 'fail', not 'pass'. "
        "For Go: all PASS/FAIL lines in output are checked; any FAIL makes it 'fail'. "
        "For Mocha: checks 'N passing' and 'N failing' counts. "
        "For Jest: checks summary line for 'failed' vs 'passed'. "
        "For compile-build: absence of error patterns in short output (< 200 chars) from go build/make = 'pass'. "
        "For syntax-check: py_compile with no output = 'pass'; any Error/SyntaxError = 'fail'.</p>"
        "<p><strong>fail</strong>: the observation matches a failure pattern. In priority order: "
        "framework-specific failure summaries (pytest 'failed', Go 'FAIL', Mocha failing > 0), "
        "then generic patterns: 'no tests ran', collection errors, tracebacks in the last 500 chars, "
        "Node.js throw/error, non-zero exit code.</p>"
        "<p><strong>unknown ('')</strong>: no pattern matched. This happens when output is from an unrecognized framework, "
        "is truncated, is ambiguous (e.g. pytest ran but the summary line was cut off), or when the observation is empty.</p>"
        "<p><strong>pass_rate</strong>: pass / (pass + fail), excluding unknowns. This measures: of the verify steps where "
        "we could determine the outcome, what fraction had all tests passing?</p>"
        "<p>Important caveat: 'pass' means 'all tests in that run passed', not 'the agent's fix is correct'. "
        "SWE-Bench Pro tasks come with existing test suites where most tests already pass on unmodified code. "
        "An agent running pytest before making any edits will often get 'pass' because the existing tests pass. "
        "This is why first_verify_pass can occur before first_edit (see Section 10).</p>")

    sections.append((
        "<h2>7. Verify Outcomes</h2>",
        sec7_phenom + sec7_viz + _html_table(headers, tbl_rows),
        sec7_notes))

    # -- Section 8: Sequence Labels --

    seq_totals = {}
    all_labels = set()
    for model in models:
        c = Counter()
        for r in results[model]:
            c.update(r.seq_label_counts)
        seq_totals[model] = c
        all_labels.update(c.keys())

    # Table
    headers = ["label"] + [f"{m}_n" for m in models]
    tbl_rows = []
    for label in sorted(all_labels):
        if label in ("", "seq-none"):
            continue
        tbl_rows.append([label] + [seq_totals[m].get(label, 0) for m in models])
    tbl_rows.sort(key=lambda r: -sum(r[1:]))

    # Edit-verify cycles per trajectory: bar + dot visualization
    loop_label = "seq-verify-after-edit"
    n_trajs = {m: len(results[m]) for m in models}

    per_traj_loops = {}
    for model in models:
        vals = [r.seq_label_counts.get(loop_label, 0)
                for r in results[model]]
        per_traj_loops[model] = {
            "mean": sum(vals) / len(vals) if vals else 0,
            "median": _median_safe(vals),
            "total": sum(vals),
        }

    max_mean = max((d["mean"] for d in per_traj_loops.values()), default=1)

    sec8_viz = '<div style="margin:16px 0 20px 0">\n'
    sec8_viz += ('<div style="font-size:11px;color:#999;margin-bottom:8px">'
                 'Average edit-then-verify cycles per trajectory</div>\n')

    for model in models:
        d = per_traj_loops[model]
        label = _model_label(model)
        bar_w = d["mean"] / max_mean * 100 if max_mean > 0 else 0

        sec8_viz += (
            f'<div style="display:grid;grid-template-columns:110px 1fr;'
            f'gap:8px;align-items:center;margin-bottom:5px">\n')
        sec8_viz += (
            f'<span style="text-align:right;font-size:12px;'
            f'color:{_model_color(model)}">{label}</span>\n')
        sec8_viz += '<div style="position:relative;height:18px">\n'
        sec8_viz += (
            f'<div style="position:absolute;top:3px;left:0;height:12px;'
            f'width:{bar_w:.1f}%;background:{_model_color(model)};'
            f'opacity:0.35;border-radius:2px"></div>\n')
        dot_left = bar_w
        sec8_viz += (
            f'<div style="position:absolute;top:2px;left:{dot_left:.1f}%;'
            f'width:8px;height:14px;margin-left:-4px;'
            f'border-radius:4px;background:{_model_color(model)};'
            f'opacity:0.9"></div>\n')
        sec8_viz += (
            f'<span style="position:absolute;top:1px;'
            f'left:calc({dot_left:.1f}% + 8px);'
            f'font-size:11px;color:#666;white-space:nowrap">'
            f'{d["mean"]:.1f} avg ({d["total"]:,} total)</span>\n')
        sec8_viz += '</div>\n'
        sec8_viz += '</div>\n'

    # Key sequence counts summary grid
    key_seqs = [
        ("seq-verify-after-edit", "edit-then-verify"),
        ("seq-edit-after-failed-verify", "fix after failure"),
        ("seq-verify-rerun-no-edit", "rerun without edit"),
        ("seq-submit-after-verify", "submit after verify"),
    ]
    sec8_viz += '<div style="margin-top:16px">\n'
    sec8_viz += ('<div style="font-size:11px;color:#999;margin-bottom:6px">'
                 'Key sequence counts per trajectory (avg)</div>\n')
    cols = '160px ' + ' '.join(['70px'] * len(models))
    sec8_viz += (f'<div style="display:grid;grid-template-columns:{cols};'
                 f'gap:4px;font-size:11px;color:#666">\n')
    sec8_viz += '<div style="font-style:italic;color:#999">sequence</div>\n'
    for m in models:
        short = _model_label(m).split()[0]
        sec8_viz += (
            f'<div style="text-align:right;color:{_model_color(m)};'
            f'font-style:italic">{short}</div>\n')
    for seq_key, seq_display in key_seqs:
        sec8_viz += f'<div>{seq_display}</div>\n'
        for m in models:
            avg = (seq_totals[m].get(seq_key, 0) / n_trajs[m]
                   if n_trajs[m] else 0)
            sec8_viz += (
                f'<div style="text-align:right;'
                f'font-variant-numeric:tabular-nums">'
                f'{avg:.1f}</div>\n')
    sec8_viz += '</div>\n</div>\n'
    sec8_viz += '</div>\n'

    sec8_phenom = (
        '<p style="font-size:13px;color:#555;margin-bottom:8px">'
        'Claude averages 6 edit-then-verify cycles per trajectory. '
        'GPT averages less than 1. The edit-verify loop is the '
        'defining structural difference.</p>')

    sec8_notes = (
        "<p>Sequence labels classify steps by their context: what "
        "happened before, whether edits or verify steps "
        "preceded them.</p>"
        "<p><strong>seq-verify-after-edit</strong>: a verify step "
        "after a source edit. The core edit-then-test loop.</p>"
        "<p><strong>seq-verify-rerun-no-edit</strong>: a verify "
        "step where no edit happened since the last verify.</p>"
        "<p><strong>seq-edit-after-failed-verify</strong>: a source "
        "edit after a failed verify step. Fixing what a test "
        "revealed.</p>"
        "<p><strong>seq-submit-after-verify</strong>: submit after "
        "at least one verify step. The agent tested before "
        "submitting.</p>"
        "<p><strong>seq-first-all-pass</strong>: the first "
        "verify-pass after the last source edit. Marks "
        "implementation completion.</p>"
        "<p>Method: classify_sequence_layer() in classify_intent.py "
        "walks the trajectory maintaining state (has a verify been "
        "seen? was there an edit since?).</p>")

    sections.append((
        "<h2>8. Sequence Labels</h2>",
        sec8_phenom + sec8_viz + _html_table(headers, tbl_rows),
        sec8_notes))

    return sections


def render_failure_sections(failure_data, models) -> list[tuple[str, str, str]]:
    if not failure_data:
        return []

    per_model = failure_data.get("per_model", {})
    models = [m for m in models if m in per_model]
    if not models:
        return []

    mode_meta = {m["key"]: m for m in failure_data.get("modes", [])}
    family_order = ["tool", "code", "test"]
    family_label = {
        "tool": "tool",
        "code": "code",
        "test": "test",
    }

    family_counts = {}
    for model in models:
        counts = {fam: 0 for fam in family_order}
        for mode, n in per_model[model].get("mode_counts", {}).items():
            fam = mode_meta.get(mode, {}).get("family", "tool")
            counts[fam] = counts.get(fam, 0) + n
        family_counts[model] = counts

    max_rate = max(
        (per_model[m]["n_failures"] / max(per_model[m]["n_steps"], 1))
        for m in models
    )

    bar_width = 560
    bar_html = '<div style="margin:16px 0 20px 0">'
    bar_html += (
        '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;'
        'font-size:11.5px;color:#666">'
    )
    for fam in family_order:
        bar_html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{FAMILY_COLOR[fam]};border-radius:2px"></span>'
            f'{family_label[fam]} failures</span>'
        )
    bar_html += '</div>'

    for model in models:
        pm = per_model[model]
        total_steps = max(pm["n_steps"], 1)
        rate = pm["n_failures"] / total_steps
        tool_share = family_counts[model]["tool"] / max(pm["n_failures"], 1) * 100
        bar_html += (
            f'<div style="display:grid;grid-template-columns:110px {bar_width}px 1fr;'
            f'align-items:center;gap:10px;margin-bottom:8px">'
            f'<div style="text-align:right;font-size:12px;color:{_model_color(model)};'
            f'font-style:italic">{_model_label(model)}</div>'
            f'<div style="height:18px;border:1px solid #e4e0d8;background:#faf8f0;'
            f'border-radius:3px;overflow:hidden;display:flex">'
        )
        for fam in family_order:
            fam_rate = family_counts[model][fam] / total_steps
            width = fam_rate / max_rate * 100 if max_rate > 0 else 0
            bar_html += (
                f'<div style="width:{width:.2f}%;background:{FAMILY_COLOR[fam]}" '
                f'title="{family_label[fam]}: {family_counts[model][fam]} '
                f'({fam_rate*100:.1f}% of all steps)"></div>'
            )
        bar_html += '</div>'
        bar_html += (
            f'<div style="font-size:11px;color:#666">'
            f'{rate*100:.1f}% of steps flagged as failures'
            f' &middot; {tool_share:.1f}% tool friction</div></div>'
        )
    bar_html += '</div>'

    gpt_counts = per_model.get("gpt5", {}).get("mode_counts", {})
    gpt_total = per_model.get("gpt5", {}).get("n_failures", 0)
    gpt_steps = per_model.get("gpt5", {}).get("n_steps", 1)
    gpt_tool_share = family_counts.get("gpt5", {}).get("tool", 0) / max(gpt_total, 1) * 100
    gpt_wrapper_modes = [
        "apply_patch_cmd_not_found",
        "apply_patch_shell_syntax",
        "apply_patch_other",
        "bash_trailing_brace",
        "bash_quote_nesting",
        "bash_heredoc_unterminated",
        "bash_syntax_error",
        "bash_broken_pipe",
    ]
    gpt_wrapper_total = sum(gpt_counts.get(k, 0) for k in gpt_wrapper_modes)
    sec_phenom = (
        '<p style="font-style:italic;color:#555;margin:0 0 14px 0">'
        f'GPT-5 records a {gpt_total / max(gpt_steps, 1) * 100:.1f}% failure-step rate. '
        f'{gpt_tool_share:.1f}% of its failures are tool-call friction, and '
        f'{gpt_wrapper_total / max(gpt_total, 1) * 100:.1f}% come from one shell-wrapper/apply_patch cluster '
        '(applypatch hallucination, trailing <code>}</code>, heredoc breakage, '
        'generic bash syntax, and the broken pipes they trigger).</p>'
    )

    sorted_modes = sorted(
        mode_meta,
        key=lambda k: (
            per_model.get("gpt5", {}).get("mode_counts", {}).get(k, 0),
            sum(per_model[m].get("mode_counts", {}).get(k, 0) for m in models),
        ),
        reverse=True,
    )
    headers = [
        "mode",
        "family",
        *[_model_label(m) for m in models],
        "GPT share",
        "GPT trajs",
    ]
    rows = []
    for mode in sorted_modes:
        meta = mode_meta[mode]
        gpt_count = per_model.get("gpt5", {}).get("mode_counts", {}).get(mode, 0)
        gpt_share = gpt_count / max(gpt_total, 1) * 100
        gpt_trajs = per_model.get("gpt5", {}).get("trajectories_with_mode", {}).get(mode, 0)
        rows.append([
            (
                f'<div>{html.escape(meta["label"])}</div>'
                f'<div style="font-size:11px;color:#888"><code>{html.escape(mode)}</code></div>'
            ),
            meta["family"],
            *[per_model[m].get("mode_counts", {}).get(mode, 0) for m in models],
            f'{gpt_share:.1f}%',
            gpt_trajs,
        ])

    def _clip(text: str, n: int) -> str:
        text = (text or "").strip()
        return text if len(text) <= n else text[: n - 1] + "…"

    sample_modes = [
        "bash_broken_pipe",
        "bash_trailing_brace",
        "apply_patch_cmd_not_found",
        "bash_heredoc_unterminated",
        "strep_invalid_range",
    ]
    sample_html = '<div style="margin:18px 0 0 0">'
    sample_html += '<h3 style="font-size:14px;font-style:italic;font-weight:400;margin:0 0 10px 0">Illustrative GPT-5 failures</h3>'
    for mode in sample_modes:
        samples = per_model.get("gpt5", {}).get("samples", {}).get(mode, [])
        if not samples:
            continue
        sample = samples[0]
        meta = mode_meta[mode]
        count = gpt_counts.get(mode, 0)
        sample_html += (
            '<div style="margin:0 0 14px 0;padding:10px 12px;border-left:3px solid '
            f'{FAMILY_COLOR.get(meta["family"], "#999")};background:#fcfbf6">'
            f'<div style="font-size:12.5px;color:#444;margin-bottom:5px">'
            f'<strong>{html.escape(meta["label"])}</strong> '
            f'<span style="color:#888">({count} GPT-5 steps)</span></div>'
            f'<div style="font-size:11.5px;color:#666;margin-bottom:6px">{html.escape(meta["desc"])}</div>'
            f'<div style="font-size:11px;color:#888;margin-bottom:4px">{html.escape(sample["instance"])}</div>'
            f'<pre style="margin:0 0 6px 0;padding:8px;background:#fffffb;border:1px solid #ece7da;white-space:pre-wrap;overflow-x:auto;font-size:11px"><strong>action</strong>\n{html.escape(_clip(sample.get("action", ""), 320))}</pre>'
            f'<pre style="margin:0;padding:8px;background:#fffffb;border:1px solid #ece7da;white-space:pre-wrap;overflow-x:auto;font-size:11px"><strong>observation</strong>\n{html.escape(_clip(sample.get("observation", ""), 420))}</pre>'
            '</div>'
        )
    sample_html += '</div>'

    notes = (
        '<p>This section uses <code>data/failure_modes.json</code>, produced by '
        '<code>scripts/build_failure_modes.py</code>. Each trajectory step is classified as '
        'either not-a-failure or one failure mode.</p>'
        '<p><strong>Counts are step counts</strong>, not unique incidents. A single trajectory '
        'can contribute many failure steps, and one underlying shell mistake can fan out into '
        'multiple observed failures.</p>'
        '<p><strong>Families</strong>: <em>tool</em> means the harness/tool call itself failed; '
        '<em>code</em> means the agent ran code that crashed; <em>test</em> means a test runner '
        'reported failures.</p>'
        '<p><strong>Interpretation caveat</strong>: <code>bash_broken_pipe</code> is often a '
        'secondary symptom. In GPT-5, many of those steps are downstream of the same wrapper '
        'pathologies that also produce trailing-brace, heredoc, or quoting failures.</p>'
        '<p>The distinctive GPT-5 signature is not ordinary test failure. It is repeated '
        'interaction friction around shell wrapping and hallucinated <code>applypatch</code> usage.</p>'
    )

    return [(
        "<h2>8b. Failure Modes</h2>",
        sec_phenom + bar_html + _html_table(headers, rows) + sample_html,
        notes,
    )]


def _lerp_color(hex_color: str, t: float) -> str:
    """Blend between #fffff8 (page bg) and hex_color. t=0 → bg, t=1 → full color."""
    bg = (255, 255, 248)
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    mr = round(bg[0] + (r - bg[0]) * t)
    mg = round(bg[1] + (g - bg[1]) * t)
    mb = round(bg[2] + (b - bg[2]) * t)
    return f"rgb({mr},{mg},{mb})"


def render_phase_profile_section(results, models) -> list[tuple[str, str, str]]:
    """Section 11: Phase profile heatmaps (pure HTML, no JS)."""
    from analysis.models import HIGH_LEVEL_LETTER, LETTER_TO_NAME, HIGH_LEVEL_COLORS
    import math

    letters = ["R", "S", "P", "E", "V", "G", "H"]
    bins = 20

    # Compute phase profiles from FileResult data
    phase_data = {}  # model -> letter -> [bin_avg, ...]
    for model in models:
        data = results[model]
        letter_bins = {l: [] for l in letters}
        for r in data:
            if not r.phase_profile:
                continue
            for l in letters:
                if l in r.phase_profile:
                    letter_bins[l].append(r.phase_profile[l])

        avg = {}
        for l in letters:
            profiles = letter_bins[l]
            if not profiles:
                avg[l] = [0.0] * bins
            else:
                avg[l] = [sum(p[b] for p in profiles) / len(profiles) for b in range(bins)]
        phase_data[model] = avg

    # Build HTML heatmaps
    viz_html = ""
    for model in models:
        avg = phase_data[model]
        label = _model_label(model)
        color = _model_color(model)

        # Renormalize per bin (so columns sum to 1)
        bin_sums = [0.0] * bins
        for l in letters:
            for b in range(bins):
                bin_sums[b] += avg[l][b]

        renormed = {}
        for l in letters:
            renormed[l] = [avg[l][b] / bin_sums[b] if bin_sums[b] > 0 else 0 for b in range(bins)]

        viz_html += f'<div style="margin-bottom:20px"><div style="font-style:italic;color:{color};margin-bottom:6px">{label}</div>'
        viz_html += '<div style="display:grid;grid-template-columns:90px repeat(20,1fr);gap:2px;font-size:10px">'

        # Header row
        viz_html += '<div></div>'
        for b in range(bins):
            viz_html += f'<div style="text-align:center;color:#999">{b*5}%</div>' if b % 5 == 0 else '<div></div>'

        # Data rows
        for l in letters:
            cat_name = LETTER_TO_NAME.get(l, l)
            cat_color = HIGH_LEVEL_COLORS.get(cat_name, "#888")
            viz_html += f'<div style="display:flex;align-items:center;color:#555;font-size:11px">{cat_name}</div>'

            vals = renormed[l]
            max_v = max(vals) if vals else 0.01

            for b in range(bins):
                v = vals[b]
                pct_val = v * 100
                ratio = v / max_v if max_v > 0 else 0
                t = 0 if pct_val < 0.5 else 0.15 + math.sqrt(ratio) * 0.85
                bg = _lerp_color(cat_color, t)
                text_color = "rgba(255,255,255,0.9)" if t > 0.5 else "rgba(0,0,0,0.45)"
                viz_html += (
                    f'<div style="height:26px;border-radius:2px;display:flex;align-items:center;'
                    f'justify-content:center;background:{bg};color:{text_color}">'
                    f'{pct_val:.0f}%</div>'
                )

        viz_html += '</div></div>'

    sections = [(
        "<h2>11. Phase Profile Heatmap</h2>",
        f'<p style="color:#777;font-style:italic;margin-bottom:14px">Each trajectory divided into 20 time-slices. '
        f'Cell intensity shows what proportion of steps in each slice belong to each category, normalized per column.</p>'
        + viz_html,
        "<p>Read left-to-right as beginning to end of trajectory. Brighter cells indicate the dominant action in that time-slice.</p>"
        "<p>Categories: read, search, reproduce, edit, verify, git, housekeeping. Failed and other are excluded.</p>"
        "<p>Normalized per column: within each time-slice, the percentages show each category's share relative to only the displayed categories.</p>"
        "<p>Method: each trajectory's step sequence is divided into 20 equal bins. Per bin, we count the fraction of steps belonging to each category, then average across all trajectories for that model.</p>"
    )]
    return sections


def render_repo_section(results, models) -> list[tuple[str, str, str]]:
    """Section 12: Per-repo breakdown with dot-plot visualization.

    Returns list of (title_html, content_html, notes_html) tuples.
    Shows resolve rate for each model across top repos as a dot plot,
    making repo-specific model advantages immediately visible.
    """
    # ── Gather per-repo, per-model stats ──────────────────
    repos = set()
    for data in results.values():
        for r in data:
            repos.add(r.repo)

    repo_stats = {}  # repo -> {model -> {n, resolve_rate, avg_steps, verify_pct}}
    for repo in repos:
        repo_stats[repo] = {}
        for model in models:
            sub = [r for r in results[model] if r.repo == repo]
            if not sub:
                continue
            steps = [r.steps for r in sub]
            resolved = sum(1 for r in sub if r.resolved)
            total_steps = sum(steps)
            verify = sum(r.high_intent_counts.get("verify", 0) for r in sub)
            repo_stats[repo][model] = {
                "n": len(sub),
                "resolve_rate": round(resolved / len(sub) * 100, 1),
                "avg_steps": round(statistics.mean(steps), 1),
                "verify_pct": round(verify / total_steps * 100, 1) if total_steps else 0,
            }

    # Sort by total instances (most common repos first), take top 12
    ranked_repos = sorted(repos,
        key=lambda rp: -sum(repo_stats[rp].get(m, {}).get("n", 0) for m in models))
    top_repos = ranked_repos[:12]

    # ── Data table ────────────────────────────────────────
    headers = ["repo"]
    for m in models:
        headers.extend([f"{_model_label(m)}_n", f"{_model_label(m)}_avg",
                        f"{_model_label(m)}_res%", f"{_model_label(m)}_ver%"])
    tbl_rows = []
    for repo in top_repos:
        short = repo.split("/")[-1] if "/" in repo else repo
        row = [short]
        for model in models:
            s = repo_stats[repo].get(model)
            if not s:
                row.extend(["", "", "", ""])
            else:
                row.extend([s["n"], s["avg_steps"],
                            f'{s["resolve_rate"]:.1f}', f'{s["verify_pct"]:.1f}'])
        tbl_rows.append(row)

    # ── Dot-plot: resolve rate by repo, one dot per model ─
    #
    # Each row is a repo. The horizontal axis runs 0--100% resolve rate.
    # Each model is a colored dot. Clustering means models agree;
    # spread means the repo differentiates them. Pure HTML/CSS.

    plot_width = 520   # px, data area
    row_height = 24
    label_width = 150

    # Build rows
    dot_rows_html = ""
    for repo in top_repos:
        short = repo.split("/")[-1] if "/" in repo else repo
        n_total = sum(repo_stats[repo].get(m, {}).get("n", 0) for m in models)

        dot_rows_html += (
            f'<div style="display:flex;align-items:center;height:{row_height}px">'
            f'<div style="width:{label_width}px;text-align:right;padding-right:12px;'
            f'font-size:11.5px;color:#555;white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis" title="{repo}">{short}'
            f' <span style="color:#aaa;font-size:10px">({n_total})</span></div>'
            f'<div style="position:relative;width:{plot_width}px;height:{row_height}px;'
            f'border-bottom:1px solid #f0f0e8">'
        )

        for model in models:
            s = repo_stats[repo].get(model)
            if not s or s["n"] < 3:
                continue  # need at least 3 tasks for a meaningful rate
            rate = s["resolve_rate"]
            left = rate / 100 * plot_width
            color = _model_color(model)
            dot_rows_html += (
                f'<div style="position:absolute;left:{left:.1f}px;'
                f'top:{(row_height - 8) / 2:.0f}px;'
                f'width:8px;height:8px;border-radius:50%;background:{color};'
                f'opacity:0.82" '
                f'title="{_model_label(model)}: {rate:.1f}% ({s["n"]} tasks)"></div>'
            )

        dot_rows_html += "</div></div>\n"

    # Axis tick marks
    tick_html = (
        f'<div style="display:flex;align-items:flex-start;height:16px;margin-top:2px">'
        f'<div style="width:{label_width}px"></div>'
        f'<div style="position:relative;width:{plot_width}px;height:16px;'
        f'font-size:10px;color:#999">'
    )
    for pct in [0, 20, 40, 60, 80, 100]:
        left = pct / 100 * plot_width
        tick_html += (
            f'<span style="position:absolute;left:{left}px;'
            f'transform:translateX(-50%)">{pct}%</span>'
        )
    tick_html += "</div></div>"

    # Vertical gridlines via repeating gradient
    grid_stops = []
    for pct in [20, 40, 60, 80]:
        px = pct / 100 * plot_width
        grid_stops.append(
            f"transparent {px - 0.5:.1f}px,"
            f"#ede8de {px - 0.5:.1f}px,"
            f"#ede8de {px + 0.5:.1f}px,"
            f"transparent {px + 0.5:.1f}px"
        )
    gridlines_css = f"background:repeating-linear-gradient(to right,{','.join(grid_stops)});"

    # Legend
    legend_html = (
        '<div style="display:flex;gap:18px;margin:6px 0 2px 0;font-size:11.5px">'
    )
    for model in models:
        color = _model_color(model)
        legend_html += (
            f'<span style="display:inline-flex;align-items:center;gap:4px">'
            f'<span style="display:inline-block;width:8px;height:8px;'
            f'border-radius:50%;background:{color}"></span>'
            f'<span style="color:#555">{_model_label(model)}</span></span>'
        )
    legend_html += "</div>"

    chart_html = (
        f'<div style="margin:16px 0 20px 0;font-family:Palatino,Georgia,serif">'
        f'<div style="font-size:12.5px;color:#777;margin-bottom:4px;font-style:italic">'
        f'Resolve rate by repository (each dot = one model, repos with &lt;3 tasks omitted)</div>'
        f'{legend_html}'
        f'<div style="margin-top:6px;padding-left:0;{gridlines_css}">'
        f'{dot_rows_html}'
        f'</div>'
        f'{tick_html}'
        f'</div>'
    )

    # ── Spread table: max-min resolve rate per repo ───────
    spread_rows = []
    for repo in top_repos:
        rates = [
            repo_stats[repo][m]["resolve_rate"]
            for m in models
            if m in repo_stats[repo] and repo_stats[repo][m]["n"] >= 3
        ]
        if len(rates) < 2:
            continue
        spread = max(rates) - min(rates)
        best = max(
            (m for m in models
             if m in repo_stats[repo] and repo_stats[repo][m]["n"] >= 3),
            key=lambda m: repo_stats[repo][m]["resolve_rate"])
        short = repo.split("/")[-1] if "/" in repo else repo
        spread_rows.append([
            short,
            f"{min(rates):.1f}",
            f"{max(rates):.1f}",
            f"{spread:.1f}",
            _model_label(best),
        ])
    spread_rows.sort(key=lambda r: -float(r[3]))
    spread_table = _html_table(
        ["repo", "min res%", "max res%", "spread (pp)", "best model"],
        spread_rows,
        caption="Cross-model spread in resolve rate, largest first")

    content_html = chart_html + "\n" + _html_table(headers, tbl_rows) + "\n" + spread_table

    notes_html = (
        "<p>Metrics broken down by source repository. SWE-Bench Pro tasks come from ~11 "
        "open-source repos. The dot plot shows whether one model dominates uniformly or "
        "whether there is repo-specific variation.</p>"
        "<p><strong>Dot plot</strong>: each dot is one model's resolve rate on that repo. "
        "When dots cluster, models perform similarly on that repo; when they spread apart, "
        "the repo differentiates models. Repos with fewer than 3 tasks per model are "
        "omitted from the plot to avoid noisy rates.</p>"
        "<p><strong>Spread table</strong>: the gap (in percentage points) between the best "
        "and worst model on each repo. Large spreads indicate repos where model choice "
        "matters most.</p>"
        "<p><strong>_n</strong>: number of task instances from this repo.</p>"
        "<p><strong>_avg</strong>: average steps per trajectory.</p>"
        "<p><strong>_res%</strong>: resolve rate (percentage of trajectories where the "
        "submitted patch fixes the failing tests).</p>"
        "<p><strong>_ver%</strong>: percentage of steps spent on verify actions.</p>"
        "<p>Sorted by total number of instances across all models (most common repos "
        "first). Top 12 shown.</p>"
    )

    return [("<h2>12. Per-Repo Breakdown</h2>", content_html, notes_html)]


# ── Intent visualization helpers ─────────────────────────

CAT_COLORS = {
    "read": "#5a7d9a", "search": "#5a7d9a", "reproduce": "#b0956a",
    "edit": "#4a8a5a", "verify": "#b56a50", "git": "#3a8a8a",
    "housekeeping": "#3a8a8a", "failed": "#a05050", "other": "#888",
}
PHASE_COLORS = {
    "understand": "#5a7d9a", "reproduce": "#b0956a",
    "edit": "#4a8a5a", "verify": "#b56a50", "cleanup": "#3a8a8a",
}
PHASE_MAP = {
    "understand": ["read", "search"], "reproduce": ["reproduce"],
    "edit": ["edit"], "verify": ["verify"], "cleanup": ["git", "housekeeping"],
}


def _stacked_bar(model_name, segments, color_map, bar_height=22):
    """One horizontal stacked bar.  segments: [(label, fraction), ...]."""
    parts = []
    for label, frac in segments:
        if frac < 0.005:
            continue
        pct = frac * 100
        col = color_map.get(label, "#888")
        text = (f"{label} {pct:.0f}%" if pct >= 6
                else (f"{pct:.0f}" if pct >= 3 else ""))
        parts.append(
            f'<span style="display:inline-block;width:{pct:.2f}%;height:{bar_height}px;'
            f'background:{col};color:#fff;font-size:10px;line-height:{bar_height}px;'
            f'text-align:center;overflow:hidden;white-space:nowrap;vertical-align:middle"'
            f' title="{label}: {pct:.1f}%">{text}</span>'
        )
    bar = "".join(parts)
    label_text = _model_label(model_name)
    return (
        f'<div style="display:flex;align-items:center;margin:3px 0">'
        f'<span style="width:100px;font-size:11px;color:#555;text-align:right;'
        f'padding-right:8px;flex-shrink:0">{label_text}</span>'
        f'<div style="flex:1;background:#f0efe8;border-radius:2px;overflow:hidden;'
        f'line-height:0">{bar}</div></div>'
    )


def _stacked_bar_chart(models, model_data, labels, color_map, title=""):
    """Set of stacked bars (one per model).

    model_data: dict[model, dict[label, fraction]].
    labels: ordered list of segment labels.
    """
    bars = "".join(
        _stacked_bar(m, [(lbl, model_data.get(m, {}).get(lbl, 0)) for lbl in labels],
                     color_map)
        for m in models
    )
    legend_items = "".join(
        f'<span style="display:inline-block;width:10px;height:10px;background:'
        f'{color_map.get(lbl, "#888")};border-radius:1px;margin:0 3px 0 10px;'
        f'vertical-align:middle"></span>'
        f'<span style="font-size:10px;color:#666;vertical-align:middle">{lbl}</span>'
        for lbl in labels
    )
    legend = f'<div style="margin:6px 0 4px 108px">{legend_items}</div>'
    heading = (f'<div style="font-size:11px;color:#888;margin-bottom:4px;'
               f'font-style:italic">{title}</div>') if title else ""
    return f'<div style="margin:12px 0 16px 0;max-width:800px">{heading}{bars}{legend}</div>'


def _paired_bar_chart(intents_pct, models, top_n=10):
    """Top-N intents as small grouped bars (one thin bar per model).

    intents_pct: dict[intent, dict[model, pct_float]] (0-100 scale).
    """
    ranked = sorted(intents_pct.keys(),
                    key=lambda i: max(intents_pct[i].values()), reverse=True)[:top_n]
    if not ranked:
        return ""
    max_pct = max(intents_pct[i][m] for i in ranked for m in models)
    scale = 100.0 / max_pct if max_pct > 0 else 1

    rows_html = []
    for intent in ranked:
        bar_lines = []
        for m in models:
            pct = intents_pct[intent].get(m, 0)
            col = _model_color(m)
            w = pct * scale
            bar_lines.append(
                f'<div style="display:flex;align-items:center;margin:1px 0">'
                f'<div style="width:{w:.1f}%;height:6px;background:{col};'
                f'border-radius:1px;min-width:1px"></div>'
                f'<span style="font-size:9px;color:#888;margin-left:4px">'
                f'{pct:.1f}%</span></div>'
            )
        rows_html.append(
            f'<div style="display:flex;align-items:flex-start;margin:5px 0">'
            f'<span style="width:130px;font-size:10.5px;color:#555;text-align:right;'
            f'padding-right:8px;flex-shrink:0;line-height:1.3">{intent}</span>'
            f'<div style="flex:1">{"".join(bar_lines)}</div></div>'
        )
    legend_items = " ".join(
        f'<span style="display:inline-block;width:8px;height:8px;background:'
        f'{_model_color(m)};border-radius:1px;margin:0 3px 0 8px;'
        f'vertical-align:middle"></span><span style="font-size:10px;color:#666;'
        f'vertical-align:middle">{_model_label(m)}</span>'
        for m in models
    )
    legend = f'<div style="margin:2px 0 8px 138px">{legend_items}</div>'
    return f'<div style="margin:12px 0 16px 0;max-width:700px">{legend}{"".join(rows_html)}</div>'


# ── Intent Classification Taxonomy ─────────────────────────
#
# For each base intent we describe (a) what it means and (b) the literal rule
# in classify_intent.py that decides the label. The rule text is intentionally
# a tight English paraphrase of the pseudocode in
# docs/intent-classification-rules.md -- it should be possible to read the
# table and match a step's action string to a label without looking at the
# code.

_TAXONOMY_RULES: dict[str, str] = {
    # read
    "read-file-full":            "str_replace_editor view &lt;file&gt; (fallback once test, config, range, and truncated views are ruled out)",
    "read-file-range":           "str_replace_editor view with <code>--view_range</code>",
    "read-file-full(truncated)": "str_replace_editor view where the observation contains <code>too large to display</code>",
    "read-test-file":            "str_replace_editor view on a filename matching <code>test_*</code>, <code>*_test.*</code>, or <code>conftest*</code>",
    "read-config-file":          "str_replace_editor view on <code>package.json</code>, <code>pytest.ini</code>, <code>setup.cfg</code>, <code>setup.py</code>, <code>go.mod</code>, <code>Makefile</code>, <code>config.json</code>",
    "read-via-bash":             "<code>cat</code>, <code>head</code>, <code>tail</code>, <code>sed -n</code>, <code>nl</code>, <code>awk</code>",
    # search
    "view-directory":            "str_replace_editor view where path has no extension, or observation lists &ldquo;files and directories&rdquo;",
    "list-directory":            "<code>ls</code>, <code>tree</code>, <code>pwd</code>",
    "search-keyword":            "<code>grep</code>, <code>rg</code>, <code>ag</code>",
    "search-files-by-name":      "<code>find ... -name</code> with no grep/xargs pipe",
    "search-files-by-content":   "<code>find ... -exec grep</code> or <code>find ... | xargs grep</code>",
    "inspect-file-metadata":     "<code>wc</code>, <code>file</code>, <code>stat</code>",
    # reproduce
    "create-repro-script":       "str_replace_editor create on a filename containing <code>repro</code>, <code>reproduce</code>, or <code>demo</code>",
    "run-repro-script":          "run a named script whose basename matches <code>repro*</code> or <code>reproduce*</code> (python, node, sh, bash, go run)",
    "run-inline-snippet":        "<code>python -c</code>, <code>python - &lt;&lt;</code>, <code>node -e</code> &mdash; residual when no inline sub-pattern matches",
    # inline snippet sub-intents
    "run-inline-verify":         "inline snippet with <code>import/from</code> + <code>assert</code>/<code>print</code> (smoke test or assertion)",
    "read-via-inline-script":    "inline snippet that reads a file (<code>.read()</code>, <code>open(...,'r')</code>, <code>readFileSync</code>) and prints, without writing",
    "edit-via-inline-script":    "inline snippet that writes (<code>.write()</code>, <code>writeFileSync</code>) together with reading or <code>.replace()</code>/<code>re.sub()</code>",
    "create-file-via-inline-script": "inline snippet that writes a file with no prior read",
    "check-version":             "inline snippet matching <code>--version</code>, <code>-V</code>, <code>sys.version</code>, or <code>node -v</code>",
    # edit
    "edit-source":               "str_replace_editor str_replace on a filename <em>not</em> matching test/repro/verify/check",
    "insert-source":             "str_replace_editor insert",
    "apply-patch":               "<code>applypatch</code> command (GPT-specific)",
    "create-file":               "str_replace_editor create on a filename <em>not</em> matching repro/test/verify/doc patterns",
    # verify
    "run-test-suite":            "<code>pytest</code>, <code>go test</code>, <code>npm test</code>, <code>npx jest</code>, <code>mocha</code>, <code>yarn test</code>, <code>python -m unittest</code> (broad; no <code>::</code> or <code>-k</code>)",
    "run-test-specific":         "a test runner command containing <code>::</code> or <code> -k </code>",
    "create-test-script":        "str_replace_editor create on a filename matching <code>test_*</code>, <code>*test.py</code>, <code>*test.js</code>, <code>*test.go</code>",
    "run-verify-script":         "run a named script whose basename contains <code>test_</code>, <code>verify</code>, <code>check</code>, <code>validate</code>, or <code>edge_case</code>",
    "create-verify-script":      "str_replace_editor create on a filename matching <code>verify*</code>, <code>check*</code>, or <code>validate*</code>",
    "edit-test-or-repro":        "str_replace_editor str_replace on a filename containing <code>test_</code>, <code>repro</code>, <code>verify</code>, or <code>check</code>",
    "run-custom-script":         "run a named python/node/sh/bash/go script whose basename doesn&rsquo;t match repro/test/verify patterns",
    "syntax-check":              "<code>py_compile</code>, <code>compileall</code>, <code>node -c</code>",
    "compile-build":             "<code>go build</code>, <code>go vet</code>, <code>make</code>, <code>tsc</code>, <code>npx tsc</code>, <code>npm run build</code>, <code>yarn build</code>",
    # git
    "git-diff":                  "<code>git diff</code> (with or without <code>-C &lt;dir&gt;</code>)",
    "git-status-log":            "<code>git status</code>, <code>git show</code>, <code>git log</code>",
    "git-stash":                 "<code>git stash</code>",
    # housekeeping
    "file-cleanup":              "<code>rm</code>, <code>mv</code>, <code>cp</code>, <code>chmod</code>",
    "create-documentation":      "str_replace_editor create on a filename matching <code>*summary*</code>, <code>*readme*</code>, <code>*changes*</code>, <code>*implementation*</code>",
    "start-service":             "<code>redis-server</code>, <code>redis-cli</code>, <code>mongod</code>, <code>sleep</code>",
    "install-deps":              "<code>pip install</code>, <code>pip list</code>, <code>npm install</code>, <code>go get</code>, <code>apt</code>",
    "check-tool-exists":         "<code>which</code>, <code>type</code>",
    # failed  (observation contains syntax error / command not found / etc.)
    "search-keyword(failed)":    "<code>grep</code>/<code>find</code> whose observation contains a shell error",
    "read-via-bash(failed)":     "<code>cat</code>/<code>head</code>/<code>sed</code>/<code>tail</code>/<code>ls</code> whose observation contains a shell error",
    "run-script(failed)":        "<code>python</code>/<code>node</code> whose observation contains a shell error",
    "run-test-suite(failed)":    "test runner whose observation contains a shell error",
    "bash-command(failed)":      "any other bash command whose observation contains a shell error",
    # other
    "submit":                    "action&rsquo;s first line starts with <code>submit</code>",
    "empty":                     "action string is blank (rate-limit or context-window exit)",
    "echo":                      "<code>echo</code>, <code>printf</code>",
    "bash-other":                "final fallback &mdash; bash command that matched no other rule (&lt;2% of steps by design)",
    "undo-edit":                 "<code>str_replace_editor undo_edit</code>",
}

# Presentation order: high-level category, then the intents within it.
_TAXONOMY_ORDER: list[tuple[str, list[str]]] = [
    ("read", [
        "read-file-full", "read-file-range", "read-file-full(truncated)",
        "read-test-file", "read-config-file", "read-via-bash",
        "read-via-inline-script",
    ]),
    ("search", [
        "view-directory", "list-directory", "search-keyword",
        "search-files-by-name", "search-files-by-content",
        "inspect-file-metadata", "check-version",
    ]),
    ("reproduce", [
        "create-repro-script", "run-repro-script", "run-inline-snippet",
    ]),
    ("edit", [
        "edit-source", "insert-source", "apply-patch", "create-file",
        "edit-via-inline-script", "create-file-via-inline-script",
    ]),
    ("verify", [
        "run-test-suite", "run-test-specific", "create-test-script",
        "run-verify-script", "create-verify-script", "edit-test-or-repro",
        "run-custom-script", "syntax-check", "compile-build",
        "run-inline-verify",
    ]),
    ("git", ["git-diff", "git-status-log", "git-stash"]),
    ("housekeeping", [
        "file-cleanup", "create-documentation", "start-service",
        "install-deps", "check-tool-exists",
    ]),
    ("failed", [
        "search-keyword(failed)", "read-via-bash(failed)",
        "run-script(failed)", "run-test-suite(failed)",
        "bash-command(failed)",
    ]),
    ("other", ["submit", "empty", "echo", "bash-other", "undo-edit"]),
]


def render_taxonomy_section(results, models) -> list[tuple[str, str, str]]:
    """One Tufte-style table: high-level category &rarr; base intent &rarr; description &rarr; matching rule.

    Also includes per-model step counts so the reader can see which intents actually fire.
    """
    # Aggregate counts per intent per model.
    intent_totals: dict[str, dict[str, int]] = {i: {} for i in INTENT_TO_HIGH_LEVEL}
    for model in models:
        for r in results[model]:
            for intent, n in r.base_intent_counts.items():
                intent_totals.setdefault(intent, {})
                intent_totals[intent][model] = intent_totals[intent].get(model, 0) + n

    def fmt_count(n: int) -> str:
        if n == 0:
            return '<span style="color:#ccc">0</span>'
        if n >= 1000:
            return f"{n/1000:.1f}k"
        return str(n)

    # Build the table: a header row per category, intent rows underneath.
    n_model_cols = len(models)
    model_headers = "".join(
        f'<th style="text-align:right;font-size:10.5px;color:{MODELS[m]["color"]};font-weight:600">'
        f'{MODELS[m]["label"]}</th>'
        for m in models
    )

    rows_html = []
    for category, intents in _TAXONOMY_ORDER:
        cat_color = HIGH_LEVEL_COLORS.get(category, "#666")
        rows_html.append(
            f'<tr style="background:#faf7ed">'
            f'<td colspan="{3 + n_model_cols}" '
            f'style="padding-top:8px;padding-bottom:4px;border-bottom:1px solid #d0d0d0">'
            f'<span style="display:inline-block;width:8px;height:8px;background:{cat_color};'
            f'border-radius:1px;margin-right:8px;vertical-align:middle"></span>'
            f'<span style="font-family:\'Palatino Linotype\',serif;font-size:13px;'
            f'font-style:italic;color:{cat_color}">{category}</span>'
            f'</td></tr>'
        )
        for intent in intents:
            desc = INTENT_DESCRIPTIONS.get(intent, "")
            rule = _TAXONOMY_RULES.get(intent, "")
            counts_cells = "".join(
                f'<td style="text-align:right;font-variant-numeric:tabular-nums;color:#888">'
                f'{fmt_count(intent_totals.get(intent, {}).get(m, 0))}</td>'
                for m in models
            )
            rows_html.append(
                f'<tr>'
                f'<td style="font-family:\'SF Mono\',Menlo,Consolas,monospace;font-size:11.5px;'
                f'white-space:nowrap;padding-left:20px;color:#333">{intent}</td>'
                f'<td style="font-size:12px;color:#555">{desc}</td>'
                f'<td style="font-size:12px;color:#444;line-height:1.45">{rule}</td>'
                f'{counts_cells}</tr>'
            )

    table = (
        '<table style="border-collapse:collapse;width:100%;margin-bottom:16px">'
        '<thead><tr>'
        '<th style="text-align:left;width:180px">intent</th>'
        '<th style="text-align:left;width:22%">description</th>'
        '<th style="text-align:left">classification rule</th>'
        f'{model_headers}'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
    )

    phenomenon = (
        '<p style="font-size:13px;color:#444;margin:0 0 8px 0">'
        'Every step is labelled by a deterministic, priority-ordered ruleset in '
        '<code>scripts/classify_intent.py</code> &mdash; no model inference. '
        'Each row shows the intent, what it means, the literal rule that fires it, '
        'and how often it fires per model.'
        '</p>'
    )

    algorithm = (
        '<div style="font-size:12px;color:#444;background:#f5f1e4;border-left:3px solid #c8b88a;'
        'padding:10px 14px;margin:10px 0 16px 0;line-height:1.55;max-width:900px">'
        '<p style="margin:0 0 4px 0"><strong>Classification order</strong> '
        '(first rule that matches wins):</p>'
        '<ol style="margin:4px 0 0 18px;padding:0">'
        '<li>Empty action &rarr; <code>empty</code>. <code>submit</code> prefix &rarr; <code>submit</code>.</li>'
        '<li><code>str_replace_editor {view, create, str_replace, insert, undo_edit}</code> '
        '&rarr; classified by sub-command and filename pattern (test/config/repro/verify/doc).</li>'
        '<li>Bash is unwrapped: strip <code>bash -lc "..."</code>, leading <code>cd ... &amp;&amp;</code>, '
        '<code>source ... &amp;&amp;</code>, <code>timeout N</code>, and <code>FOO=bar</code> env prefixes.</li>'
        '<li>If the observation shows a shell-level error '
        '(<code>syntax error</code>, <code>command not found</code>, <code>unexpected token</code>, &hellip;), '
        'the command head routes to the matching <code>(failed)</code> label.</li>'
        '<li>Otherwise match the command head: test runners, compile/syntax, search, read, list, '
        'git, python/node scripts (named vs. inline, with inline further sub-classified by code shape), '
        'file cleanup, install, service, tool-exists, metadata, echo.</li>'
        '<li>Anything that reached the end is <code>bash-other</code> (&lt;2% of steps by design).</li>'
        '</ol>'
        '</div>'
    )

    notes = (
        '<p>The label describes <em>what the command is</em>, derived from the action '
        'string and filename alone &mdash; no positional context (before/after first edit) '
        'and no outcome signal is used. A failed grep is still a search attempt.</p>'
        '<p><strong>(failed)</strong> variants classify by intended action, not outcome. '
        'They require a shell-level error in the first 500 chars of the observation.</p>'
        '<p><strong>run-inline-snippet</strong> is a residual &mdash; inline snippets '
        '(<code>python -c</code>, <code>python - &lt;&lt;</code>, <code>node -e</code>) are first '
        'routed to <code>run-inline-verify</code> / <code>read-via-inline-script</code> / '
        '<code>edit-via-inline-script</code> / <code>create-file-via-inline-script</code> / '
        '<code>check-version</code> by inspecting the code shape.</p>'
        '<p><strong>Pass/fail outcome</strong> for verify intents (used by '
        '<code>seq-first-all-pass</code> / <code>seq-work-done</code>) is a separate detector '
        'that reads the observation for unambiguous runner summaries: '
        'e.g.&nbsp;pytest <code>N passed in Xs</code> / <code>N failed</code>, '
        'go <code>ok package</code> / <code>FAIL package</code>, '
        'jest <code>Tests: N passed</code> / <code>N failed</code>. '
        'Ambiguous output returns unknown.</p>'
        '<p>Canonical source: <code>scripts/classify_intent.py</code> and '
        '<code>docs/intent-classification-rules.md</code>.</p>'
    )

    return [(
        "<h2>1b. Intent Classification Taxonomy</h2>",
        phenomenon + algorithm + table,
        notes,
    )]


def render_intent_sections(results, models):
    """Return sections 2, 3, 4 as list of (title_html, content_html, notes_html).

    Each section: one-line phenomenon, inline visualization, data table, notes.
    results: dict[str, list[FileResult]] from process_all.
    models: sorted list of model keys.
    """
    sections = []

    # ── 2. Base Intent Frequencies ──────────────────────────
    totals = {}
    for model in models:
        c = Counter()
        for r in results[model]:
            c.update(r.base_intent_counts)
        totals[model] = c
    all_intents = sorted(set().union(*[set(c.keys()) for c in totals.values()]))
    model_totals = {m: sum(totals[m].values()) for m in models}

    intents_pct = {}
    for intent in all_intents:
        intents_pct[intent] = {
            m: (totals[m].get(intent, 0) / model_totals[m] * 100) if model_totals[m] else 0
            for m in models
        }

    intent_rows = []
    for intent in all_intents:
        row = [intent]
        for m in models:
            n = totals[m].get(intent, 0)
            row.extend([n, _pct(n, model_totals[m])])
        intent_rows.append(row)
    intent_rows.sort(key=lambda r: -sum(r[i] for i in range(1, len(r), 2) if isinstance(r[i], int)))
    headers = ["intent"]
    for m in models:
        headers.extend([f"{m}_n", f"{m}_%"])

    phenomenon = ('<p style="font-size:13px;color:#444;margin:0 0 8px 0">'
                  'The top 10 intents account for the bulk of all steps; '
                  'the long tail of ~40 others fills in the edges.</p>')
    viz = _paired_bar_chart(intents_pct, models, top_n=10)
    table = _html_table(headers, intent_rows)

    sections.append((
        "<h2>2. Base Intent Frequencies</h2>",
        phenomenon + viz + table,
        "<p>Every trajectory step is classified into one of ~50 base intents using "
        "deterministic rules (regex matching on the action string, file names, and "
        "observation text). No LLM is used for classification.</p>"
        "<p><strong>_n</strong>: total count of that intent across all trajectories "
        "for the model.</p>"
        "<p><strong>_%</strong>: that count as a percentage of all steps for the model. "
        "Percentages sum to 100% within each model.</p>"
        "<p>Sorted by total count across all models (most frequent first).</p>"
        "<p>Method: each step's action string is pattern-matched against a "
        "priority-ordered ruleset in classify_intent.py. For example, an action "
        "starting with <code>str_replace_editor view</code> is classified as a read "
        "intent, while <code>grep</code> or <code>rg</code> becomes search-keyword.</p>"
    ))

    # ── 3. High-Level Category Frequencies ──────────────────
    cats = ["read", "search", "reproduce", "edit", "verify",
            "git", "housekeeping", "failed", "other"]

    cat_fracs = {}
    for model in models:
        total_steps = sum(r.steps for r in results[model])
        cat_fracs[model] = {}
        for cat in cats:
            cat_steps = sum(r.high_intent_counts.get(cat, 0) for r in results[model])
            cat_fracs[model][cat] = cat_steps / total_steps if total_steps else 0

    headers = ["category"]
    for m in models:
        headers.extend([f"{m}_n", f"{m}_%", f"{m}_per_traj"])
    tbl_rows = []
    for cat in cats:
        row = [cat]
        for model in models:
            total_steps = sum(r.steps for r in results[model])
            n_trajs = len(results[model])
            cat_steps = sum(r.high_intent_counts.get(cat, 0) for r in results[model])
            row.extend([cat_steps,
                        _pct(cat_steps, total_steps),
                        round(cat_steps / n_trajs, 1) if n_trajs else 0])
        tbl_rows.append(row)

    phenomenon = ('<p style="font-size:13px;color:#444;margin:0 0 8px 0">'
                  'Read and search dominate every model; the gap is in verify '
                  'and edit proportions.</p>')
    viz = _stacked_bar_chart(models, cat_fracs, cats, CAT_COLORS,
                             title="Proportion of steps by category")
    table = _html_table(headers, tbl_rows)

    sections.append((
        "<h2>3. High-Level Category Frequencies</h2>",
        phenomenon + viz + table,
        "<p>Each base intent maps to one of 9 high-level categories: read, search, "
        "reproduce, edit, verify, git, housekeeping, failed, other.</p>"
        "<p><strong>_n</strong>: total steps in that category.</p>"
        "<p><strong>_%</strong>: percentage of all steps.</p>"
        "<p><strong>_per_traj</strong>: average number of steps in that category per "
        "trajectory (total category steps / number of trajectories).</p>"
        "<p>The mapping from base intent to category is defined in classify_intent.py "
        "(INTENT_TO_HIGH_LEVEL). For example, read-file-full, read-file-range, and "
        "read-via-bash all map to 'read'.</p>"
    ))

    # ── 4. Phase Groupings ──────────────────────────────────
    phase_names = list(PHASE_MAP.keys())

    phase_fracs = {}
    for model in models:
        total_steps = sum(r.steps for r in results[model])
        phase_fracs[model] = {}
        for phase, phase_cats in PHASE_MAP.items():
            phase_steps = sum(
                r.high_intent_counts.get(c, 0)
                for r in results[model] for c in phase_cats
            )
            phase_fracs[model][phase] = phase_steps / total_steps if total_steps else 0

    headers = ["phase"] + [f"{m}_%" for m in models]
    tbl_rows = []
    for phase in phase_names:
        row = [phase]
        for model in models:
            row.append(f"{phase_fracs[model].get(phase, 0) * 100:.1f}%")
        tbl_rows.append(row)

    phenomenon = ('<p style="font-size:13px;color:#444;margin:0 0 8px 0">'
                  'Claude spends 28% verifying; GPT-5 spends 3.6%. '
                  'Gemini reads the most; Claude cleans up the most.</p>')
    viz = _stacked_bar_chart(models, phase_fracs, phase_names, PHASE_COLORS,
                             title="Proportion of steps by phase")
    table = _html_table(headers, tbl_rows)

    sections.append((
        "<h2>4. Phase Groupings</h2>",
        phenomenon + viz + table,
        "<p>The 9 high-level categories are further grouped into 5 phases that "
        "represent the broad arc of a trajectory:</p>"
        "<p><strong>understand</strong> = read + search. The agent is reading code "
        "and searching for information.</p>"
        "<p><strong>reproduce</strong> = reproduce. The agent is writing or running "
        "reproduction scripts to confirm the bug.</p>"
        "<p><strong>edit</strong> = edit. The agent is making source code changes.</p>"
        "<p><strong>verify</strong> = verify. The agent is running tests, compiling, "
        "or checking its work.</p>"
        "<p><strong>cleanup</strong> = git + housekeeping. The agent is reviewing "
        "changes (git diff/log) or cleaning up (rm, mv, writing docs).</p>"
        "<p>These phases are used in the stacked area charts (Typical Trajectory Shape) "
        "to show how the mix of actions evolves from start to end.</p>"
    ))

    return sections



def render_resolution_sections(results, models) -> list[tuple[str, str, str]]:
    """Render sections 0, 0b, and 1 with inline Tufte-style visualizations.

    Returns list of (title_html, content_html, notes_html) tuples.
    """
    sections = []

    # ── 0. Resolution and Submission ──────────────────────
    # Compute funnel data per model: n -> submitted -> resolved
    funnel = {}
    for model in models:
        data = results[model]
        n = len(data)
        clean_submit = sum(1 for r in data if r.exit_status == "submitted")
        submitted_w_error = sum(
            1 for r in data
            if (r.exit_status or "").startswith("submitted ("))
        not_submitted = n - clean_submit - submitted_w_error
        resolved = sum(1 for r in data if r.resolved)
        submitted_total = clean_submit + submitted_w_error
        submitted_not_resolved = submitted_total - resolved
        funnel[model] = {
            "n": n, "resolved": resolved,
            "submitted_not_resolved": submitted_not_resolved,
            "not_submitted": not_submitted,
            "clean_submit": clean_submit,
            "submitted_w_error": submitted_w_error,
        }

    # Phenomenon line
    top_model = max(models,
                    key=lambda m: funnel[m]["resolved"] / funnel[m]["n"])
    top_rate = funnel[top_model]["resolved"] / funnel[top_model]["n"] * 100
    phenomenon_0 = (
        f'<p style="font-style:italic;color:#555;margin:0 0 14px 0">'
        f'Of {funnel[top_model]["n"]} tasks, the highest resolve rate is '
        f'{top_rate:.0f}% ({_model_label(top_model)}). '
        f'Most submitted patches do not resolve the issue.</p>'
    )

    # Stacked horizontal bars: resolved | submitted-but-wrong | not-submitted
    bar_html = '<div style="margin:0 0 18px 0">\n'
    for model in models:
        f = funnel[model]
        n = f["n"]
        color = _model_color(model)
        r_pct = f["resolved"] / n * 100
        s_pct = f["submitted_not_resolved"] / n * 100
        ns_pct = f["not_submitted"] / n * 100
        label = _model_label(model)
        bar_html += (
            f'<div style="display:flex;align-items:center;margin:4px 0">'
            f'<span style="width:110px;font-size:12px;color:#555;'
            f'text-align:right;padding-right:10px">{label}</span>'
            f'<div style="display:flex;flex:1;height:18px;max-width:500px;'
            f'border-radius:2px;overflow:hidden">'
            f'<div title="Resolved: {f["resolved"]}" '
            f'style="width:{r_pct:.1f}%;background:{color}"></div>'
            f'<div title="Submitted, not resolved: '
            f'{f["submitted_not_resolved"]}" '
            f'style="width:{s_pct:.1f}%;background:{color};'
            f'opacity:0.35"></div>'
            f'<div title="Not submitted: {f["not_submitted"]}" '
            f'style="width:{ns_pct:.1f}%;background:#ddd"></div>'
            f'</div>'
            f'<span style="font-size:11px;color:#888;padding-left:8px">'
            f'{f["resolved"]}/{n}</span>'
            f'</div>\n'
        )
    bar_html += (
        '<div style="display:flex;align-items:center;margin:6px 0 0 110px;'
        'font-size:11px;color:#888;gap:14px">'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:#999;border-radius:1px;vertical-align:middle"></span>'
        ' resolved</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:#999;opacity:0.35;border-radius:1px;'
        'vertical-align:middle"></span>'
        ' submitted, not resolved</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:#ddd;border-radius:1px;vertical-align:middle"></span>'
        ' not submitted</span>'
        '</div></div>\n'
    )

    # Table
    headers = ["model", "n", "submitted (clean)", "submitted (w/ error)",
               "not submitted", "resolved", "resolve rate"]
    tbl_rows = []
    for model in models:
        f = funnel[model]
        n = f["n"]
        tbl_rows.append([model, n, f["clean_submit"],
                         f["submitted_w_error"], f["not_submitted"],
                         f["resolved"], _pct(f["resolved"], n)])

    sections.append((
        "<h2>0. Resolution and Submission</h2>",
        phenomenon_0 + bar_html + _html_table(headers, tbl_rows),
        "<p>Each SWE-Bench Pro trajectory ends with an exit status. "
        "The agent either submits a patch or doesn't.</p>"
        "<p><strong>submitted (clean)</strong>: exit_status is exactly "
        "<code>submitted</code>.</p>"
        "<p><strong>submitted (w/ error)</strong>: the agent produced a "
        "submission, but the harness also recorded an error condition "
        "(timeout, context overflow, cost limit, format error).</p>"
        "<p><strong>not submitted</strong>: the agent never ran the submit "
        "command. It hit an error before submitting.</p>"
        "<p><strong>resolved</strong>: the submitted patch actually fixes "
        "the failing tests (from benchmark evaluation).</p>"
        "<p>Method: we read <code>info.exit_status</code> and "
        "<code>info.submission</code> from each .traj file.</p>"
    ))

    # ── 0b. Exit Status Breakdown ──────────────────────────
    exit_counts = {}
    for model in models:
        for r in results[model]:
            exit_counts.setdefault(r.exit_status, {}).setdefault(model, 0)
            exit_counts[r.exit_status][model] += 1
    sorted_exits = sorted(
        exit_counts.keys(),
        key=lambda e: -sum(exit_counts[e].get(m, 0) for m in models))
    headers = ["exit_status"] + models
    tbl_rows = [[es] + [exit_counts[es].get(m, 0) for m in models]
                for es in sorted_exits]

    sections.append((
        "<h2>0b. Exit Status Breakdown</h2>",
        _html_table(headers, tbl_rows),
        "<p>Every distinct exit_status string from the .traj files, with "
        "counts per model.</p>"
        "<p><strong>submitted</strong>: clean exit after submitting.</p>"
        "<p><strong>submitted (exit_*)</strong>: submitted, but also hit "
        "an error condition (timeout, context, cost, format, etc.).</p>"
        "<p><strong>exit_*</strong> (without submitted prefix): the agent "
        "hit that condition and never submitted.</p>"
        "<p>These statuses are set by the SWE-Agent harness, not by the "
        "model itself.</p>"
    ))

    # ── 1. Trajectory Metadata ─────────────────────────────
    meta = {}
    for model in models:
        data = results[model]
        steps = [r.steps for r in data]
        resolved = sum(1 for r in data if r.resolved)
        meta[model] = {
            "n": len(data), "steps": steps, "total": sum(steps),
            "avg": round(statistics.mean(steps), 1),
            "median": _median_safe(steps),
            "p25": _p25(steps), "p75": _p75(steps),
            "mn": min(steps), "mx": max(steps),
            "resolved": resolved,
        }

    # Phenomenon line
    medians = {m: meta[m]["median"] for m in models}
    max_med_model = max(models, key=lambda m: medians[m])
    min_med_model = min(models, key=lambda m: medians[m])
    ratio = (medians[max_med_model] / medians[min_med_model]
             if medians[min_med_model] else 0)
    phenomenon_1 = (
        f'<p style="font-style:italic;color:#555;margin:0 0 14px 0">'
        f'Median trajectory length varies from '
        f'{medians[min_med_model]:.0f} to '
        f'{medians[max_med_model]:.0f} steps '
        f'({_model_label(min_med_model)} vs. '
        f'{_model_label(max_med_model)}). '
        f'The longest trajectories take ~{ratio:.1f}x more steps '
        f'per task.</p>'
    )

    # Range chart: p25 --- median --- p75 per model
    global_min = min(meta[m]["p25"] for m in models)
    global_max = max(meta[m]["p75"] for m in models)
    pad = (global_max - global_min) * 0.1
    scale_min = max(0, global_min - pad)
    scale_max = global_max + pad

    def _scale(val):
        return (val - scale_min) / (scale_max - scale_min) * 100

    range_html = '<div style="margin:0 0 18px 0">\n'
    for model in models:
        m = meta[model]
        color = _model_color(model)
        label = _model_label(model)
        left_pct = _scale(m["p25"])
        right_pct = _scale(m["p75"])
        med_pct = _scale(m["median"])
        width_pct = right_pct - left_pct

        range_html += (
            f'<div style="display:flex;align-items:center;margin:5px 0">'
            f'<span style="width:110px;font-size:12px;color:#555;'
            f'text-align:right;padding-right:10px">{label}</span>'
            f'<div style="position:relative;flex:1;height:20px;'
            f'max-width:500px">'
            # p25-p75 range bar
            f'<div style="position:absolute;left:{left_pct:.1f}%;'
            f'width:{width_pct:.1f}%;top:7px;height:6px;'
            f'background:{color};opacity:0.3;border-radius:2px"></div>'
            # median dot
            f'<div style="position:absolute;left:{med_pct:.1f}%;'
            f'top:4px;width:12px;height:12px;background:{color};'
            f'border-radius:50%;margin-left:-6px"></div>'
            # median value label
            f'<span style="position:absolute;left:{med_pct:.1f}%;'
            f'top:-11px;font-size:10px;color:{color};'
            f'transform:translateX(-50%)">{m["median"]:.0f}</span>'
            f'</div>'
            # p25-p75 text annotation
            f'<span style="font-size:10px;color:#aaa;'
            f'padding-left:8px">'
            f'{m["p25"]:.0f}&ndash;{m["p75"]:.0f}</span>'
            f'</div>\n'
        )
    range_html += (
        '<div style="display:flex;align-items:center;'
        'margin:6px 0 0 110px;font-size:11px;color:#888;gap:14px">'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'border-radius:50%;background:#999;vertical-align:middle"></span>'
        ' median</span>'
        '<span><span style="display:inline-block;width:16px;height:5px;'
        'background:#999;opacity:0.3;border-radius:1px;'
        'vertical-align:middle"></span>'
        ' p25&ndash;p75</span>'
        '</div></div>\n'
    )

    # Table
    headers = ["model", "n", "total_steps", "avg", "median", "p25", "p75",
               "min", "max", "resolved", "resolve_rate"]
    tbl_rows = []
    for model in models:
        m = meta[model]
        tbl_rows.append([model, m["n"], m["total"], m["avg"], m["median"],
                         m["p25"], m["p75"], m["mn"], m["mx"],
                         m["resolved"], _pct(m["resolved"], m["n"])])

    sections.append((
        "<h2>1. Trajectory Metadata</h2>",
        phenomenon_1 + range_html + _html_table(headers, tbl_rows),
        "<p>Summary statistics for trajectory length (number of "
        "action-observation steps per task).</p>"
        "<p><strong>n</strong>: trajectories (one per task instance).</p>"
        "<p><strong>total_steps</strong>: sum of all steps across all "
        "trajectories for this model.</p>"
        "<p><strong>avg / median / p25 / p75 / min / max</strong>: "
        "distribution of steps per trajectory.</p>"
        "<p><strong>resolved</strong>: trajectories where the submitted "
        "patch fixes the failing tests (from "
        "<code>agent_runs_data.csv</code>).</p>"
        "<p><strong>resolve_rate</strong>: resolved / n.</p>"
    ))

    return sections


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


def render_html(results, failure_data=None) -> str:
    """Results is dict[str, list[FileResult]] from process_all."""
    models = sorted(results.keys())
    sections = []

    # Sections 0, 0b, 1: resolution, exit status, metadata (with viz)
    sections.extend(render_resolution_sections(results, models))

    # 1b: Intent classification taxonomy (definitions, rules, per-model counts)
    sections.extend(render_taxonomy_section(results, models))

    sections.extend(render_intent_sections(results, models))

    # 5, 7, 8. Verify sections (with inline viz)
    sections.extend(render_verify_sections(results, models))

    # 8b. Failure mode breakdown, with GPT-focused examples
    sections.extend(render_failure_sections(failure_data, models))

    # 9 & 10. Work-done vs Resolved + Structural Markers (with inline viz)
    sections.extend(render_structure_sections(results, models))

    # 11. Phase profile heatmap
    sections.extend(render_phase_profile_section(results, models))

    # 12. Per-repo breakdown
    sections.extend(render_repo_section(results, models))

    body = "\n".join(
        f"{title}\n{table}" + (f'\n<div class="notes">{notes}</div>' if notes else "")
        for title, table, notes in sections
    )

    html_out = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Reference Tables — SWE-Bench Pro Trajectory Analysis</title>
<style>
  body {{
    font-family: 'Palatino Linotype', Palatino, Georgia, serif;
    background: #fffff8;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
    line-height: 1.5;
  }}
  h1 {{ font-size: 22px; font-weight: 400; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; font-weight: 400; font-style: italic; margin: 36px 0 10px 0; border-top: 1px solid #e0e0e0; padding-top: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 12.5px; }}
  th {{ text-align: left; padding: 4px 8px; border-bottom: 2px solid #ccc; font-weight: 600; color: #555; }}
  td {{ padding: 3px 8px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: rgba(0,0,0,0.02); }}
  caption {{ text-align: left; font-style: italic; color: #777; margin-bottom: 6px; }}
  .notes {{ font-size: 12px; color: #666; line-height: 1.6; margin: 8px 0 20px 0; max-width: 900px; }}
  .notes p {{ margin: 4px 0; }}
  .notes strong {{ color: #444; }}
</style>
</head>
<body>
<h1>Reference Tables</h1>
<p style="color:#777;font-size:14px;margin-bottom:30px">SWE-Bench Pro trajectory analysis. {len(models)} models, {sum(len(d) for d in results.values())} trajectories.</p>
{body}
</body>
</html>"""
    return _add_h2_ids(html_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("-o", "--output", default="docs/reference.html")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys to include (e.g. claude45,gpt5). Defaults to all.",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None

    data_root = Path(args.data_root)

    print("Processing trajectories...")
    results = process_all(data_root, models=models)
    for model, data in sorted(results.items()):
        print(f"  {model}: {len(data)} trajectories")

    failure_data = None
    failure_path = data_root / "failure_modes.json"
    if failure_path.exists():
        failure_data = json.loads(failure_path.read_text())
        print(f"Loaded failure modes from {failure_path}")
    else:
        print(f"Skipping failure modes section ({failure_path} not found)")

    print("Rendering HTML...")
    html = render_html(results, failure_data=failure_data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
