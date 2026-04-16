#!/usr/bin/env python3
"""
Build a reference HTML with all raw data tables for the analytics report.

Uses the analysis/ package for data. Supports N models automatically.

Usage:
  python scripts/build_reference_tables.py --data-root data -o docs/reference.html
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.orchestrate import process_all
from analysis.models import MODELS, INTENT_TO_HIGH_LEVEL

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
        "<p><strong>work-done</strong>: the trajectory contains a seq-first-all-pass label, "
        "meaning after the agent's last source code edit, it ran a verify step that passed. "
        "This is a signal that the implementation is complete and working.</p>"
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
        "<p>Method: 'work-done' is computed by finding the last source edit step, "
        "then checking if any subsequent verify step has a 'pass' outcome. "
        "'resolved' comes from the benchmark CSV (<code>agent_runs_data.csv</code>).</p>"
    )

    sections.append((
        "<h2>9. Work-Done vs Resolved</h2>",
        content_9,
        notes_9,
    ))

    # ── Section 10: Structural Markers ───────────────────────
    marker_keys = ["first_edit", "last_edit", "first_verify", "first_verify_pass", "submit"]
    marker_display = {
        "first_edit": "first edit",
        "last_edit": "last edit",
        "first_verify": "first verify",
        "first_verify_pass": "first pass",
        "submit": "submit",
    }
    # Marker shapes: distinct visual identity via CSS
    # Circles, diamonds, squares via border-radius and rotation
    marker_styles = {
        "first_edit":        "width:8px;height:8px;border-radius:50%",                                # circle
        "last_edit":         "width:8px;height:8px;border-radius:50%;border:2px solid {color};background:#fffff8",  # hollow circle
        "first_verify":      "width:7px;height:7px;border-radius:1px;transform:rotate(45deg)",        # diamond
        "first_verify_pass": "width:8px;height:8px;border-radius:1px",                                # square
        "submit":            "width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-bottom:9px solid {color}", # triangle
    }

    # Compute median positions per model
    marker_data = {}  # model -> {marker_key -> {med, p25, p75, n}}
    for model in models:
        marker_data[model] = {}
        for mk in marker_keys:
            vals = [r.positions[mk] for r in results[model]
                    if r.positions.get(mk) is not None]
            marker_data[model][mk] = {
                "med": _median_safe(vals),
                "p25": _p25(vals),
                "p75": _p75(vals),
                "n": len(vals),
            }

    # Data table
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
    table_html_10 = _html_table(headers, tbl_rows)

    # Timeline visualization
    track_width = 560  # px
    timeline_html = '<div style="margin:20px 0 16px 0">'

    # Marker legend
    timeline_html += '<div style="display:flex;gap:16px;margin-bottom:14px;font-size:11px;color:#555">'
    for mk in marker_keys:
        style = marker_styles[mk].replace("{color}", "#888")
        # For the hollow circle and triangle, we need special handling
        if mk == "last_edit":
            swatch = (
                f'<span style="display:inline-block;{style};box-sizing:border-box"></span>'
            )
        elif mk == "submit":
            swatch = (
                f'<span style="display:inline-block;{style}"></span>'
            )
        else:
            swatch = (
                f'<span style="display:inline-block;{style};background:#888"></span>'
            )
        timeline_html += (
            f'<span style="display:inline-flex;align-items:center;gap:4px">'
            f'{swatch} {marker_display[mk]}</span>'
        )
    timeline_html += '</div>'

    # Scale header
    timeline_html += (
        f'<div style="display:grid;grid-template-columns:100px {track_width}px;'
        f'gap:8px;margin-bottom:2px">'
        f'<div></div>'
        f'<div style="position:relative;height:14px;font-size:9px;color:#999">'
    )
    for tick in (0, 25, 50, 75, 100):
        left = tick / 100 * track_width
        timeline_html += (
            f'<span style="position:absolute;left:{left:.0f}px;'
            f'transform:translateX(-50%)">{tick}%</span>'
        )
    timeline_html += '</div></div>'

    # One row per model
    for model in models:
        color = _model_color(model)
        label = _model_label(model)

        timeline_html += (
            f'<div style="display:grid;grid-template-columns:100px {track_width}px;'
            f'gap:8px;align-items:center;margin-bottom:10px">'
            f'<div style="text-align:right;font-size:12px;color:{color};'
            f'font-style:italic">{label}</div>'
            f'<div style="position:relative;height:26px">'
        )

        # Baseline track
        timeline_html += (
            f'<div style="position:absolute;top:12px;left:0;width:100%;'
            f'height:1px;background:#ddd"></div>'
        )

        # IQR whiskers (p25-p75 range) as thin colored bars
        for mk in marker_keys:
            d = marker_data[model][mk]
            if d["med"] is None:
                continue
            p25 = d["p25"] if d["p25"] is not None else d["med"]
            p75 = d["p75"] if d["p75"] is not None else d["med"]
            left_pct = p25
            width_pct = max(p75 - p25, 0.3)
            timeline_html += (
                f'<div style="position:absolute;top:10px;height:5px;'
                f'left:{left_pct:.2f}%;width:{width_pct:.2f}%;'
                f'background:{color};opacity:0.18;border-radius:2px" '
                f'title="{marker_display[mk]} IQR: {p25:.1f}%-{p75:.1f}%"></div>'
            )

        # Median markers
        for mk in marker_keys:
            d = marker_data[model][mk]
            if d["med"] is None:
                continue
            left_pct = d["med"]
            mk_style = marker_styles[mk].replace("{color}", color)

            if mk == "last_edit":
                # Hollow circle
                marker_el = (
                    f'<div style="position:absolute;left:{left_pct:.2f}%;top:5px;'
                    f'transform:translateX(-50%);{mk_style};box-sizing:border-box" '
                    f'title="{marker_display[mk]}: {d["med"]:.1f}%"></div>'
                )
            elif mk == "submit":
                # Triangle
                marker_el = (
                    f'<div style="position:absolute;left:{left_pct:.2f}%;top:4px;'
                    f'transform:translateX(-50%);{mk_style}" '
                    f'title="{marker_display[mk]}: {d["med"]:.1f}%"></div>'
                )
            else:
                # Filled shapes
                marker_el = (
                    f'<div style="position:absolute;left:{left_pct:.2f}%;top:5px;'
                    f'transform:translateX(-50%);{mk_style};background:{color}" '
                    f'title="{marker_display[mk]}: {d["med"]:.1f}%"></div>'
                )
            timeline_html += marker_el

        timeline_html += '</div></div>'

    timeline_html += '</div>'

    content_10 = timeline_html + table_html_10

    notes_10 = (
        "<p>Key events in each trajectory, expressed as a percentage of the way through "
        "(0% = first step, 100% = last step). Aggregated across all trajectories per model.</p>"
        "<p>The timeline shows median positions as shaped markers, with faint bands for the "
        "interquartile range (p25-p75). Hover over markers for exact values.</p>"
        "<p><strong>first_edit</strong>: the position of the first source code edit "
        "(str_replace, insert, apply-patch, create-file).</p>"
        "<p><strong>last_edit</strong>: the position of the last source code edit. "
        "The gap between last_edit and submit is the 'tail' where the agent is verifying, "
        "cleaning up, or submitting but no longer changing code.</p>"
        "<p><strong>first_verify</strong>: the position of the first verify step "
        "(test run, compile, syntax check).</p>"
        "<p><strong>first_verify_pass</strong>: the position of the first verify step "
        "with a 'pass' outcome. Only counted for trajectories where at least one verify "
        "step passed.</p>"
        "<p><strong>submit</strong>: the position of the submit step.</p>"
        "<p><strong>_med / _p25 / _p75</strong>: median, 25th percentile, and 75th "
        "percentile across all trajectories.</p>"
        "<p><strong>_n</strong>: number of trajectories where this event occurred "
        "(some trajectories never edit, never verify, or never submit).</p>"
        "<p>Method: for each trajectory, we scan for the first/last occurrence of the "
        "relevant intent, compute index / (total_steps - 1) * 100, then take the "
        "median across trajectories.</p>"
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
        "<p>For each verify-type step, we determine whether the "
        "test/build passed or failed by inspecting the observation "
        "text.</p>"
        "<p><strong>pass</strong>: the observation contains a "
        "recognizable pass pattern (e.g., pytest's 'X passed', "
        "a clean exit code).</p>"
        "<p><strong>fail</strong>: the observation contains a "
        "recognizable failure (traceback, non-zero exit code, "
        "'FAILED').</p>"
        "<p><strong>unknown</strong>: the observation didn't match "
        "any pass or fail pattern (ambiguous, truncated, or "
        "unrecognized framework).</p>"
        "<p><strong>pass_rate</strong>: pass / (pass + fail), "
        "excluding unknowns.</p>"
        "<p>Method: classify_verify_outcome() in classify_intent.py "
        "pattern-matches the observation text against pytest summary "
        "lines, Go test output, Node.js errors, tracebacks, and "
        "exit codes.</p>")

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


def render_html(results) -> str:
    """Results is dict[str, list[FileResult]] from process_all."""
    models = sorted(results.keys())
    sections = []

    # Sections 0, 0b, 1: resolution, exit status, metadata (with viz)
    sections.extend(render_resolution_sections(results, models))

    sections.extend(render_intent_sections(results, models))

    # 5, 7, 8. Verify sections (with inline viz)
    sections.extend(render_verify_sections(results, models))

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

    return f"""<!doctype html>
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("-o", "--output", default="docs/reference.html")
    args = parser.parse_args()

    print("Processing trajectories...")
    results = process_all(Path(args.data_root))
    for model, data in sorted(results.items()):
        print(f"  {model}: {len(data)} trajectories")

    print("Rendering HTML...")
    html = render_html(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
