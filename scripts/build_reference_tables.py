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


# ── HTML rendering ───────────────────────────────────────

def _html_table(headers: list[str], rows: list[list], caption: str = "") -> str:
    h = "".join(f"<th>{hdr}</th>" for hdr in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>\n"
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def render_html(results) -> str:
    """Results is dict[str, list[FileResult]] from process_all."""
    models = sorted(results.keys())
    sections = []

    # 0. Resolution / Submission summary
    headers = ["model", "n", "submitted (clean)", "submitted (w/ error)", "not submitted", "has submission"]
    tbl_rows = []
    for model in models:
        data = results[model]
        n = len(data)
        clean_submit = sum(1 for r in data if r.exit_status == "submitted")
        has_submission = sum(1 for r in data if r.resolved)
        submitted_w_error = sum(1 for r in data if (r.exit_status or "").startswith("submitted (") )
        not_submitted = n - clean_submit - submitted_w_error
        tbl_rows.append([model, n, clean_submit, submitted_w_error, not_submitted, has_submission])
    sections.append(("<h2>0. Resolution and Submission</h2>",
        _html_table(headers, tbl_rows)))

    # 0b. Exit status breakdown
    headers = ["exit_status"] + models
    exit_counts = {}
    for model in models:
        for r in results[model]:
            exit_counts.setdefault(r.exit_status, {}).setdefault(model, 0)
            exit_counts[r.exit_status][model] += 1
    sorted_exits = sorted(exit_counts.keys(), key=lambda e: -sum(exit_counts[e].get(m, 0) for m in models))
    tbl_rows = [[es] + [exit_counts[es].get(m, 0) for m in models] for es in sorted_exits]
    sections.append(("<h2>0b. Exit Status Breakdown</h2>",
        _html_table(headers, tbl_rows)))

    # 1. Metadata
    headers = ["model", "n", "total_steps", "avg", "median", "p25", "p75", "min", "max", "resolved", "resolve_rate"]
    tbl_rows = []
    for model in models:
        data = results[model]
        steps = [r.steps for r in data]
        resolved = sum(1 for r in data if r.resolved)
        tbl_rows.append([model, len(data), sum(steps),
                         round(statistics.mean(steps), 1), _median_safe(steps),
                         _p25(steps), _p75(steps), min(steps), max(steps),
                         resolved, _pct(resolved, len(data))])
    sections.append(("<h2>1. Trajectory Metadata</h2>",
        _html_table(headers, tbl_rows)))

    # 2. Base intents
    totals = {}
    for model in models:
        c = Counter()
        for r in results[model]:
            c.update(r.base_intent_counts)
        totals[model] = c
    all_intents = sorted(set().union(*[set(c.keys()) for c in totals.values()]))
    model_totals = {m: sum(totals[m].values()) for m in models}
    intent_rows = []
    for intent in all_intents:
        row = [intent]
        for m in models:
            n = totals[m].get(intent, 0)
            row.extend([n, f"{n / model_totals[m] * 100:.1f}%" if model_totals[m] else "0%"])
        intent_rows.append(row)
    intent_rows.sort(key=lambda r: -sum(r[i] for i in range(1, len(r), 2) if isinstance(r[i], int)))
    headers = ["intent"]
    for m in models:
        headers.extend([f"{m}_n", f"{m}_%"])
    sections.append(("<h2>2. Base Intent Frequencies</h2>",
        _html_table(headers, intent_rows)))

    # 3. High-level categories
    cats = ["read", "search", "reproduce", "edit", "verify", "git", "housekeeping", "failed", "other"]
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
            row.extend([cat_steps, f"{cat_steps / total_steps * 100:.1f}%" if total_steps else "0%",
                        round(cat_steps / n_trajs, 1) if n_trajs else 0])
        tbl_rows.append(row)
    sections.append(("<h2>3. High-Level Category Frequencies</h2>",
        _html_table(headers, tbl_rows)))

    # 4. Phase groupings
    phase_map = {"understand": ["read", "search"], "reproduce": ["reproduce"],
                 "edit": ["edit"], "verify": ["verify"], "cleanup": ["git", "housekeeping"]}
    headers = ["phase"] + [f"{m}_%" for m in models]
    tbl_rows = []
    for phase, phase_cats in phase_map.items():
        row = [phase]
        for model in models:
            total_steps = sum(r.steps for r in results[model])
            phase_steps = sum(r.high_intent_counts.get(c, 0) for r in results[model] for c in phase_cats)
            row.append(f"{phase_steps / total_steps * 100:.1f}%" if total_steps else "0%")
        tbl_rows.append(row)
    sections.append(("<h2>4. Phase Groupings</h2>",
        _html_table(headers, tbl_rows)))

    # 5. Verify sub-intents
    verify_intents = [i for i, h in INTENT_TO_HIGH_LEVEL.items() if h == "verify"]
    headers = ["intent"] + [f"{m}_n" for m in models]
    tbl_rows = []
    for intent in sorted(verify_intents):
        row = [intent]
        for model in models:
            row.append(sum(r.base_intent_counts.get(intent, 0) for r in results[model]))
        tbl_rows.append(row)
    tbl_rows.sort(key=lambda r: -sum(r[1:]))
    sections.append(("<h2>5. Verify Sub-Intent Breakdown</h2>",
        _html_table(headers, tbl_rows)))

    # 7. Verify outcomes
    headers = ["model", "pass", "fail", "unknown", "total", "pass_rate"]
    tbl_rows = []
    for model in models:
        c = Counter()
        for r in results[model]:
            c.update(r.verify_outcome_counts)
        p, f_, u = c.get("pass", 0), c.get("fail", 0), c.get("", 0)
        det = p + f_
        tbl_rows.append([model, p, f_, u, p + f_ + u, f"{p / det * 100:.1f}%" if det else "n/a"])
    sections.append(("<h2>7. Verify Outcomes</h2>",
        _html_table(headers, tbl_rows)))

    # 8. Sequence labels
    seq_totals = {}
    all_labels = set()
    for model in models:
        c = Counter()
        for r in results[model]:
            c.update(r.seq_label_counts)
        seq_totals[model] = c
        all_labels.update(c.keys())
    headers = ["label"] + [f"{m}_n" for m in models]
    tbl_rows = []
    for label in sorted(all_labels):
        if label in ("", "seq-none"):
            continue
        tbl_rows.append([label] + [seq_totals[m].get(label, 0) for m in models])
    tbl_rows.sort(key=lambda r: -sum(r[1:]))
    sections.append(("<h2>8. Sequence Labels</h2>",
        _html_table(headers, tbl_rows)))

    # 9. Work-done vs resolved
    headers = ["model", "wd+resolved", "wd+unresolved", "no_wd+resolved", "no_wd+unresolved", "total"]
    tbl_rows = []
    for model in models:
        b = {"wr": 0, "wu": 0, "nr": 0, "nu": 0}
        for r in results[model]:
            if r.work_done and r.resolved: b["wr"] += 1
            elif r.work_done: b["wu"] += 1
            elif r.resolved: b["nr"] += 1
            else: b["nu"] += 1
        tbl_rows.append([model, b["wr"], b["wu"], b["nr"], b["nu"], len(results[model])])
    sections.append(("<h2>9. Work-Done vs Resolved</h2>",
        _html_table(headers, tbl_rows)))

    # 10. Structural markers
    marker_keys = ["first_edit", "last_edit", "first_verify", "first_verify_pass", "submit"]
    headers = ["marker"]
    for m in models:
        headers.extend([f"{m}_med", f"{m}_p25", f"{m}_p75", f"{m}_n"])
    tbl_rows = []
    for mk in marker_keys:
        row = [mk]
        for model in models:
            vals = [r.positions[mk] for r in results[model] if r.positions.get(mk) is not None]
            row.extend([_median_safe(vals), _p25(vals), _p75(vals), len(vals)])
        tbl_rows.append(row)
    sections.append(("<h2>10. Structural Markers (% of trajectory)</h2>",
        _html_table(headers, tbl_rows)))

    # 12. Per-repo breakdown
    repos = set()
    for data in results.values():
        for r in data:
            repos.add(r.repo)
    headers = ["repo"]
    for m in models:
        headers.extend([f"{m}_n", f"{m}_avg", f"{m}_res%", f"{m}_ver%"])
    tbl_rows = []
    for repo in sorted(repos):
        row = [repo]
        for model in models:
            sub = [r for r in results[model] if r.repo == repo]
            if not sub:
                row.extend(["", "", "", ""])
                continue
            steps = [r.steps for r in sub]
            resolved = sum(1 for r in sub if r.resolved)
            total_steps = sum(steps)
            verify = sum(r.high_intent_counts.get("verify", 0) for r in sub)
            row.extend([len(sub), round(statistics.mean(steps), 1),
                        round(resolved / len(sub) * 100, 1),
                        round(verify / total_steps * 100, 1) if total_steps else 0])
        tbl_rows.append(row)
    tbl_rows.sort(key=lambda r: -sum(c for c in r[1:] if isinstance(c, (int, float))))
    sections.append(("<h2>12. Per-Repo Breakdown</h2>",
        _html_table(headers, tbl_rows[:20])))

    body = "\n".join(f"{title}\n{table}" for title, table in sections)

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
