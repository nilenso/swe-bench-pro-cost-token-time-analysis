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
        _html_table(headers, tbl_rows),
        "<p>Each SWE-Bench Pro trajectory ends with an exit status. The agent either submits a patch or doesn't.</p>"
        "<p><strong>submitted (clean)</strong>: the agent ran its submit command and exited normally. The exit_status field is exactly <code>submitted</code>.</p>"
        "<p><strong>submitted (w/ error)</strong>: the agent produced a submission, but the harness also recorded an error condition (timeout, context window overflow, cost limit, format error). The exit_status looks like <code>submitted (exit_command_timeout)</code>. The submission exists and will be evaluated, but the agent didn't finish cleanly.</p>"
        "<p><strong>not submitted</strong>: the agent never ran the submit command. It hit an error (crash, timeout, cost limit) before submitting. No patch was produced.</p>"
        "<p><strong>has submission</strong>: total trajectories where a submission field is present in the .traj file, regardless of how the agent exited. This is the broadest count of 'produced some output'. It includes both clean and error submissions.</p>"
        "<p>Method: we read the <code>info.exit_status</code> and <code>info.submission</code> fields from each .traj file. A trajectory is 'resolved' if exit_status starts with 'submitted' or if the submission field is non-empty.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>Every distinct exit_status string from the .traj files, with counts per model.</p>"
        "<p><strong>submitted</strong>: clean exit after submitting a patch.</p>"
        "<p><strong>submitted (exit_command_timeout)</strong>: submitted, but a command timed out during the run.</p>"
        "<p><strong>submitted (exit_context)</strong>: submitted, but hit the context window limit.</p>"
        "<p><strong>submitted (exit_cost)</strong>: submitted, but hit the cost/token budget limit.</p>"
        "<p><strong>submitted (exit_format)</strong>: submitted, but the agent produced a malformed action at some point.</p>"
        "<p><strong>submitted (exit_error)</strong>: submitted, but an internal error occurred.</p>"
        "<p><strong>submitted (exit_total_execution_time)</strong>: submitted, but hit the total wall-clock time limit.</p>"
        "<p><strong>exit_error / exit_command_timeout / exit_format / etc.</strong>: the agent hit that condition and never submitted. No patch was produced.</p>"
        "<p>These statuses are set by the SWE-Agent harness, not by the model itself.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>Summary statistics for trajectory length (number of action-observation steps per task).</p>"
        "<p><strong>n</strong>: number of trajectories (one per SWE-Bench Pro task instance).</p>"
        "<p><strong>total_steps</strong>: sum of all steps across all trajectories for this model.</p>"
        "<p><strong>avg / median / p25 / p75 / min / max</strong>: distribution of steps per trajectory. A 'step' is one action taken by the agent (a command, a file edit, a search, etc.) followed by the environment's observation.</p>"
        "<p><strong>resolved</strong>: number of trajectories that produced a submission (see Table 0).</p>"
        "<p><strong>resolve_rate</strong>: resolved / n, as a percentage.</p>"))

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
        _html_table(headers, intent_rows),
        "<p>Every trajectory step is classified into one of ~50 base intents using deterministic rules (regex matching on the action string, file names, and observation text). No LLM is used for classification.</p>"
        "<p><strong>_n</strong>: total count of that intent across all trajectories for the model.</p>"
        "<p><strong>_%</strong>: that count as a percentage of all steps for the model. Percentages sum to 100% within each model.</p>"
        "<p>Sorted by total count across all models (most frequent first).</p>"
        "<p>Method: each step's action string is pattern-matched against a priority-ordered ruleset in classify_intent.py. For example, an action starting with <code>str_replace_editor view</code> is classified as a read intent, while <code>grep</code> or <code>rg</code> becomes search-keyword.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>Each base intent maps to one of 9 high-level categories: read, search, reproduce, edit, verify, git, housekeeping, failed, other.</p>"
        "<p><strong>_n</strong>: total steps in that category.</p>"
        "<p><strong>_%</strong>: percentage of all steps.</p>"
        "<p><strong>_per_traj</strong>: average number of steps in that category per trajectory (total category steps / number of trajectories).</p>"
        "<p>The mapping from base intent to category is defined in classify_intent.py (INTENT_TO_HIGH_LEVEL). For example, read-file-full, read-file-range, and read-via-bash all map to 'read'.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>The 9 high-level categories are further grouped into 5 phases that represent the broad arc of a trajectory:</p>"
        "<p><strong>understand</strong> = read + search. The agent is reading code and searching for information.</p>"
        "<p><strong>reproduce</strong> = reproduce. The agent is writing or running reproduction scripts to confirm the bug.</p>"
        "<p><strong>edit</strong> = edit. The agent is making source code changes.</p>"
        "<p><strong>verify</strong> = verify. The agent is running tests, compiling, or checking its work.</p>"
        "<p><strong>cleanup</strong> = git + housekeeping. The agent is reviewing changes (git diff/log) or cleaning up (rm, mv, writing docs).</p>"
        "<p>These phases are used in the stacked area charts (Typical Trajectory Shape) to show how the mix of actions evolves from start to end.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>The 'verify' category contains ~10 sub-intents. This table shows where each model's verification volume comes from.</p>"
        "<p><strong>run-test-suite</strong>: broad test runs (pytest, go test, npm test, mocha) without targeting specific tests.</p>"
        "<p><strong>run-test-specific</strong>: targeted test runs using pytest -k or :: to run specific test functions.</p>"
        "<p><strong>run-verify-script</strong>: running a script named verify*, check*, validate*, or edge_case*.</p>"
        "<p><strong>create-test-script</strong>: creating a new test file (test_*, *test.py, etc.).</p>"
        "<p><strong>run-inline-verify</strong>: an inline python -c / node -e snippet that imports project code or runs assertions. Classified as verify (not reproduce) because the snippet structure indicates it's checking correctness, not reproducing a bug.</p>"
        "<p><strong>compile-build</strong>: go build, go vet, make, npx tsc. Compilation as a verification step.</p>"
        "<p><strong>edit-test-or-repro</strong>: editing an existing test or repro file (str_replace on test_* or repro* files).</p>"
        "<p><strong>run-custom-script</strong>: running a named script that doesn't match repro/test/verify naming patterns.</p>"
        "<p><strong>create-verify-script</strong>: creating a new file named verify*, check*, validate*.</p>"
        "<p><strong>syntax-check</strong>: py_compile, compileall, node -c. Quick syntax validation.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>For each verify-type step (run-test-suite, run-test-specific, run-verify-script, compile-build, etc.), we attempt to determine whether the test/build passed or failed by inspecting the observation text.</p>"
        "<p><strong>pass</strong>: the observation contains a recognizable pass pattern (e.g., pytest's 'X passed', a clean exit code, 'Tests passed').</p>"
        "<p><strong>fail</strong>: the observation contains a recognizable failure (traceback, non-zero exit code, 'FAILED', assertion error).</p>"
        "<p><strong>unknown</strong>: the observation didn't match any pass or fail pattern. This happens when the output is ambiguous, truncated, or uses a test framework we don't recognize.</p>"
        "<p><strong>pass_rate</strong>: pass / (pass + fail), excluding unknowns. This tells you: of the verify steps where we could determine the outcome, what fraction passed?</p>"
        "<p>Method: classify_verify_outcome() in classify_intent.py pattern-matches the observation text against pytest summary lines, Go test output, Node.js errors, tracebacks, and exit codes.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>Sequence labels are a second-pass classification that looks at the context around each step (what came before it, whether an edit happened since the last verify, etc.).</p>"
        "<p><strong>seq-verify-after-edit</strong>: a verify step that occurs after a source edit. This is the core edit-then-test loop.</p>"
        "<p><strong>seq-verify-rerun-no-edit</strong>: a verify step where no edit happened since the last verify, but it's a different command. The agent is trying a different test or check.</p>"
        "<p><strong>seq-verify-rerun-same-command</strong>: the agent re-ran the exact same verify command without editing anything in between. Often indicates retrying after a transient failure or re-checking.</p>"
        "<p><strong>seq-edit-after-failed-verify</strong>: a source edit immediately following a failed verify step. The agent is fixing a problem that a test revealed.</p>"
        "<p><strong>seq-diagnose-read-after-failed-verify</strong>: the agent reads a file right after a verify failure. Investigating what went wrong.</p>"
        "<p><strong>seq-diagnose-search-after-failed-verify</strong>: the agent searches (grep) right after a verify failure. Looking for the cause.</p>"
        "<p><strong>seq-repro-after-edit</strong>: running a reproduce script after making edits. Checking if the original bug is fixed.</p>"
        "<p><strong>seq-repro-rerun-same-command</strong>: re-running the same repro command without edits.</p>"
        "<p><strong>seq-submit-after-verify</strong>: the submit step occurs after at least one verify step. The agent tested before submitting.</p>"
        "<p><strong>seq-reread-edited-file</strong>: reading a file that was previously edited. Reviewing own changes.</p>"
        "<p><strong>seq-first-all-pass</strong>: the first verify-pass that occurs after the last source edit. This marks the moment the agent's implementation first passes all tests.</p>"
        "<p>Method: classify_sequence_layer() in classify_intent.py walks through the trajectory maintaining state (has a verify been seen? was there an edit since?). Labels are assigned based on the transition patterns.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>A confusion matrix crossing two signals: whether the agent reached 'work-done' and whether the trajectory produced a submission.</p>"
        "<p><strong>work-done</strong>: the trajectory contains a seq-first-all-pass label, meaning: after the agent's last source code edit, it ran a verify step that passed. This is a signal that the implementation is complete and working.</p>"
        "<p><strong>resolved</strong>: the trajectory produced a submission (the agent ran the submit command, possibly with an error exit).</p>"
        "<p><strong>wd+resolved</strong>: the agent's tests passed after its last edit, and it submitted. The best case.</p>"
        "<p><strong>wd+unresolved</strong>: tests passed but the agent didn't submit. It may have continued editing after a passing test, or hit a timeout before submitting.</p>"
        "<p><strong>no_wd+resolved</strong>: the agent submitted without ever reaching a clean test pass after its final edit. It submitted without confirmation that its code works.</p>"
        "<p><strong>no_wd+unresolved</strong>: the agent neither achieved passing tests nor submitted. Complete failure to produce output.</p>"
        "<p>Method: 'work-done' is computed by finding the last source edit step, then checking if any subsequent verify step has a 'pass' outcome. 'resolved' comes from the .traj metadata.</p>"))

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
        _html_table(headers, tbl_rows),
        "<p>Key events in each trajectory, expressed as a percentage of the way through (0% = first step, 100% = last step). Aggregated across all trajectories per model.</p>"
        "<p><strong>first_edit</strong>: the position of the first source code edit (str_replace, insert, apply-patch, create-file).</p>"
        "<p><strong>last_edit</strong>: the position of the last source code edit. The gap between last_edit and 100% is the 'tail' where the agent is verifying, cleaning up, or submitting but no longer changing code.</p>"
        "<p><strong>first_verify</strong>: the position of the first verify step (test run, compile, syntax check).</p>"
        "<p><strong>first_verify_pass</strong>: the position of the first verify step with a 'pass' outcome. Only counted for trajectories where at least one verify step passed.</p>"
        "<p><strong>submit</strong>: the position of the submit step.</p>"
        "<p><strong>_med / _p25 / _p75</strong>: median, 25th percentile, and 75th percentile across all trajectories.</p>"
        "<p><strong>_n</strong>: number of trajectories where this event occurred (some trajectories never edit, never verify, or never submit).</p>"
        "<p>Method: for each trajectory, we scan for the first/last occurrence of the relevant intent, compute index / (total_steps - 1) * 100, then take the median across trajectories.</p>"))

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
        _html_table(headers, tbl_rows[:20]),
        "<p>Metrics broken down by source repository. SWE-Bench Pro tasks come from 11 open-source repos. This table shows whether the patterns hold across repos or are driven by specific ones.</p>"
        "<p><strong>_n</strong>: number of task instances from this repo.</p>"
        "<p><strong>_avg</strong>: average steps per trajectory.</p>"
        "<p><strong>_res%</strong>: resolve rate (percentage of trajectories that produced a submission).</p>"
        "<p><strong>_ver%</strong>: percentage of steps spent on verify actions.</p>"
        "<p>Sorted by total number of instances across all models (most common repos first). Top 20 shown.</p>"))

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
