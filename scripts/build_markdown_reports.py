#!/usr/bin/env python3
"""
Build agent-friendly Markdown + CSV versions of the four existing HTML reports.

Generates:
  docs/reports-md/analytics.md        (benchmark analytics, 2 models: claude45 + gpt5)
  docs/reports-md/reference.md        (benchmark reference, 2 models: claude45 + gpt5)
  docs/reports-md/pi-analytics.md     (Pi analytics, 2 models: claude-opus-4-5, gpt-5.4)
  docs/reports-md/pi-reference.md     (Pi reference, 2 models: claude-opus-4-5, gpt-5.4)
  docs/reports-md/data/{analytics,reference,pi-analytics,pi-reference}/*.csv

The Markdown preserves the prose from the existing build scripts (stripped of
HTML), and either inlines small tables or references medium/large tables as CSV.

Usage:
  python scripts/build_markdown_reports.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# Project root on sys.path so we can import analysis / analysis_pi modules.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.aggregate import (
    base_intent_frequencies,
    bigram_matrix,
    build_analytics_payload,
    high_level_frequencies,
    metadata_summary,
    per_repo_breakdown,
    phase_frequencies,
    phase_profiles,
    sequence_labels,
    step_distribution,
    structural_markers,
    verify_outcomes,
    work_done_vs_resolved,
)
from analysis.failure_modes import (
    FAILURE_MODES,
    FAMILY_LABEL,
    MODE_DESC,
    MODE_FAMILY,
    MODE_LABEL,
)
from analysis.models import (
    HIGH_LEVEL_COLORS,
    HIGH_LEVEL_LETTER,
    INTENT_DESCRIPTIONS,
    INTENT_TO_HIGH_LEVEL,
    LETTER_TO_NAME,
    MODELS as BENCH_MODEL_META,
    PHASES,
)
from analysis.orchestrate import process_all as process_benchmark
from analysis_pi import aggregate as pi_aggregate
from analysis_pi.models import (
    HIGH_LEVEL_COLORS as PI_HIGH_LEVEL_COLORS,
    INTENT_DESCRIPTIONS as PI_INTENT_DESCRIPTIONS,
    INTENT_TO_HIGH_LEVEL as PI_INTENT_TO_HIGH_LEVEL,
    build_model_registry as build_pi_model_registry,
)
from analysis_pi.orchestrate import process_all as process_pi
from analysis_pi.resolved import compute_resolution_by_model
from analysis_pi.session_filter import SessionFilter, collect_filtered_paths
from analysis_pi.user_messages import CLASS_ORDER, analyze_user_messages


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCH_MODELS = ["claude45", "gpt5"]
PI_MODELS = ["claude-opus-4-5", "gpt-5.4"]

# Pi ↔ benchmark pairing for the "paired" phase profiles on pi-analytics.md.
BENCHMARK_PAIR_FOR_PI_MODEL = {
    "claude-opus-4-5": "claude45",
    "gpt-5.4": "gpt5",
}

# Pi display labels (per task spec).
PI_DISPLAY_NAMES = {
    "claude-opus-4-5": "Opus 4.5",
    "gpt-5.4": "GPT-5.4 (high)",
}

REPORTS_DIR = ROOT / "docs" / "reports-md"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, headers: list[str], rows: list[list]) -> Path:
    """Write a CSV to `path` (creating parents), return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    return path


def _fmt_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        # keep trailing .0 off ints-as-floats, but 1-decimal otherwise
        if abs(v - round(v)) < 1e-9:
            return f"{v:.0f}"
        return f"{v:.1f}"
    return str(v)


def md_table(headers: list[str], rows: list[list]) -> str:
    """Render a Markdown pipe-table. Cells are coerced via _fmt_cell()."""
    if not rows:
        return ""
    heads = [str(h) for h in headers]
    out = ["| " + " | ".join(heads) + " |",
           "| " + " | ".join("---" for _ in heads) + " |"]
    for row in rows:
        cells = [_fmt_cell(c) for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total > 0 else "0%"


def _median(vals) -> float | None:
    return round(statistics.median(vals), 1) if vals else None


def _p25(vals) -> float | None:
    s = sorted(vals)
    return round(s[len(s) // 4], 1) if s else None


def _p75(vals) -> float | None:
    s = sorted(vals)
    return round(s[len(s) * 3 // 4], 1) if s else None


def bench_label(model: str) -> str:
    return BENCH_MODEL_META.get(model, {}).get("label", model)


def pi_label(model: str) -> str:
    return PI_DISPLAY_NAMES.get(model, model)


# ---------------------------------------------------------------------------
# Report 1: benchmark analytics (analytics.md)
# ---------------------------------------------------------------------------

def build_benchmark_analytics(results: dict, out_dir: Path) -> None:
    """Mirror docs/analytics-sonnet-gpt5.html as analytics.md + CSVs."""
    payload = build_analytics_payload(results)
    models = [m for m in BENCH_MODELS if m in results]  # preserves order
    # Section 4 panels ordered by resolve rate desc (matches HTML convention).
    rr = payload["resolve_rate"]
    models_by_resolve = sorted(models, key=lambda m: -rr.get(m, 0))

    csv_dir = REPORTS_DIR / "data" / "analytics"

    # ── Section 1: High-Level Action Frequencies (inline) ─────────────
    cats = ["read", "search", "reproduce", "edit", "verify", "git", "housekeeping"]
    hi = payload["high_proportions"]
    sec1_headers = ["category"] + [bench_label(m) + " (% of steps)" for m in models]
    sec1_rows = []
    for cat in cats:
        row = [cat]
        for m in models:
            row.append(f"{(hi.get(m, {}).get(cat, 0) * 100):.1f}")
        sec1_rows.append(row)
    sec1_md = md_table(sec1_headers, sec1_rows)

    # ── Section 2: Intent Comparison (CSV) ─────────────────────────────
    intents = payload["top_low_intents"]
    low_prop = payload["low_proportions"]
    intent_to_cat = payload["intent_to_category"]
    display = payload["intent_display_names"]
    sec2_headers = ["intent", "category", "display_name"] + [
        f"{bench_label(m)}_per_100_steps" for m in models
    ]
    sec2_rows = []
    for intent in intents:
        row = [
            intent,
            intent_to_cat.get(intent, ""),
            display.get(intent, intent),
        ]
        for m in models:
            v = low_prop.get(m, {}).get(intent, 0) * 100
            row.append(f"{v:.2f}")
        sec2_rows.append(row)
    write_csv(csv_dir / "intent-frequencies.csv", sec2_headers, sec2_rows)

    # ── Section 3: Steps per trajectory (CSV, 5-step bins) ─────────────
    step_dist = payload["step_dist"]
    bins = sorted({int(b) for m in models for b in step_dist.get(m, {}).keys()})
    sec3_headers = ["bin_start"] + [bench_label(m) for m in models]
    sec3_rows = []
    for b in bins:
        row = [b] + [step_dist.get(m, {}).get(b, 0) for m in models]
        sec3_rows.append(row)
    write_csv(csv_dir / "step-distribution.csv", sec3_headers, sec3_rows)

    # ── Section 4: Phase profiles — one CSV per model (20 bins × categories)
    avg_phase = payload["avg_phase"]  # model -> letter -> list[20]
    median_last_edit = payload["median_last_edit"]
    phase_cats = ["read", "search", "reproduce", "edit", "verify", "git", "housekeeping"]
    # Convert letters back to names for rendering.
    for m in models_by_resolve:
        phase = avg_phase.get(m, {})
        # Convert letter->list into category columns.
        letter_to_name = LETTER_TO_NAME
        rows = []
        for b in range(20):
            row = [b]
            for cat in phase_cats:
                letter = HIGH_LEVEL_LETTER.get(cat, "?")
                vals = phase.get(letter, [])
                v = vals[b] if b < len(vals) else 0.0
                row.append(f"{v:.4f}")
            rows.append(row)
        write_csv(
            csv_dir / f"phase-profile-{m}.csv",
            ["bin"] + phase_cats,
            rows,
        )

    # Small inline summary: resolve rate + median last-edit per model
    sec4_rows = []
    for m in models_by_resolve:
        sec4_rows.append([
            bench_label(m),
            f"{rr.get(m, 0):.1f}%",
            f"{median_last_edit.get(m)}%" if median_last_edit.get(m) is not None else "—",
        ])
    sec4_summary_md = md_table(
        ["model", "resolve rate", "median last-edit position (% of trajectory)"],
        sec4_rows,
    )

    # ── Assemble the Markdown ─────────────────────────────────────────
    n_trajs = payload["num_trajs"]
    total = sum(n_trajs.get(m, 0) for m in models)
    lede = (
        f"SWE-Bench Pro — multi-model comparison. {len(models)} models, {total} trajectories: "
        + ", ".join(f"{bench_label(m)} ({n_trajs.get(m, 0)})" for m in models)
        + "."
    )

    md = [
        "# Trajectory Analytics",
        "",
        lede,
        "",
        "## 1. High-Level Action Frequencies",
        "",
        "Proportion of steps in each high-level category. Normalised so models are "
        "comparable despite different step counts.",
        "",
        sec1_md,
        "",
        "## 2. Intent Comparison",
        "",
        "Frequency per 100 steps, compared across all models. Full table (one row per "
        "intent that appears in the data, grouped by high-level category) in "
        "[data/analytics/intent-frequencies.csv](data/analytics/intent-frequencies.csv).",
        "",
        "## 3. Steps per trajectory, by model",
        "",
        "Distribution of trajectory length, binned into 5-step buckets. Trajectories "
        "that reach the 250-step cap are the right-most plateau in the original chart; "
        "the underlying data is in "
        "[data/analytics/step-distribution.csv](data/analytics/step-distribution.csv). "
        "The original chart renders cumulative share of runs that finished within N "
        "steps, with a dashed line at the 250-step cap.",
        "",
        "## 4. Typical Trajectory Shape",
        "",
        "Each trajectory is divided into 20 equal-width time bins. Per bin we compute "
        "the share of steps that fell into each high-level category, then average "
        "across trajectories. The original chart renders these as a stacked area: "
        "how the mix of actions evolves from start to end of the average trajectory. "
        "Panels are ordered by resolve rate (descending).",
        "",
        sec4_summary_md,
        "",
        "Full phase profiles (rows = bin 0..19, columns = category share of that bin, "
        "pre-normalisation) per model:",
        "",
    ]
    for m in models_by_resolve:
        md.append(f"- {bench_label(m)}: "
                  f"[data/analytics/phase-profile-{m}.csv](data/analytics/phase-profile-{m}.csv)")
    md.append("")

    (REPORTS_DIR / "analytics.md").write_text("\n".join(md))
    print("Wrote docs/reports-md/analytics.md")


# ---------------------------------------------------------------------------
# Report 2: benchmark reference (reference.md)
# ---------------------------------------------------------------------------

_TAXONOMY_RULES = {
    "read-file-full":            "str_replace_editor view <file> (fallback once test, config, range, and truncated views are ruled out)",
    "read-file-range":           "str_replace_editor view with --view_range",
    "read-file-full(truncated)": "str_replace_editor view where the observation contains 'too large to display'",
    "read-test-file":            "str_replace_editor view on a filename matching test_*, *_test.*, or conftest*",
    "read-config-file":          "str_replace_editor view on package.json, pytest.ini, setup.cfg, setup.py, go.mod, Makefile, config.json",
    "read-via-bash":             "cat, head, tail, sed -n, nl, awk",
    "view-directory":            "str_replace_editor view where path has no extension, or observation lists 'files and directories'",
    "list-directory":            "ls, tree, pwd",
    "search-keyword":            "grep, rg, ag",
    "search-files-by-name":      "find ... -name with no grep/xargs pipe",
    "search-files-by-content":   "find ... -exec grep or find ... | xargs grep",
    "inspect-file-metadata":     "wc, file, stat",
    "create-repro-script":       "str_replace_editor create on a filename containing repro, reproduce, or demo",
    "run-repro-script":          "run a named script whose basename matches repro* or reproduce* (python, node, sh, bash, go run)",
    "run-inline-snippet":        "python -c, python - <<, node -e — residual when no inline sub-pattern matches",
    "run-inline-verify":         "inline snippet with import/from + assert/print (smoke test or assertion)",
    "read-via-inline-script":    "inline snippet that reads a file (.read(), open(...,'r'), readFileSync) and prints, without writing",
    "edit-via-inline-script":    "inline snippet that writes (.write(), writeFileSync) together with reading or .replace()/re.sub()",
    "create-file-via-inline-script": "inline snippet that writes a file with no prior read",
    "check-version":             "inline snippet matching --version, -V, sys.version, or node -v",
    "edit-source":               "str_replace_editor str_replace on a filename not matching test/repro/verify/check",
    "insert-source":             "str_replace_editor insert",
    "apply-patch":               "applypatch command (GPT-specific)",
    "create-file":               "str_replace_editor create on a filename not matching repro/test/verify/doc patterns",
    "run-test-suite":            "pytest, go test, npm test, npx jest, mocha, yarn test, python -m unittest (broad; no :: or -k)",
    "run-test-specific":         "a test runner command containing :: or -k",
    "create-test-script":        "str_replace_editor create on a filename matching test_*, *test.py, *test.js, *test.go",
    "run-verify-script":         "run a named script whose basename contains test_, verify, check, validate, or edge_case",
    "create-verify-script":      "str_replace_editor create on a filename matching verify*, check*, or validate*",
    "edit-test-or-repro":        "str_replace_editor str_replace on a filename containing test_, repro, verify, or check",
    "run-custom-script":         "run a named python/node/sh/bash/go script whose basename doesn't match repro/test/verify patterns",
    "syntax-check":              "py_compile, compileall, node -c",
    "compile-build":             "go build, go vet, make, tsc, npx tsc, npm run build, yarn build",
    "git-diff":                  "git diff (with or without -C <dir>)",
    "git-status-log":            "git status, git show, git log",
    "git-stash":                 "git stash",
    "file-cleanup":              "rm, mv, cp, chmod",
    "create-documentation":      "str_replace_editor create on a filename matching *summary*, *readme*, *changes*, *implementation*",
    "start-service":             "redis-server, redis-cli, mongod, sleep",
    "install-deps":              "pip install, pip list, npm install, go get, apt",
    "check-tool-exists":         "which, type",
    "search-keyword(failed)":    "grep/find whose observation contains a shell error",
    "read-via-bash(failed)":     "cat/head/sed/tail/ls whose observation contains a shell error",
    "run-script(failed)":        "python/node whose observation contains a shell error",
    "run-test-suite(failed)":    "test runner whose observation contains a shell error",
    "bash-command(failed)":      "any other bash command whose observation contains a shell error",
    "submit":                    "action's first line starts with 'submit'",
    "empty":                     "action string is blank (rate-limit or context-window exit)",
    "echo":                      "echo, printf",
    "bash-other":                "final fallback — bash command that matched no other rule (<2% of steps by design)",
    "undo-edit":                 "str_replace_editor undo_edit",
}

_TAXONOMY_ORDER = [
    ("read", ["read-file-full", "read-file-range", "read-file-full(truncated)",
              "read-test-file", "read-config-file", "read-via-bash",
              "read-via-inline-script"]),
    ("search", ["view-directory", "list-directory", "search-keyword",
                "search-files-by-name", "search-files-by-content",
                "inspect-file-metadata", "check-version"]),
    ("reproduce", ["create-repro-script", "run-repro-script", "run-inline-snippet"]),
    ("edit", ["edit-source", "insert-source", "apply-patch", "create-file",
              "edit-via-inline-script", "create-file-via-inline-script"]),
    ("verify", ["run-test-suite", "run-test-specific", "create-test-script",
                "run-verify-script", "create-verify-script", "edit-test-or-repro",
                "run-custom-script", "syntax-check", "compile-build",
                "run-inline-verify"]),
    ("git", ["git-diff", "git-status-log", "git-stash"]),
    ("housekeeping", ["file-cleanup", "create-documentation", "start-service",
                      "install-deps", "check-tool-exists"]),
    ("failed", ["search-keyword(failed)", "read-via-bash(failed)",
                "run-script(failed)", "run-test-suite(failed)",
                "bash-command(failed)"]),
    ("other", ["submit", "empty", "echo", "bash-other", "undo-edit"]),
]


def build_benchmark_reference(results: dict, failure_data, out_dir: Path) -> None:
    models = [m for m in BENCH_MODELS if m in results]
    csv_dir = REPORTS_DIR / "data" / "reference"

    md: list[str] = [
        "# Reference Tables",
        "",
        f"SWE-Bench Pro trajectory analysis. {len(models)} models, "
        f"{sum(len(results[m]) for m in models)} trajectories: "
        + ", ".join(f"{bench_label(m)} ({len(results[m])})" for m in models)
        + ".",
        "",
    ]

    # ── §0 Resolution and Submission ──────────────────────────────────
    funnel = {}
    for m in models:
        data = results[m]
        n = len(data)
        clean_submit = sum(1 for r in data if r.exit_status == "submitted")
        submitted_w_error = sum(
            1 for r in data if (r.exit_status or "").startswith("submitted (")
        )
        not_submitted = n - clean_submit - submitted_w_error
        resolved = sum(1 for r in data if r.resolved)
        funnel[m] = {
            "n": n, "clean": clean_submit, "err": submitted_w_error,
            "ns": not_submitted, "res": resolved,
        }
    rows = []
    for m in models:
        f = funnel[m]
        rows.append([
            m, f["n"], f["clean"], f["err"], f["ns"], f["res"],
            pct(f["res"], f["n"]),
        ])

    md += [
        "## 0. Resolution and Submission",
        "",
        "Each SWE-Bench Pro trajectory ends with an exit status. The agent either "
        "submits a patch or doesn't.",
        "",
        md_table(
            ["model", "n", "submitted (clean)", "submitted (w/ error)",
             "not submitted", "resolved", "resolve rate"],
            rows,
        ),
        "",
        "- **submitted (clean)**: exit_status is exactly `submitted`.",
        "- **submitted (w/ error)**: the agent produced a submission, but the harness "
        "also recorded an error condition (timeout, context overflow, cost limit, "
        "format error).",
        "- **not submitted**: the agent never ran the submit command. It hit an error "
        "before submitting.",
        "- **resolved**: the submitted patch actually fixes the failing tests (from "
        "benchmark evaluation).",
        "",
        "Method: we read `info.exit_status` and `info.submission` from each `.traj` file.",
        "",
    ]

    # ── §0b Exit Status Breakdown (inline) ─────────────────────────────
    exit_counts: dict[str, dict[str, int]] = {}
    for m in models:
        for r in results[m]:
            exit_counts.setdefault(r.exit_status, {}).setdefault(m, 0)
            exit_counts[r.exit_status][m] += 1
    sorted_exits = sorted(
        exit_counts.keys(),
        key=lambda e: -sum(exit_counts[e].get(m, 0) for m in models),
    )
    rows = [[es] + [exit_counts[es].get(m, 0) for m in models] for es in sorted_exits]
    md += [
        "## 0b. Exit Status Breakdown",
        "",
        "Every distinct `exit_status` string from the `.traj` files, with counts per model.",
        "",
        md_table(["exit_status"] + models, rows),
        "",
        "- **submitted**: clean exit after submitting.",
        "- **submitted (exit_\\*)**: submitted, but also hit an error condition "
        "(timeout, context, cost, format, etc.).",
        "- **exit_\\*** (without submitted prefix): the agent hit that condition and "
        "never submitted.",
        "",
        "These statuses are set by the SWE-Agent harness, not by the model itself.",
        "",
    ]

    # ── §1 Trajectory Metadata (inline) ────────────────────────────────
    meta = metadata_summary(results)
    rows = []
    for m in models:
        mm = meta[m]
        rows.append([
            m, mm["n"], mm["total_steps"], mm["avg"], mm["median"],
            mm["p25"], mm["p75"], mm["min"], mm["max"],
            mm["resolved"], mm["resolve_rate"],
        ])
    md += [
        "## 1. Trajectory Metadata",
        "",
        "Summary statistics for trajectory length (number of action-observation steps "
        "per task).",
        "",
        md_table(
            ["model", "n", "total_steps", "avg", "median", "p25", "p75",
             "min", "max", "resolved", "resolve_rate"],
            rows,
        ),
        "",
        "- **n**: trajectories (one per task instance).",
        "- **total_steps**: sum of all steps across all trajectories for this model.",
        "- **avg / median / p25 / p75 / min / max**: distribution of steps per trajectory.",
        "- **resolved**: trajectories where the submitted patch fixes the failing tests "
        "(from `agent_runs_data.csv`).",
        "- **resolve_rate**: resolved / n.",
        "",
    ]

    # ── §1b Intent Classification Taxonomy (CSV) ───────────────────────
    intent_totals: dict[str, dict[str, int]] = {i: {} for i in INTENT_TO_HIGH_LEVEL}
    for m in models:
        for r in results[m]:
            for intent, n in r.base_intent_counts.items():
                intent_totals.setdefault(intent, {})
                intent_totals[intent][m] = intent_totals[intent].get(m, 0) + n
    tax_headers = ["intent", "high_level_category", "description", "classification_rule"] + [
        f"{m}_count" for m in models
    ]
    tax_rows = []
    for cat, intents in _TAXONOMY_ORDER:
        for intent in intents:
            row = [
                intent,
                cat,
                INTENT_DESCRIPTIONS.get(intent, ""),
                _TAXONOMY_RULES.get(intent, ""),
            ] + [intent_totals.get(intent, {}).get(m, 0) for m in models]
            tax_rows.append(row)
    write_csv(csv_dir / "intent-taxonomy.csv", tax_headers, tax_rows)

    md += [
        "## 1b. Intent Classification Taxonomy",
        "",
        "Every step is labelled by a deterministic, priority-ordered ruleset in "
        "`scripts/classify_intent.py` — no model inference. Each row in the CSV shows "
        "the intent, what it means, the literal rule that fires it, and how often it "
        "fires per model.",
        "",
        "Classification order (first rule that matches wins):",
        "",
        "1. Empty action → `empty`. `submit` prefix → `submit`.",
        "2. `str_replace_editor {view, create, str_replace, insert, undo_edit}` → "
        "classified by sub-command and filename pattern (test/config/repro/verify/doc).",
        "3. Bash is unwrapped: strip `bash -lc \"...\"`, leading `cd ... &&`, "
        "`source ... &&`, `timeout N`, and `FOO=bar` env prefixes.",
        "4. If the observation shows a shell-level error (`syntax error`, "
        "`command not found`, `unexpected token`, …), the command head routes to the "
        "matching `(failed)` label.",
        "5. Otherwise match the command head: test runners, compile/syntax, search, "
        "read, list, git, python/node scripts (named vs. inline, with inline further "
        "sub-classified by code shape), file cleanup, install, service, tool-exists, "
        "metadata, echo.",
        "6. Anything that reached the end is `bash-other` (<2% of steps by design).",
        "",
        "Full per-intent taxonomy + per-model counts: "
        "[data/reference/intent-taxonomy.csv](data/reference/intent-taxonomy.csv).",
        "",
        "The label describes *what the command is*, derived from the action string "
        "and filename alone — no positional context (before/after first edit) and no "
        "outcome signal is used. A failed grep is still a search attempt. The "
        "`(failed)` variants classify by intended action, not outcome. They require a "
        "shell-level error in the first 500 chars of the observation. "
        "`run-inline-snippet` is a residual — inline snippets (`python -c`, "
        "`python - <<`, `node -e`) are first routed to `run-inline-verify` / "
        "`read-via-inline-script` / `edit-via-inline-script` / "
        "`create-file-via-inline-script` / `check-version` by inspecting the code "
        "shape.",
        "",
    ]

    # ── §2 Base Intent Frequencies (CSV) ───────────────────────────────
    bi = base_intent_frequencies(results)
    all_intents = sorted({i for m in models for i in bi["counts"].get(m, {}).keys()})
    model_totals = {m: sum(bi["counts"].get(m, {}).values()) for m in models}
    bi_headers = ["intent"] + sum([[f"{m}_n", f"{m}_%"] for m in models], [])
    bi_rows = []
    for intent in all_intents:
        row = [intent]
        for m in models:
            n = bi["counts"].get(m, {}).get(intent, 0)
            row += [n, pct(n, model_totals[m])]
        bi_rows.append(row)
    # Sort by total count desc
    bi_rows.sort(key=lambda r: -sum(r[i] for i in range(1, len(r), 2) if isinstance(r[i], int)))
    write_csv(csv_dir / "base-intent-frequencies.csv", bi_headers, bi_rows)

    md += [
        "## 2. Base Intent Frequencies",
        "",
        "Every trajectory step is classified into one of ~50 base intents using "
        "deterministic rules (regex matching on the action string, file names, and "
        "observation text). No LLM is used for classification.",
        "",
        "- **_n**: total count of that intent across all trajectories for the model.",
        "- **_%**: that count as a percentage of all steps for the model. "
        "Percentages sum to 100% within each model.",
        "",
        "Sorted by total count across all models (most frequent first). Full table: "
        "[data/reference/base-intent-frequencies.csv](data/reference/base-intent-frequencies.csv).",
        "",
    ]

    # ── §3 High-Level Category Frequencies (inline) ────────────────────
    cats = ["read", "search", "reproduce", "edit", "verify",
            "git", "housekeeping", "failed", "other"]
    rows = []
    for cat in cats:
        row = [cat]
        for m in models:
            total_steps = sum(r.steps for r in results[m])
            n_trajs = len(results[m])
            cat_steps = sum(r.high_intent_counts.get(cat, 0) for r in results[m])
            row += [cat_steps, pct(cat_steps, total_steps),
                    round(cat_steps / n_trajs, 1) if n_trajs else 0]
        rows.append(row)
    headers = ["category"] + sum(
        [[f"{m}_n", f"{m}_%", f"{m}_per_traj"] for m in models], []
    )
    md += [
        "## 3. High-Level Category Frequencies",
        "",
        "Each base intent maps to one of 9 high-level categories: read, search, "
        "reproduce, edit, verify, git, housekeeping, failed, other.",
        "",
        md_table(headers, rows),
        "",
        "- **_n**: total steps in that category.",
        "- **_%**: percentage of all steps.",
        "- **_per_traj**: average number of steps in that category per trajectory "
        "(total category steps / number of trajectories).",
        "",
        "The mapping from base intent to category is defined in `classify_intent.py` "
        "(`INTENT_TO_HIGH_LEVEL`). For example, `read-file-full`, `read-file-range`, "
        "and `read-via-bash` all map to `read`.",
        "",
    ]

    # ── §4 Phase Groupings (inline) ────────────────────────────────────
    phase_map = {p: PHASES[p]["categories"] for p in PHASES}
    rows = []
    for phase, phase_cats in phase_map.items():
        row = [phase]
        for m in models:
            total_steps = sum(r.steps for r in results[m])
            phase_steps = sum(
                r.high_intent_counts.get(c, 0)
                for r in results[m] for c in phase_cats
            )
            row.append(f"{phase_steps / total_steps * 100:.1f}%" if total_steps else "0%")
        rows.append(row)
    md += [
        "## 4. Phase Groupings",
        "",
        "The 9 high-level categories are further grouped into 5 phases that represent "
        "the broad arc of a trajectory:",
        "",
        "- **understand** = read + search. The agent is reading code and searching for "
        "information.",
        "- **reproduce** = reproduce. The agent is writing or running reproduction "
        "scripts to confirm the bug.",
        "- **edit** = edit. The agent is making source code changes.",
        "- **verify** = verify. The agent is running tests, compiling, or checking "
        "its work.",
        "- **cleanup** = git + housekeeping. The agent is reviewing changes "
        "(git diff/log) or cleaning up (rm, mv, writing docs).",
        "",
        md_table(["phase"] + [f"{m}_%" for m in models], rows),
        "",
        "These phases are used in the stacked area charts (Typical Trajectory Shape) "
        "to show how the mix of actions evolves from start to end.",
        "",
    ]

    # ── §5 Verify Sub-Intent Breakdown (inline) ────────────────────────
    verify_intents = [i for i, h in INTENT_TO_HIGH_LEVEL.items() if h == "verify"]
    sub_totals = {m: Counter() for m in models}
    for m in models:
        for r in results[m]:
            for intent in verify_intents:
                sub_totals[m][intent] += r.base_intent_counts.get(intent, 0)
    rows = []
    for intent in sorted(
        verify_intents,
        key=lambda i: -sum(sub_totals[m].get(i, 0) for m in models),
    ):
        total_i = sum(sub_totals[m].get(intent, 0) for m in models)
        if total_i == 0:
            continue
        rows.append([intent] + [sub_totals[m].get(intent, 0) for m in models])
    md += [
        "## 5. Verify Sub-Intent Breakdown",
        "",
        "The 'verify' category contains ~10 sub-intents. This table shows where each "
        "model's verification volume comes from.",
        "",
        md_table(["intent"] + [f"{m}_n" for m in models], rows),
        "",
        "- **run-test-suite**: broad test runs (pytest, go test, npm test, mocha) "
        "without targeting specific tests.",
        "- **run-test-specific**: targeted test runs using `pytest -k` or `::` to run "
        "specific test functions.",
        "- **run-verify-script**: running a script named `verify*`, `check*`, "
        "`validate*`, or `edge_case*`.",
        "- **create-test-script**: creating a new test file (`test_*`, `*test.py`, etc.).",
        "- **run-inline-verify**: an inline `python -c` / `node -e` snippet that "
        "imports project code or runs assertions.",
        "- **compile-build**: `go build`, `go vet`, `make`, `npx tsc`. Compilation as "
        "a verification step.",
        "- **edit-test-or-repro**: editing an existing test or repro file "
        "(`str_replace` on `test_*` or `repro*` files).",
        "- **run-custom-script**: running a named script that doesn't match "
        "repro/test/verify naming patterns.",
        "- **create-verify-script**: creating a new file named `verify*`, `check*`, "
        "`validate*`.",
        "- **syntax-check**: `py_compile`, `compileall`, `node -c`. Quick syntax "
        "validation.",
        "",
    ]

    # ── §7 Verify Outcomes (inline) ────────────────────────────────────
    vo = verify_outcomes(results)
    rows = []
    for m in models:
        v = vo[m]
        rows.append([m, v["pass"], v["fail"], v["unknown"], v["total"], v["pass_rate"]])
    md += [
        "## 7. Verify Outcomes",
        "",
        "Only steps classified as one of these intents are evaluated for outcome: "
        "`run-test-suite`, `run-test-specific`, `run-verify-script`, "
        "`run-custom-script`, `run-inline-verify`, `compile-build`, `syntax-check`. "
        "All other steps get outcome `\"\"`.",
        "",
        md_table(["model", "pass", "fail", "unknown", "total", "pass_rate"], rows),
        "",
        "- **pass**: the observation's last 2000 characters match a framework-specific "
        "all-pass pattern. For pytest: the summary line (e.g. '200 passed in 12.3s') "
        "must contain 'passed' and must NOT contain 'failed' or 'error'. If even one "
        "test fails ('195 passed, 5 failed'), the outcome is 'fail', not 'pass'. For "
        "Go: all PASS/FAIL lines in output are checked; any FAIL makes it 'fail'. For "
        "Mocha: checks 'N passing' and 'N failing' counts. For Jest: checks summary "
        "line for 'failed' vs 'passed'. For `compile-build`: absence of error patterns "
        "in short output (< 200 chars) from `go build`/`make` = 'pass'. For "
        "`syntax-check`: `py_compile` with no output = 'pass'; any `Error`/"
        "`SyntaxError` = 'fail'.",
        "- **fail**: the observation matches a failure pattern.",
        "- **unknown ('')**: no pattern matched.",
        "- **pass_rate**: pass / (pass + fail), excluding unknowns.",
        "",
        "Important caveat: 'pass' means 'all tests in that run passed', not 'the "
        "agent's fix is correct'. SWE-Bench Pro tasks come with existing test suites "
        "where most tests already pass on unmodified code. An agent running pytest "
        "before making any edits will often get 'pass' because the existing tests pass.",
        "",
    ]

    # ── §8 Sequence Labels (inline) ────────────────────────────────────
    sq = sequence_labels(results)
    rows = []
    for label in sq["all_labels"]:
        if label in ("", "seq-none"):
            continue
        rows.append([label] + [sq["counts"].get(m, {}).get(label, 0) for m in models])
    rows.sort(key=lambda r: -sum(r[1:]))
    md += [
        "## 8. Sequence Labels",
        "",
        "Sequence labels classify steps by their context: what happened before, "
        "whether edits or verify steps preceded them.",
        "",
        md_table(["label"] + [f"{m}_n" for m in models], rows),
        "",
        "- **seq-verify-after-edit**: a verify step after a source edit. The core "
        "edit-then-test loop.",
        "- **seq-verify-rerun-no-edit**: a verify step where no edit happened since "
        "the last verify.",
        "- **seq-edit-after-failed-verify**: a source edit after a failed verify step. "
        "Fixing what a test revealed.",
        "- **seq-submit-after-verify**: submit after at least one verify step. The "
        "agent tested before submitting.",
        "- **seq-first-all-pass**: the first verify-pass after the last source edit. "
        "Marks implementation completion.",
        "",
        "Method: `classify_sequence_layer()` in `classify_intent.py` walks the "
        "trajectory maintaining state (has a verify been seen? was there an edit "
        "since?).",
        "",
    ]

    # ── §8b Failure Modes (CSV) ────────────────────────────────────────
    if failure_data:
        per_model = failure_data.get("per_model", {})
        mode_meta = {m["key"]: m for m in failure_data.get("modes", [])}
        active_models = [m for m in models if m in per_model]
        headers = ["mode", "family", "label", "description"] + [
            f"{m}_count" for m in active_models
        ] + [f"{m}_trajectories_with_mode" for m in active_models]
        rows = []
        sorted_modes = sorted(
            mode_meta,
            key=lambda k: -sum(
                per_model[m].get("mode_counts", {}).get(k, 0)
                for m in active_models
            ),
        )
        for mk in sorted_modes:
            meta_m = mode_meta[mk]
            counts = [per_model[m].get("mode_counts", {}).get(mk, 0) for m in active_models]
            tws = [per_model[m].get("trajectories_with_mode", {}).get(mk, 0) for m in active_models]
            rows.append([mk, meta_m["family"], meta_m["label"], meta_m["desc"]] + counts + tws)
        write_csv(csv_dir / "failure-modes.csv", headers, rows)

        # Short per-model totals inline
        summary_rows = []
        for m in active_models:
            pm = per_model[m]
            steps = max(pm.get("n_steps", 1), 1)
            summary_rows.append([
                bench_label(m),
                pm.get("n_failures", 0),
                f"{pm.get('n_failures', 0) / steps * 100:.2f}%",
                pm.get("n_steps", 0),
            ])
        md += [
            "## 8b. Failure Modes",
            "",
            "Each trajectory step is classified as either not-a-failure or one failure "
            "mode. **Counts are step counts**, not unique incidents. A single "
            "trajectory can contribute many failure steps, and one underlying shell "
            "mistake can fan out into multiple observed failures. Families: *tool* "
            "means the harness/tool call itself failed; *code* means the agent ran "
            "code that crashed; *test* means a test runner reported failures.",
            "",
            md_table(
                ["model", "failure steps", "failure rate", "total steps"],
                summary_rows,
            ),
            "",
            "Full per-mode breakdown with descriptions and per-model counts / unique "
            "trajectories: [data/reference/failure-modes.csv](data/reference/failure-modes.csv).",
            "",
            "Interpretation caveat: `bash_broken_pipe` is often a secondary symptom. "
            "In GPT-5, many of those steps are downstream of the same wrapper "
            "pathologies that also produce trailing-brace, heredoc, or quoting "
            "failures. The distinctive GPT-5 signature is not ordinary test failure. "
            "It is repeated interaction friction around shell wrapping and "
            "hallucinated `applypatch` usage.",
            "",
        ]

    # ── §9 Work-Done vs Resolved (inline) ──────────────────────────────
    wd = work_done_vs_resolved(results)
    rows = []
    for m in models:
        b = wd[m]
        rows.append([
            m, b["wd_resolved"], b["wd_unresolved"],
            b["no_wd_resolved"], b["no_wd_unresolved"], b["n"],
        ])
    md += [
        "## 9. Work-Done vs Resolved",
        "",
        "A confusion matrix crossing two signals: whether the agent reached "
        "'work-done' and whether the benchmark evaluated the patch as correct.",
        "",
        md_table(
            ["model", "wd+resolved", "wd+unresolved", "no_wd+resolved",
             "no_wd+unresolved", "total"],
            rows,
        ),
        "",
        "- **work-done**: the trajectory contains a `seq-first-all-pass` label. "
        "Specifically: we find the last step classified as a source edit "
        "(`edit-source`, `insert-source`, `apply-patch`, `edit-via-inline-script`). "
        "Then we check if any verify step after that point has a 'pass' outcome (all "
        "tests passed, per the rules in Section 7). If yes, the trajectory is "
        "'work-done'.",
        "- **resolved**: the submitted patch actually fixes the failing tests, as "
        "judged by the SWE-Bench Pro benchmark evaluation (from "
        "`agent_runs_data.csv`). This is different from 'submitted', which only means "
        "the agent produced a patch.",
        "- **wd+resolved**: the agent's tests passed after its last edit, and the "
        "benchmark confirmed the patch is correct. The best case.",
        "- **wd+unresolved**: tests passed but the patch was wrong. The agent's own "
        "verification was a false positive.",
        "- **no_wd+resolved**: the agent never reached a clean test pass after its "
        "final edit, yet the benchmark accepted the patch.",
        "- **no_wd+unresolved**: the agent neither achieved passing tests nor "
        "produced a correct patch.",
        "",
        "The `no_wd+resolved` column is notably large across all models, meaning the "
        "agent's own test-passing signal is not a reliable predictor of benchmark "
        "resolution.",
        "",
    ]

    # ── §10 Structural Markers (inline) ────────────────────────────────
    sm = structural_markers(results)
    marker_keys = ["first_edit", "last_edit", "first_verify", "first_verify_pass", "submit"]
    headers = ["marker"] + sum(
        [[f"{m}_med", f"{m}_p25", f"{m}_p75", f"{m}_n"] for m in models], []
    )
    rows = []
    for mk in marker_keys:
        row = [mk]
        for m in models:
            d = sm[mk][m]
            row += [d["median"], d["p25"], d["p75"], d["n"]]
        rows.append(row)
    md += [
        "## 10. Structural Markers (% of trajectory)",
        "",
        "Key events in each trajectory, expressed as a percentage of the way through "
        "(0% = first step, 100% = last step). Aggregated across all trajectories per "
        "model.",
        "",
        md_table(headers, rows),
        "",
        "- **first_edit**: the first step whose base intent is one of: `edit-source`, "
        "`insert-source`, `apply-patch`, or `edit-via-inline-script`. Does not "
        "include `create-file` or `edit-test-or-repro`, which are classified "
        "differently.",
        "- **last_edit**: the last step matching those same intents. The gap between "
        "`last_edit` and `submit` is the 'tail' where the agent is verifying, cleaning "
        "up, or submitting but no longer changing source code.",
        "- **first_verify**: the first step whose intent is in "
        "`SEQUENCE_VERIFY_INTENTS`: `run-test-suite`, `run-test-specific`, "
        "`run-verify-script`, `run-custom-script`, `compile-build`, `syntax-check`, "
        "`run-inline-verify`.",
        "- **first_verify_pass**: the first step where `classify_verify_outcome()` "
        "returns 'pass' (see Section 7 for what 'pass' means). This does NOT mean "
        "'the agent's fix worked'. It means 'the first time a test/build command "
        "produced output where all tests passed'. Because SWE-Bench Pro tasks have "
        "existing test suites that mostly pass on unmodified code, an agent that "
        "runs pytest before making any edits will often get a 'pass' here. For a "
        "marker that means 'the fix works', see `work_done` in Section 9, which "
        "requires a verify pass after the last source edit.",
        "- **submit**: the first step with intent 'submit'.",
        "- **_med / _p25 / _p75**: median, 25th percentile, and 75th percentile across "
        "trajectories where the event occurred.",
        "- **_n**: number of trajectories where this event occurred.",
        "",
    ]

    # ── §11 Phase Profile Heatmap (CSVs per model) ─────────────────────
    letters = ["R", "S", "P", "E", "V", "G", "H"]
    cats_named = [LETTER_TO_NAME[l] for l in letters]
    for m in models:
        letter_bins: dict[str, list[list[float]]] = {l: [] for l in letters}
        for r in results[m]:
            if not r.phase_profile:
                continue
            for l in letters:
                if l in r.phase_profile:
                    letter_bins[l].append(r.phase_profile[l])
        bins = 20
        avg = {}
        for l in letters:
            profiles = letter_bins[l]
            if not profiles:
                avg[l] = [0.0] * bins
            else:
                avg[l] = [sum(p[b] for p in profiles) / len(profiles) for b in range(bins)]
        # Normalize per bin so each column sums to 1 (same as HTML heatmap).
        bin_sums = [sum(avg[l][b] for l in letters) for b in range(bins)]
        renormed = {
            l: [avg[l][b] / bin_sums[b] if bin_sums[b] > 0 else 0 for b in range(bins)]
            for l in letters
        }
        rows = []
        for b in range(bins):
            row = [b]
            for l in letters:
                row.append(f"{renormed[l][b]:.4f}")
            rows.append(row)
        write_csv(
            csv_dir / f"phase-heatmap-{m}.csv",
            ["bin"] + cats_named,
            rows,
        )

    md += [
        "## 11. Phase Profile Heatmap",
        "",
        "Each trajectory is divided into 20 equal-width time-slices. Per slice, we "
        "count the fraction of steps belonging to each category and average across "
        "trajectories. Then the row for each time-slice is normalised so the seven "
        "displayed categories (read, search, reproduce, edit, verify, git, "
        "housekeeping) sum to 1. The original HTML renders this as a heatmap; the "
        "underlying numeric data is here:",
        "",
    ]
    for m in models:
        md.append(
            f"- {bench_label(m)}: "
            f"[data/reference/phase-heatmap-{m}.csv](data/reference/phase-heatmap-{m}.csv)"
        )
    md += [
        "",
        "Brighter (higher-value) cells in the original chart indicate the dominant "
        "action in that time-slice. Failed and other are excluded.",
        "",
    ]

    # ── §12 Per-Repo Breakdown (CSV) ───────────────────────────────────
    per_repo = per_repo_breakdown(results)
    # sort repos by total n desc
    ranked = sorted(
        per_repo.keys(),
        key=lambda rp: -sum(per_repo[rp].get(m, {}).get("n", 0) for m in models),
    )
    repo_headers = ["repo"]
    for m in models:
        repo_headers += [f"{m}_n", f"{m}_avg_steps", f"{m}_resolve_rate",
                         f"{m}_verify_pct", f"{m}_edit_pct", f"{m}_ve_ratio"]
    repo_rows = []
    for repo in ranked:
        row = [repo]
        for m in models:
            s = per_repo[repo].get(m)
            if s is None:
                row += ["", "", "", "", "", ""]
            else:
                row += [s["n"], s["avg_steps"], s["resolve_rate"],
                        s["verify_pct"], s["edit_pct"], s["ve_ratio"]]
        repo_rows.append(row)
    write_csv(csv_dir / "per-repo-breakdown.csv", repo_headers, repo_rows)

    md += [
        "## 12. Per-Repo Breakdown",
        "",
        "Metrics broken down by source repository. SWE-Bench Pro tasks come from "
        "~11 open-source repos. Columns per model:",
        "",
        "- **_n**: number of task instances from this repo.",
        "- **_avg_steps**: average steps per trajectory.",
        "- **_resolve_rate**: resolve rate (%, of trajectories where the submitted "
        "patch fixes the failing tests).",
        "- **_verify_pct**: percentage of steps spent on verify actions.",
        "- **_edit_pct**: percentage of steps spent on edit actions.",
        "- **_ve_ratio**: verify_steps / edit_steps.",
        "",
        "Sorted by total number of instances across all models (most common repos "
        "first). Full table: "
        "[data/reference/per-repo-breakdown.csv](data/reference/per-repo-breakdown.csv).",
        "",
    ]

    (REPORTS_DIR / "reference.md").write_text("\n".join(md))
    print("Wrote docs/reports-md/reference.md")


# ---------------------------------------------------------------------------
# Pi helpers — shared between pi-analytics.md and pi-reference.md
# ---------------------------------------------------------------------------

def _filter_pi_results(results: dict, allowed_paths: dict[str, set[str]]) -> dict:
    out: dict = {}
    for model, rows in results.items():
        keep = allowed_paths.get(model)
        if not keep:
            continue
        filtered = [row for row in rows if row.path in keep]
        if filtered:
            out[model] = filtered
    return out


def _build_last_edit_marker(file_results) -> dict:
    vals = [fr.positions["last_edit"] for fr in file_results
            if fr.positions.get("last_edit") is not None]
    if not vals:
        return {"median": None, "p25": None, "p75": None}
    vs = sorted(vals)
    if len(vs) >= 2:
        p25 = round(statistics.quantiles(vs, n=4)[0], 1)
        p75 = round(statistics.quantiles(vs, n=4)[2], 1)
    else:
        p25 = p75 = round(vs[0], 1)
    return {
        "median": round(statistics.median(vs), 1),
        "p25": p25,
        "p75": p75,
    }


def _pi_intervention_markers(allowed_paths, models):
    """Return {model_or_'__all__': {macro_key: {median, p25, p75, session_count, session_pct}}}.

    Mirrors _compute_intervention_markers in build_pi_analytics.py.
    """
    user_data = analyze_user_messages(allowed_paths)
    per_model = user_data.get("per_model", {})
    macros = [
        {"key": "authorization", "members": ["authorize_work"]},
        {"key": "steering", "members": [
            "solution_steer", "evidence_or_repro", "qa_or_critique", "validation_request"
        ]},
        {"key": "closeout", "members": ["workflow_closeout"]},
    ]

    def compute_rows(classes: dict, num_sessions: int):
        rows = {}
        for macro in macros:
            recs = []
            for lbl in macro["members"]:
                recs.extend(classes.get(lbl, {}).get("messages", []))
            firsts_by_path = {}
            for rec in sorted(recs, key=lambda r: (r["path"], r["message_index"], r["progress_pct"])):
                firsts_by_path.setdefault(rec["path"], rec["progress_pct"])
            vals = sorted(firsts_by_path.values())
            if vals:
                med = round(statistics.median(vals), 1)
                if len(vals) >= 2:
                    p25 = round(statistics.quantiles(vals, n=4)[0], 1)
                    p75 = round(statistics.quantiles(vals, n=4)[2], 1)
                else:
                    p25 = p75 = round(vals[0], 1)
            else:
                med = p25 = p75 = None
            rows[macro["key"]] = {
                "median": med, "p25": p25, "p75": p75,
                "session_count": len(firsts_by_path),
                "session_pct": round(len(firsts_by_path) / num_sessions * 100, 1) if num_sessions else 0.0,
            }
        return rows

    out = {}
    overall_classes = {lbl: user_data.get("overall", {}).get(lbl, {}) for lbl in user_data.get("class_order", [])}
    out["__all__"] = compute_rows(overall_classes, user_data.get("total_sessions", 0))
    for m in models:
        pdata = per_model.get(m, {})
        out[m] = compute_rows(pdata.get("classes", {}), pdata.get("num_sessions", 0))
    return out, user_data


# ---------------------------------------------------------------------------
# Report 3: Pi analytics (pi-analytics.md)
# ---------------------------------------------------------------------------

def build_pi_analytics(pi_results, pi_allowed_paths, raw_counts,
                       benchmark_results, out_dir: Path) -> None:
    csv_dir = REPORTS_DIR / "data" / "pi-analytics"
    pi_payload = pi_aggregate.build_analytics_payload(pi_results)
    models = sorted(
        [m for m in PI_MODELS if m in pi_results],
        key=lambda m: -raw_counts.get(m, 0),
    )

    # ── Section 1: High-Level Action Frequencies (inline) ─────────────
    cats = ["read", "search", "reproduce", "edit", "verify", "git", "housekeeping"]
    hi = pi_payload["high_proportions"]
    rows = []
    for cat in cats:
        row = [cat]
        for m in models:
            row.append(f"{hi.get(m, {}).get(cat, 0) * 100:.1f}")
        rows.append(row)
    sec1_md = md_table(
        ["category"] + [f"{pi_label(m)} (% of steps)" for m in models],
        rows,
    )

    # ── Section 2: Intent Comparison (CSV) ─────────────────────────────
    top_intents = pi_payload["top_low_intents"]
    low_prop = pi_payload["low_proportions"]
    intent_cat = pi_payload["intent_to_category"]
    display = pi_payload["intent_display_names"]
    headers = ["intent", "category", "display_name"] + [
        f"{pi_label(m)}_per_100_steps" for m in models
    ]
    rows = []
    for intent in top_intents:
        row = [intent, intent_cat.get(intent, ""), display.get(intent, intent)]
        for m in models:
            row.append(f"{low_prop.get(m, {}).get(intent, 0) * 100:.2f}")
        rows.append(row)
    write_csv(csv_dir / "intent-frequencies.csv", headers, rows)

    # ── Section 3: Step distribution (CSV) ─────────────────────────────
    step_dist = pi_payload["step_dist"]
    bins = sorted({int(b) for m in models for b in step_dist.get(m, {}).keys()})
    rows = []
    for b in bins:
        rows.append([b] + [step_dist.get(m, {}).get(b, 0) for m in models])
    write_csv(
        csv_dir / "step-distribution.csv",
        ["bin_start"] + [pi_label(m) for m in models],
        rows,
    )

    # ── Intervention markers (inline small table) ──────────────────────
    interventions, user_data = _pi_intervention_markers(pi_allowed_paths, models)
    last_edit_markers = {m: _build_last_edit_marker(pi_results.get(m, [])) for m in models}
    # Schema: one row per (model × marker), columns: symbol, label, median, p25, p75, session_pct
    iv_rows = []
    iv_labels = [
        ("authorization", "△", "authorization"),
        ("steering", "○", "steering"),
        ("closeout", "□", "closeout"),
    ]
    for m in models:
        for key, sym, label in iv_labels:
            d = interventions.get(m, {}).get(key, {})
            iv_rows.append([
                pi_label(m), sym, label,
                d.get("median"), d.get("p25"), d.get("p75"),
                f"{d.get('session_pct', 0):.1f}%",
            ])
        le = last_edit_markers[m]
        iv_rows.append([
            pi_label(m), "◇", "last edit",
            le.get("median"), le.get("p25"), le.get("p75"), "",
        ])
    iv_md = md_table(
        ["model", "symbol", "marker", "median %", "p25 %", "p75 %", "% sessions"],
        iv_rows,
    )

    # ── Section 4: Paired phase profiles (CSVs per model, pi + benchmark) ─
    pi_phase = pi_payload["avg_phase"]
    letters = ["R", "S", "P", "E", "V", "G", "H"]
    phase_cats = [LETTER_TO_NAME[l] for l in letters]

    def _phase_rows(phase_map):
        rows = []
        for b in range(20):
            row = [b]
            for l in letters:
                vals = phase_map.get(l, [])
                v = vals[b] if b < len(vals) else 0.0
                row.append(f"{v:.4f}")
            rows.append(row)
        return rows

    for m in models:
        write_csv(
            csv_dir / f"phase-profile-{m}-pi.csv",
            ["bin"] + phase_cats,
            _phase_rows(pi_phase.get(m, {})),
        )
        # Benchmark baseline pairing
        bench_m = BENCHMARK_PAIR_FOR_PI_MODEL.get(m)
        if bench_m and bench_m in benchmark_results:
            b_phase_all = phase_profiles({bench_m: benchmark_results[bench_m]})
            b_phase = b_phase_all.get(bench_m, {})
            write_csv(
                csv_dir / f"phase-profile-{m}-benchmark.csv",
                ["bin"] + phase_cats,
                _phase_rows(b_phase),
            )

    # Small summary table: resolve rate + num trajs per model (Pi) and its benchmark pair
    rr = pi_payload["resolve_rate"]
    n_trajs = pi_payload["num_trajs"]
    pair_rows = []
    for m in models:
        bench_m = BENCHMARK_PAIR_FOR_PI_MODEL.get(m)
        bench_n = len(benchmark_results.get(bench_m, [])) if bench_m else 0
        bench_rr = None
        if bench_m and bench_m in benchmark_results:
            bench_resolved = sum(1 for fr in benchmark_results[bench_m] if fr.resolved)
            bench_rr = f"{bench_resolved / bench_n * 100:.1f}%" if bench_n else "0%"
        pair_rows.append([
            pi_label(m),
            raw_counts.get(m, 0),
            n_trajs.get(m, 0),
            f"{rr.get(m, 0):.1f}%",
            bench_label(bench_m) if bench_m else "—",
            bench_n,
            bench_rr if bench_rr is not None else "—",
        ])
    pair_md = md_table(
        ["Pi model", "single-model sessions", "analyzed", "completion rate",
         "benchmark baseline", "benchmark trajectories", "benchmark resolve rate"],
        pair_rows,
    )

    # Assemble markdown
    total = sum(n_trajs.get(m, 0) for m in models)
    lede = (
        f"Pi transcript sessions — strict single-model issue sessions. "
        f"{len(models)} models, {total} analyzed trajectories: "
        + ", ".join(f"{pi_label(m)} ({n_trajs.get(m, 0)})" for m in models)
        + "."
    )

    md = [
        "# Pi Trajectory Analytics",
        "",
        lede,
        "",
        "## 1. High-Level Action Frequencies",
        "",
        "Proportion of steps in each high-level category. Normalised so models are "
        "comparable despite different step counts.",
        "",
        sec1_md,
        "",
        "## 2. Intent Comparison",
        "",
        "Frequency per 100 steps, compared across all models. For Pi, the git rows use "
        "a more semantic sub-taxonomy: GitHub context, repo inspection, diff review, "
        "sync/integrate, local state change, and publish. Full table in "
        "[data/pi-analytics/intent-frequencies.csv](data/pi-analytics/intent-frequencies.csv).",
        "",
        "## 3. Steps per trajectory, by model",
        "",
        "5-step-binned distribution of trajectory length. The HTML chart renders the "
        "cumulative share of runs that finished within N steps, with a dashed line at "
        "the 250-step cap. Underlying bins: "
        "[data/pi-analytics/step-distribution.csv](data/pi-analytics/step-distribution.csv).",
        "",
        "## 4. Typical Trajectory Shape",
        "",
        "Each model is shown as a pair: benchmark (agent alone) above, "
        "maintainer-guided Pi sessions below. Markers are median-only, with no bands: "
        "△ = authorization, ○ = steering, □ = closeout, ◇ = last edit. Where a direct "
        "public benchmark run is unavailable, the benchmark row uses the closest "
        "family baseline we do have: GPT-5 for the `gpt-5.*` models and Sonnet 4.5 "
        "for the `claude-opus-4-*` models.",
        "",
        "### Pi ↔ benchmark pairing",
        "",
        pair_md,
        "",
        "### Intervention marker positions (% of trajectory)",
        "",
        "Median first occurrence of each intervention type per model, with IQR. "
        "This is the user-message analogue of structural markers like `first_edit` / "
        "`last_edit`. `steering` lumps together `solution_steer`, `evidence_or_repro`, "
        "`qa_or_critique`, and `validation_request`.",
        "",
        iv_md,
        "",
        "### Phase profiles (20-bin category mix)",
        "",
        "Per model we emit two CSVs — one for the Pi maintainer-guided run and one "
        "for the benchmark baseline (agent alone). Rows = bin 0..19, columns = share "
        "of steps in each high-level category (pre-normalisation).",
        "",
    ]
    for m in models:
        md.append(f"- {pi_label(m)} (Pi): "
                  f"[data/pi-analytics/phase-profile-{m}-pi.csv](data/pi-analytics/phase-profile-{m}-pi.csv)")
        bench_m = BENCHMARK_PAIR_FOR_PI_MODEL.get(m)
        if bench_m and bench_m in benchmark_results:
            md.append(f"- {pi_label(m)} (benchmark baseline = {bench_label(bench_m)}): "
                      f"[data/pi-analytics/phase-profile-{m}-benchmark.csv](data/pi-analytics/phase-profile-{m}-benchmark.csv)")
    md.append("")

    (REPORTS_DIR / "pi-analytics.md").write_text("\n".join(md))
    print("Wrote docs/reports-md/pi-analytics.md")


# ---------------------------------------------------------------------------
# Report 4: Pi reference (pi-reference.md)
# ---------------------------------------------------------------------------

def build_pi_reference(pi_results, pi_allowed_paths, raw_counts, resolution_stats,
                       user_data, out_dir: Path) -> None:
    csv_dir = REPORTS_DIR / "data" / "pi-reference"
    models = sorted(
        [m for m in PI_MODELS if m in pi_results],
        key=lambda m: -raw_counts.get(m, 0),
    )
    model_meta = build_pi_model_registry(models)
    # Override labels to the spec's display names.
    for m in models:
        model_meta[m] = {"label": pi_label(m), "color": model_meta[m]["color"]}

    meta = pi_aggregate.metadata_summary(pi_results)
    base = pi_aggregate.base_intent_frequencies(pi_results)
    high = pi_aggregate.high_level_frequencies(pi_results)
    phases = pi_aggregate.phase_frequencies(pi_results)
    verify_out = pi_aggregate.verify_outcomes(pi_results)
    seq = pi_aggregate.sequence_labels(pi_results)
    markers = pi_aggregate.structural_markers(pi_results)
    step_dist = pi_aggregate.step_distribution(pi_results)
    wd = pi_aggregate.work_done_vs_completed(pi_results)

    md: list[str] = [
        "# Pi Transcript Reference Tables",
        "",
        "Same high-level taxonomy as the SWE-Agent analysis, adapted to Pi tool calls "
        "and filtered to strict single-model issue sessions. Pi uses a more semantic "
        "low-level git decomposition while classifying Pi's `read`, `edit`, `write`, "
        "`bash`, and auxiliary tools into that shared scheme.",
        "",
        f"Models: {', '.join(pi_label(m) for m in models)}.",
        "",
    ]

    # ── §0 Task resolution rate (inline) ───────────────────────────────
    kinds_order = ["push", "gh_close", "gh_merge", "gh_comment", "user_close"]
    kind_labels = {
        "push": "git push", "gh_close": "gh issue close", "gh_merge": "gh pr merge",
        "gh_comment": "triage comment", "user_close": "maintainer close",
    }
    rows = []
    for m in models:
        s = resolution_stats.get(m)
        if s is None:
            continue
        rows.append([
            pi_label(m),
            s.n_sessions,
            s.n_issues_attempted,
            s.n_issues_resolved,
            f"{s.resolve_rate:.1f}%",
        ] + [s.kind_counts.get(k, 0) for k in kinds_order])
    headers = ["model", "sessions (filtered)", "distinct issues",
               "resolved", "resolve rate"] + [kind_labels[k] for k in kinds_order]
    md += [
        "## 0. Task resolution rate (per issue)",
        "",
        "Per-model resolve rate using SWE-bench-style unit of analysis: one "
        "*(model, issue#)* pair per attempt, resolved if **any** same-model session "
        "on that issue reached a terminal action. Issues are joined across sessions "
        "by the GitHub issue/PR number that appears in the session name or in the "
        "first user message. This counts all legitimate maintainer completion "
        "mechanisms (ship, triage-close, duplicate-close, won't-fix), not just "
        "code-shipped resolutions.",
        "",
        md_table(headers, rows),
        "",
        "- `git push` — agent pushed the fix to the remote.",
        "- `gh issue close` — agent ran `gh issue close`.",
        "- `gh pr merge` — agent ran `gh pr merge`.",
        "- `triage comment` — agent posted a triage comment after the maintainer "
        "asked to comment/close.",
        "- `maintainer close` — maintainer gave a terminal close/triage instruction; "
        "the agent didn't ship a shell action but the task was decided.",
        "",
        "Precedence: `push` > `gh_merge` > `gh_close` > `gh_comment` > `user_close`. "
        "A session with both a push and a triage comment is counted as `push`.",
        "",
    ]

    # ── §1 Session metadata (inline) ───────────────────────────────────
    rows = []
    for m in models:
        mm = meta[m]
        rows.append([
            pi_label(m),
            raw_counts.get(m, 0),
            mm["n"],
            mm["avg"], mm["median"], mm["p25"], mm["p75"],
            mm["min"], mm["max"],
            mm["completed"],
            mm["completion_rate"],
        ])
    md += [
        "## 1. Session metadata",
        "",
        "Strict single-model purity is determined from both `assistant.message.model` "
        "and `model_change.modelId`. 'single-model sessions' is the raw eligible "
        "count; 'analyzed sessions' is the subset with tool-call trajectories that "
        "can actually be classified.",
        "",
        md_table(
            ["model", "single-model sessions", "analyzed sessions", "avg steps",
             "median", "p25", "p75", "min", "max", "completed", "completion rate"],
            rows,
        ),
        "",
    ]

    # ── §2 End states (inline) ─────────────────────────────────────────
    # Each cell will describe counts.
    rows = []
    all_exits: set = set()
    for m in models:
        for k in meta[m]["exits"]:
            all_exits.add(k)
    sorted_exits = sorted(
        all_exits, key=lambda e: -sum(meta[m]["exits"].get(e, 0) for m in models)
    )
    headers = ["end state"] + [pi_label(m) for m in models]
    rows = []
    for e in sorted_exits:
        rows.append([e] + [meta[m]["exits"].get(e, 0) for m in models])
    md += [
        "## 2. End states",
        "",
        "These are transcript-level end states, derived from the final assistant "
        "stop reason recorded in the session.",
        "",
        md_table(headers, rows),
        "",
    ]

    # ── §3 High-level category mix (inline) ────────────────────────────
    cat_headers = ["read", "search", "reproduce", "edit", "verify",
                   "git", "housekeeping", "failed", "other"]
    rows = []
    for m in models:
        props = high["proportions"].get(m, {})
        rows.append([pi_label(m)] + [f"{props.get(k, 0) * 100:.1f}%" for k in cat_headers])
    md += [
        "## 3. High-level category mix",
        "",
        "Shares of all classified tool steps by high-level category.",
        "",
        md_table(["model"] + cat_headers, rows),
        "",
    ]

    # ── §3b Detailed classification breakdown (CSV) ────────────────────
    counts_per_intent: dict[str, dict[str, int]] = {}
    totals_per_model = {m: 0 for m in models}
    for m in models:
        c: dict[str, int] = {}
        total = 0
        for row in pi_results[m]:
            total += row.steps
            for intent, n in row.base_intent_counts.items():
                c[intent] = c.get(intent, 0) + n
        totals_per_model[m] = total
        counts_per_intent[m] = c

    category_order = ["read", "search", "reproduce", "edit", "verify",
                      "git", "housekeeping", "failed", "other"]
    dc_headers = ["category", "intent", "description"] + sum(
        [[f"{m}_count", f"{m}_%"] for m in models], []
    )
    dc_rows = []
    for cat in category_order:
        intents = [
            intent for intent, high_ in PI_INTENT_TO_HIGH_LEVEL.items()
            if high_ == cat and any(counts_per_intent[m].get(intent, 0) > 0 for m in models)
        ]
        intents.sort(key=lambda i: (-sum(counts_per_intent[m].get(i, 0) for m in models), i))
        for intent in intents:
            row = [cat, intent, PI_INTENT_DESCRIPTIONS.get(intent, "")]
            for m in models:
                n = counts_per_intent[m].get(intent, 0)
                p = (n / totals_per_model[m] * 100) if totals_per_model[m] else 0
                row += [n, f"{p:.2f}"]
            dc_rows.append(row)
    write_csv(csv_dir / "detailed-classification.csv", dc_headers, dc_rows)

    md += [
        "## 3b. Detailed classification breakdown",
        "",
        "Detailed per-intent table from the Pi harness. Each row shows **count** and "
        "**share of all classified tool steps for that model**, grouped by the same "
        "high-level taxonomy as the original reference tables — so you can inspect "
        "exactly what is inside categories like *cleanup* and *other*. Full table: "
        "[data/pi-reference/detailed-classification.csv](data/pi-reference/detailed-classification.csv).",
        "",
    ]

    # ── §4 Phase mix (inline) ──────────────────────────────────────────
    if phases:
        phase_keys = list(next(iter(phases.values())).keys())
    else:
        phase_keys = []
    rows = []
    for m in models:
        rows.append([pi_label(m)] + [f"{phases[m].get(p, 0):.1f}%" for p in phase_keys])
    md += [
        "## 4. Phase mix",
        "",
        "Phase grouping reused from the original analysis: understand = read + "
        "search, cleanup = git + housekeeping.",
        "",
        md_table(["model"] + phase_keys, rows),
        "",
    ]

    # ── §4b Cleanup decomposition (inline) ─────────────────────────────
    preferred_order = [
        "git-github-context", "git-repo-inspect", "git-diff-review",
        "git-sync-integrate", "git-local-state-change", "git-publish",
        "file-cleanup", "create-documentation", "start-service",
        "install-deps", "check-tool-exists",
    ]
    cleanup_intents = [
        i for i in preferred_order
        if any(counts_per_intent[m].get(i, 0) > 0 for m in models)
    ]
    rows = []
    for intent in cleanup_intents:
        high_ = PI_INTENT_TO_HIGH_LEVEL.get(intent, "")
        row = [high_, intent, PI_INTENT_DESCRIPTIONS.get(intent, "")]
        for m in models:
            n = counts_per_intent[m].get(intent, 0)
            p = (n / totals_per_model[m] * 100) if totals_per_model[m] else 0
            row.append(f"{n} ({p:.2f}%)")
        rows.append(row)
    # Summary rows
    for label, members in [
        ("git total", [i for i in cleanup_intents if PI_INTENT_TO_HIGH_LEVEL.get(i) == "git"]),
        ("housekeeping total", [i for i in cleanup_intents if PI_INTENT_TO_HIGH_LEVEL.get(i) == "housekeeping"]),
        ("cleanup phase total", cleanup_intents),
    ]:
        row = ["summary", label, ""]
        for m in models:
            n = sum(counts_per_intent[m].get(i, 0) for i in members)
            p = (n / totals_per_model[m] * 100) if totals_per_model[m] else 0
            row.append(f"{n} ({p:.2f}%)")
        rows.append(row)
    md += [
        "## 4b. Cleanup decomposition",
        "",
        "In the inherited phase schema, **cleanup = git + housekeeping**. For Pi "
        "transcripts this phase is mostly repo workflow, not literal cleanup. This "
        "table makes the git side explicit.",
        "",
        md_table(
            ["high-level", "intent", "description"] + [pi_label(m) for m in models],
            rows,
        ),
        "",
    ]

    # ── §5 Verify outcomes (inline) ────────────────────────────────────
    rows = []
    for m in models:
        v = verify_out[m]
        rows.append([
            pi_label(m), v["pass"], v["fail"], v["unknown"], v["total"], v["pass_rate"],
        ])
    md += [
        "## 5. Verify outcomes",
        "",
        "Verify pass/fail uses the original deterministic parser over bash "
        "observations. This mainly applies to test/build commands run through Pi's "
        "`bash` tool.",
        "",
        md_table(["model", "pass", "fail", "unknown", "total verify steps", "pass rate"], rows),
        "",
    ]

    # ── §6 Most common base intents (CSV) ──────────────────────────────
    top_intents = base["top_intents"]
    rows = []
    for intent in top_intents:
        row = [intent]
        for m in models:
            pct_v = base["proportions"].get(m, {}).get(intent, 0) * 100
            row.append(f"{pct_v:.2f}")
        rows.append(row)
    write_csv(
        csv_dir / "base-intents.csv",
        ["intent"] + [f"{pi_label(m)}_%" for m in models],
        rows,
    )
    # Top-10 inline
    top_rows = rows[:10]
    md += [
        "## 6. Most common base intents",
        "",
        "Same base-intent taxonomy as the SWE-Agent analysis, but applied to Pi tool "
        "calls by mapping `read`/`edit`/`write`/`bash` into equivalent intent "
        "semantics. Top 10 shown inline; full list (percentages across all "
        "classified tool steps per model) in "
        "[data/pi-reference/base-intents.csv](data/pi-reference/base-intents.csv).",
        "",
        md_table(["intent"] + [f"{pi_label(m)} %" for m in models], top_rows),
        "",
    ]

    # ── §7 Sequence-layer labels (inline) ──────────────────────────────
    rows = []
    for label in seq["all_labels"][:20]:
        row = [label]
        for m in models:
            row.append(seq["counts"].get(m, {}).get(label, 0))
        rows.append(row)
    md += [
        "## 7. Sequence-layer labels",
        "",
        "These are second-pass labels derived from nearby history, such as "
        "verify-after-edit or first-all-pass after the last source edit.",
        "",
        md_table(["label"] + [pi_label(m) for m in models], rows),
        "",
    ]

    # ── §8 Structural markers (CSV) ────────────────────────────────────
    marker_order = [
        ("first_edit", "first edit"),
        ("last_edit", "last edit"),
        ("first_verify", "first verify"),
        ("first_verify_pass", "first verify pass"),
        ("submit", "finish / submit"),
    ]
    sm_rows = []
    for key, label in marker_order:
        for m in models:
            info = markers[key][m]
            sm_rows.append([
                pi_label(m), label, info["median"], info["p25"], info["p75"],
                info.get("completed_median"), info.get("incomplete_median"),
            ])
    write_csv(
        csv_dir / "structural-markers.csv",
        ["model", "marker", "median_%", "p25", "p75",
         "completed_median", "incomplete_median"],
        sm_rows,
    )
    md += [
        "## 8. Structural markers",
        "",
        "Marker positions are measured as a percentage of session length. 'Completed' "
        "and 'incomplete' split by clean session completion, not correctness. Full "
        "table: [data/pi-reference/structural-markers.csv](data/pi-reference/structural-markers.csv).",
        "",
    ]

    # ── §9 Work done vs completion (inline) ────────────────────────────
    rows = []
    for m in models:
        info = wd[m]
        n = max(info["n"], 1)
        rows.append([
            pi_label(m),
            info["wd_completed"], info["wd_incomplete"],
            info["no_wd_completed"], info["no_wd_incomplete"],
            f"{info['wd_completed'] / n * 100:.1f}%",
        ])
    md += [
        "## 9. Work done vs completion",
        "",
        "`work done` keeps the original meaning: the transcript reaches a verify "
        "pass after its last source edit. Here we compare that against whether the "
        "session ended cleanly.",
        "",
        md_table(
            ["model", "wd + completed", "wd + incomplete", "no wd + completed",
             "no wd + incomplete", "wd+completed rate"],
            rows,
        ),
        "",
    ]

    # ── §10 Step-count distribution (inline) ───────────────────────────
    # Collapsed view: show bins where at least one model has >=1 sessions.
    bins_set = set()
    for m in models:
        bins_set.update(step_dist.get(m, {}).keys())
    bin_list = sorted(bins_set)
    rows = []
    for b in bin_list:
        rows.append([f"{b}-{b + 4}"] + [step_dist.get(m, {}).get(b, 0) for m in models])
    md += [
        "## 10. Step-count distribution",
        "",
        "5-step bins. Counts are number of analyzed sessions per model that ended "
        "with that number of tool-call steps.",
        "",
        md_table(["bin"] + [pi_label(m) for m in models], rows),
        "",
    ]

    # ── §11 Maintainer intervention markers (inline) ────────────────────
    macro_defs = [
        ("analysis_start", "analysis starts", ["task_brief"]),
        ("work_start", "work starts (authorized)", ["authorize_work"]),
        ("solution_steering", "solution steering",
         ["solution_steer", "evidence_or_repro", "qa_or_critique", "validation_request"]),
        ("workflow_closeout", "workflow closeout", ["workflow_closeout"]),
    ]

    def _macro_stats(classes: dict, num_sessions: int):
        rows_ = {}
        for key, _, members in macro_defs:
            recs = []
            for lbl in members:
                recs.extend(classes.get(lbl, {}).get("messages", []))
            per_path: dict = {}
            for rec in recs:
                per_path.setdefault(rec["path"], []).append(rec)
            firsts = []
            for pth, msgs in per_path.items():
                msgs.sort(key=lambda m_: (m_["message_index"], m_["progress_pct"]))
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
            rows_[key] = {
                "session_count": len(per_path),
                "session_pct": round(len(per_path) / num_sessions * 100, 1) if num_sessions else 0.0,
                "median": med, "p25": p25, "p75": p75,
            }
        return rows_

    # Overall + per-model
    overall_classes = {
        lbl: {"messages": sum(
            (user_data["per_model"][m]["classes"][lbl]["messages"] for m in models),
            [])}
        for lbl in CLASS_ORDER
    }
    overall_stats = _macro_stats(overall_classes, user_data["total_sessions"])

    headers = ["scope"]
    for _, label, _m in macro_defs:
        headers += [f"{label} %sessions", f"{label} median", f"{label} p25", f"{label} p75"]
    rows = []
    row = ["All models combined"]
    for key, _, _m in macro_defs:
        s = overall_stats[key]
        row += [f"{s['session_pct']:.1f}", s["median"], s["p25"], s["p75"]]
    rows.append(row)
    for m in models:
        classes = user_data["per_model"][m]["classes"]
        num_sessions = user_data["per_model"][m]["num_sessions"]
        stats = _macro_stats(classes, num_sessions)
        row = [pi_label(m)]
        for key, _, _m in macro_defs:
            s = stats[key]
            row += [f"{s['session_pct']:.1f}", s["median"], s["p25"], s["p75"]]
        rows.append(row)

    md += [
        "## 11. Maintainer intervention markers",
        "",
        "Higher-level phase view derived from the 7 user-message classes. For each "
        "scope we show the **median first occurrence** of that intervention type as a "
        "percentage of trajectory progress, with p25/p75. Solution steering lumps "
        "together `solution_steer`, `evidence_or_repro`, `qa_or_critique`, and "
        "`validation_request`.",
        "",
        "`% sessions` is the share of sessions in that scope that contained at least "
        "one message of that macro class.",
        "",
        md_table(headers, rows),
        "",
    ]

    # ── §12 User message classes (inline) ──────────────────────────────
    total_sessions = user_data["total_sessions"]
    total_messages = user_data["total_messages"]
    class_desc = user_data["class_descriptions"]
    rows = []
    for label in CLASS_ORDER:
        stats = user_data["overall"][label]
        rows.append([
            label,
            class_desc[label],
            stats["message_count"],
            f"{stats['message_pct']:.1f}%",
            stats["session_count"],
            f"{stats['session_pct']:.1f}%",
            stats["first_progress_median"],
            stats["first_progress_p25"],
            stats["first_progress_p75"],
        ])
    md += [
        "## 12. User message classes",
        "",
        "These counts are computed over the raw filtered issue sessions, not just "
        "the classified tool-step subset. Messages are assigned a single primary "
        "class using a deterministic, dataset-tuned rule set. The timing columns use "
        "the same trajectory-normalised 0-100% progress scale as the stacked "
        "trajectory-shape charts: for each user message, we count how many assistant "
        "tool calls have already happened in that session.",
        "",
        f"**{total_messages}** user messages across **{total_sessions}** strict "
        "single-model issue sessions.",
        "",
        md_table(
            ["class", "description", "messages", "% of messages", "sessions",
             "% of sessions", "median first %", "p25", "p75"],
            rows,
        ),
        "",
    ]

    # ── §13 User intervention timing by model (CSV) ────────────────────
    # One row per (model × class × bin). Store full 20-bin presence as wide columns.
    headers = [
        "model", "class", "messages", "message_pct", "sessions",
        "session_pct", "first_progress_median", "first_progress_p25",
        "first_progress_p75",
    ] + [f"bin_{i:02d}_pct" for i in range(20)]
    rows = []
    for m in models:
        pdata = user_data["per_model"].get(m, {})
        classes = pdata.get("classes", {})
        for label in CLASS_ORDER:
            s = classes.get(label, {})
            bin_pct = s.get("bin_session_pct", [0.0] * 20)
            rows.append([
                pi_label(m), label,
                s.get("message_count", 0),
                f"{s.get('message_pct', 0):.2f}",
                s.get("session_count", 0),
                f"{s.get('session_pct', 0):.2f}",
                s.get("first_progress_median"),
                s.get("first_progress_p25"),
                s.get("first_progress_p75"),
            ] + [f"{v:.2f}" for v in bin_pct])
    write_csv(csv_dir / "user-intervention-timing.csv", headers, rows)

    md += [
        "## 13. User intervention timing by model",
        "",
        "Per-model timing for every user-message class. Each row has a full 20-bin "
        "(5% each) trajectory-normalised timeline: `bin_00_pct` is the percentage of "
        "that model's sessions with at least one message of that class in bin 0 "
        "(steps 0-5%), `bin_01_pct` bin 1 (5-10%), etc. The original HTML renders "
        "this as a heatmap strip per model per class. "
        "[data/pi-reference/user-intervention-timing.csv](data/pi-reference/user-intervention-timing.csv).",
        "",
    ]

    # ── §14 All user messages by class (separate MD file) ──────────────
    ums_path = csv_dir / "user-messages.md"
    ums_lines: list[str] = [
        "# Pi Reference — User Messages (§14)",
        "",
        "Every classified user message in the filtered issue subset. Each entry "
        "records the session, user-turn index, and trajectory progress when that "
        "interruption happened. Originally rendered as collapsible `<details>` "
        "blocks in the HTML report.",
        "",
    ]
    for label in CLASS_ORDER:
        overall = user_data["overall"][label]
        ums_lines += [
            f"## {label}",
            "",
            class_desc[label],
            "",
            f"{overall['message_count']} messages across "
            f"{overall['session_count']} sessions "
            f"({overall['session_pct']:.1f}% of sessions).",
            "",
        ]
        for m in models:
            stats = user_data["per_model"][m]["classes"][label]
            if not stats["messages"]:
                continue
            ums_lines += [
                f"### {pi_label(m)} — {stats['message_count']} messages in "
                f"{stats['session_count']} sessions",
                "",
            ]
            for msg in stats["messages"]:
                session_name = msg["session_name"] or Path(msg["path"]).name
                text = (msg["text"] or "<empty>").replace("\n", " ").strip()
                ums_lines.append(
                    f"- **{session_name}** (turn {msg['message_index']}, "
                    f"{msg['progress_pct']:.1f}% through trajectory, "
                    f"{Path(msg['path']).name}): {text}"
                )
            ums_lines.append("")
    ums_path.write_text("\n".join(ums_lines))

    md += [
        "## 14. All user messages by class",
        "",
        "Every classified user message in the filtered issue subset, grouped by "
        "class and then by model. This is free-form multi-line text rather than "
        "tabular data, so it lives in a separate Markdown file: "
        "[data/pi-reference/user-messages.md](data/pi-reference/user-messages.md).",
        "",
    ]

    (REPORTS_DIR / "pi-reference.md").write_text("\n".join(md))
    print("Wrote docs/reports-md/pi-reference.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_root = ROOT / "data"
    pi_data_root = ROOT / "data" / "pi-mono"

    # ── Benchmark data (for analytics.md, reference.md, pi-analytics.md pairing) ──
    print("Processing benchmark trajectories (claude45, gpt5)...")
    bench_results = process_benchmark(data_root, models=BENCH_MODELS)
    for m in BENCH_MODELS:
        print(f"  {m}: {len(bench_results.get(m, []))} trajectories")

    # Build analytics.md + reference.md
    failure_path = data_root / "failure_modes.json"
    failure_data = json.loads(failure_path.read_text()) if failure_path.exists() else None

    build_benchmark_analytics(bench_results, REPORTS_DIR)
    build_benchmark_reference(bench_results, failure_data, REPORTS_DIR)

    # ── Pi data (for pi-analytics.md, pi-reference.md) ───────────────────
    print("Processing Pi transcripts...")
    session_filter = SessionFilter(
        allowed_models=PI_MODELS,
        require_single_model=True,
        session_name_prefixes=["Issue:"],
    )
    allowed_paths, raw_counts, _ = collect_filtered_paths(pi_data_root, session_filter)
    pi_results = process_pi(pi_data_root, models=PI_MODELS)
    pi_results = _filter_pi_results(pi_results, allowed_paths)
    for m in PI_MODELS:
        print(f"  {m}: {raw_counts.get(m, 0)} filtered sessions, "
              f"{len(pi_results.get(m, []))} analyzed")

    user_data = analyze_user_messages(allowed_paths)
    resolution_stats = compute_resolution_by_model(pi_data_root, session_filter)

    build_pi_analytics(pi_results, allowed_paths, raw_counts, bench_results, REPORTS_DIR)
    build_pi_reference(pi_results, allowed_paths, raw_counts, resolution_stats,
                       user_data, REPORTS_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
