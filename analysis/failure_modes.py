"""
Failure mode classifier.

For each trajectory step, determine whether the step represents a *failure*
(tool call rejected, command crashed, code raised, etc.) and what kind.

Returns one of FAILURE_MODES keys, or None if the step is not a failure.

Categories are organized into three families:

  TOOL    — the agent's tool invocation itself failed (the tool refused or
            errored before useful work happened). E.g. apply_patch command not
            found, str_replace old_str didn't match, bash quoting broke.
  CODE    — the agent ran code (script / interpreter / one-liner) and the
            interpreter raised or printed a traceback.
  TEST    — the agent ran a test suite that reported failing or erroring tests.

GPT-5-specific patterns (apply_patch hallucination, trailing `}` from JSON
serialization leak, double-quote nesting in bash -lc) are intentionally
captured as their own modes because they account for most of GPT-5's friction.
"""

from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Failure mode catalog: (key, family, label, description)
# ---------------------------------------------------------------------------

FAILURE_MODES: list[tuple[str, str, str, str]] = [
    # ── TOOL: apply_patch (GPT-5 hallucinated tool) ──────────────────────────
    ("apply_patch_cmd_not_found",  "tool", "apply_patch missing",
     "Agent invoked `applypatch` but the binary doesn't exist in the sandbox. "
     "GPT-5 hallucinates this from its OpenAI/Codex training where apply_patch "
     "is the canonical edit tool."),
    ("apply_patch_shell_syntax",   "tool", "apply_patch shell syntax",
     "`applypatch <<'PATCH' …` heredoc broke bash parsing — typically because "
     "the patch body contains `(`, `)`, or backticks that the shell tried to "
     "interpret before reaching applypatch."),
    ("apply_patch_other",          "tool", "apply_patch other",
     "Other failure paths involving applypatch (path errors, malformed patch, "
     "history-expansion `!keys`, etc.)."),

    # ── TOOL: bash -lc wrapper (GPT-5-specific quoting/serialization bugs) ──
    ("bash_trailing_brace",        "tool", "trailing `}` leak",
     "Action ends with a stray `}` (from JSON tool-call serialization), so the "
     "shell sees `… || true}` and reports `true}: command not found`. "
     "Almost exclusive to GPT-5's `bash -lc \"…\"}` wrapping pattern."),
    ("bash_quote_nesting",         "tool", "bash quote nesting",
     "Inner unescaped `\"` inside `bash -lc \"…\"` terminates the outer quote "
     "early, leaving fragments the shell tries to execute as separate commands."),
    ("bash_history_expansion",     "tool", "bash `!` history expansion",
     "`!foo: event not found` — patch/script body contains `!` (e.g. `!keys`, "
     "`!isPrivileged`) which bash interprets as history expansion."),
    ("bash_heredoc_unterminated",  "tool", "heredoc unterminated",
     "`warning: here-document at line N delimited by end-of-file` — the "
     "heredoc end-marker was emitted on the same line, escaped, or never sent."),
    ("bash_command_not_found",     "tool", "command not found",
     "Agent tried a binary that isn't installed (`rg`, `go`, `ripgrep`, …) or "
     "a token that the shell parsed as a command."),
    ("bash_syntax_error",          "tool", "bash syntax error",
     "`syntax error near unexpected token` — unbalanced parentheses, braces, "
     "or backticks in the agent's shell command."),
    ("bash_broken_pipe",           "tool", "broken pipe",
     "`grep: write error: Broken pipe` — usually downstream of one of the "
     "above (the rest of the pipeline already crashed)."),

    # ── TOOL: str_replace_editor ─────────────────────────────────────────────
    ("strep_no_match",             "tool", "str_replace no match",
     "`--old_str` did not appear verbatim in the file — a single character of "
     "drift (whitespace, quote style, an old line that was already changed) "
     "is enough to reject the edit."),
    ("strep_multiple_matches",     "tool", "str_replace not unique",
     "`--old_str` matches multiple locations and the editor refuses to guess."),
    ("strep_file_not_found",       "tool", "str_replace path missing",
     "Target path doesn't exist in the repo — typically the agent inferred a "
     "path from the PR description that turned out wrong."),
    ("strep_invalid_range",        "tool", "view_range out of bounds",
     "`Invalid view_range` — agent asked to view lines past EOF or with a "
     "reversed range."),
    ("strep_create_exists",        "tool", "create over existing file",
     "`File already exists … cannot overwrite using create` — agent forgot "
     "it had already created e.g. `repro.py`."),

    # ── CODE: script the agent ran crashed ───────────────────────────────────
    ("py_module_not_found",        "code", "ModuleNotFoundError",
     "Reproduction or verify script failed at import time — env doesn't have "
     "the module, or the script ran from the wrong cwd."),
    ("py_syntax_error",            "code", "SyntaxError",
     "Python interpreter rejected the source — usually a heredoc or inline "
     "`python -c` snippet whose quoting got mangled in transit."),
    ("py_indentation_error",       "code", "IndentationError",
     "Same family as SyntaxError, but specifically broken indentation — "
     "common when GPT-5 inserts code via str_replace and tabs/spaces drift."),
    ("py_traceback_other",         "code", "Python traceback",
     "Any other Python traceback — TypeError, AttributeError, NameError, "
     "etc. surfaced from the agent's repro script, not from the test suite."),
    ("node_error",                 "code", "Node error",
     "Node.js threw at top level (TypeError, SyntaxError in `node -e`, etc.)."),

    # ── TEST: test runner reported failures ─────────────────────────────────
    ("test_failed",                "test", "test suite failed",
     "Test runner ran to completion but reported `N failed`/`N error`/`FAIL`."),
    ("test_collection_error",      "test", "test collection error",
     "pytest or jest couldn't even collect the tests — usually an import error "
     "in the test file itself, often caused by the agent's edit."),
]

MODES_BY_FAMILY: dict[str, list[str]] = {}
for key, family, _label, _desc in FAILURE_MODES:
    MODES_BY_FAMILY.setdefault(family, []).append(key)

MODE_LABEL: dict[str, str] = {k: lbl for k, _, lbl, _ in FAILURE_MODES}
MODE_DESC: dict[str, str] = {k: desc for k, _, _, desc in FAILURE_MODES}
MODE_FAMILY: dict[str, str] = {k: fam for k, fam, _, _ in FAILURE_MODES}

FAMILY_LABEL = {
    "tool": "Tool-call failure",
    "code": "Agent script crashed",
    "test": "Test-suite failure",
}

FAMILY_COLOR = {
    "tool": "#b0856a",   # warm brown — agent infrastructure friction
    "code": "#7a8aaa",   # muted blue — agent's own code crashed
    "test": "#aa7a8a",   # muted plum — actual test outcome
}

# Per-mode tints (lighten/darken family color)
MODE_COLOR: dict[str, str] = {}
_TINTS_TOOL = ["#7a4f3a", "#8a5e48", "#985f44", "#a3725d", "#b0856a", "#bb9479",
               "#c4a08a", "#cdac9a", "#d6b9aa", "#dfc6ba", "#e7d2c8", "#eedfd4"]
_TINTS_CODE = ["#5a6a8a", "#6a7a9a", "#7a8aaa", "#8a9aba", "#9aaaca", "#aabbda"]
_TINTS_TEST = ["#8a4a5a", "#aa7a8a"]
for i, k in enumerate(MODES_BY_FAMILY.get("tool", [])):
    MODE_COLOR[k] = _TINTS_TOOL[i % len(_TINTS_TOOL)]
for i, k in enumerate(MODES_BY_FAMILY.get("code", [])):
    MODE_COLOR[k] = _TINTS_CODE[i % len(_TINTS_CODE)]
for i, k in enumerate(MODES_BY_FAMILY.get("test", [])):
    MODE_COLOR[k] = _TINTS_TEST[i % len(_TINTS_TEST)]


# ---------------------------------------------------------------------------
# Regex helpers — compiled once.
# ---------------------------------------------------------------------------

_RE_PYTEST_FAIL = re.compile(r"=+\s+\d+\s+(?:failed|error)", re.IGNORECASE)
_RE_PYTEST_PASS_ONLY = re.compile(r"=+\s+\d+\s+passed[\s,]+in\s+", re.IGNORECASE)
_RE_PYTEST_COLLECT_ERR = re.compile(r"errors? during collection|collection error",
                                    re.IGNORECASE)
_RE_MOCHA_FAILING = re.compile(r"\d+\s+failing")
_RE_GO_FAIL = re.compile(r"^FAIL\s+\S+", re.MULTILINE)
_RE_TRACEBACK = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_RE_NODE_THROW = re.compile(r"^\s*throw\s+|node:internal/", re.MULTILINE)


def _ends_with_brace_leak(action: str) -> bool:
    """Detect the `bash -lc \"...\"}` JSON serialization leak."""
    s = action.rstrip()
    if not s.endswith("}"):
        return False
    # Heuristic: stray `}` is on a line that started with `bash -lc`
    first_line = s.split("\n", 1)[0]
    return first_line.lower().startswith("bash -lc") and first_line.endswith("}")


def classify_failure(action: str, observation: str) -> str | None:
    """Return failure mode key, or None if step is not a failure.

    Order matters: most specific patterns first so e.g. an apply_patch shell
    syntax error doesn't get bucketed as a generic bash syntax error.
    """
    action = action or ""
    observation = observation or ""
    al = action.lower()
    ol = observation.lower()

    # Skip obvious non-errors quickly
    if not ol:
        return None

    action_is_applypatch = al.lstrip().startswith("applypatch") or "applypatch <<" in al
    action_is_bash_lc = al.lstrip().startswith("bash -lc")
    action_is_strep = al.lstrip().startswith("str_replace_editor")

    # ── apply_patch family ──────────────────────────────────────────────────
    if action_is_applypatch:
        if "applypatch: command not found" in ol or "applypatch:command not found" in ol:
            return "apply_patch_cmd_not_found"
        if "syntax error near unexpected token" in ol or "here-document at line" in ol:
            return "apply_patch_shell_syntax"
        if "event not found" in ol:
            return "bash_history_expansion"
        if "command not found" in ol:
            # Some applypatch attempts produce other CNF (e.g. `can: line 14`)
            return "apply_patch_cmd_not_found"
        if any(s in ol for s in ("no such file", "is a directory",
                                  "invalid patch", "patch failed",
                                  "could not find", "context does not match")):
            return "apply_patch_other"
        # No error markers — assume it didn't actually run
        return None

    # ── str_replace_editor family ────────────────────────────────────────────
    if action_is_strep:
        if ("did not appear verbatim" in ol
                or "no replacement was performed" in ol
                or "no match" in ol
                or "did not match" in ol):
            return "strep_no_match"
        if ("multiple occurrences" in ol
                or "multiple matches" in ol
                or "matches the string to replace" in ol):
            return "strep_multiple_matches"
        if "file already exists" in ol and "create" in al:
            return "strep_create_exists"
        if "invalid `view_range`" in ol or "invalid view_range" in ol:
            return "strep_invalid_range"
        if "does not exist" in ol and "please provide a valid path" in ol:
            return "strep_file_not_found"
        if "syntax error near unexpected token" in ol:
            # The str_replace tool itself shells out — sometimes the error
            # leaks back through the wrapper.
            return "bash_syntax_error"
        return None

    # ── bash -lc / direct shell command failures ────────────────────────────
    # Trailing `}` from JSON tool-call serialization is highest-signal.
    if _ends_with_brace_leak(action) and "command not found" in ol:
        return "bash_trailing_brace"

    if "event not found" in ol:
        return "bash_history_expansion"

    if "warning: here-document" in ol or "unexpected end of file" in ol:
        return "bash_heredoc_unterminated"

    if "syntax error near unexpected token" in ol:
        return "bash_syntax_error"

    if "broken pipe" in ol and "command not found" not in ol:
        # Pure broken-pipe cases (no upstream cnf)
        return "bash_broken_pipe"

    if "command not found" in ol:
        # Heuristic: distinguish quote-nesting (multiple cnfs from inner-quote
        # fragmentation) from true missing binaries.
        cnf_count = ol.count("command not found")
        if action_is_bash_lc and cnf_count >= 2 and '"' in action:
            return "bash_quote_nesting"
        return "bash_command_not_found"

    # ── Code-run failures (the agent ran a script that raised) ──────────────
    if "modulenotfounderror" in ol or "no module named" in ol:
        return "py_module_not_found"
    if "syntaxerror" in ol and "python" in al + "  " + ol[:200]:
        return "py_syntax_error"
    if "indentationerror" in ol:
        return "py_indentation_error"
    if _RE_NODE_THROW.search(observation[-1000:]) or "node:internal" in ol:
        return "node_error"

    # ── Test runner outcomes (only when action was clearly a test run) ──────
    is_test_run = any(t in al for t in (
        "pytest", "npm test", "npm run test", "mocha", "jest", "go test",
        "rspec", "bundle exec", "tox", "phpunit", "make test"))
    if is_test_run:
        if _RE_PYTEST_COLLECT_ERR.search(observation):
            return "test_collection_error"
        if _RE_PYTEST_FAIL.search(observation) or _RE_MOCHA_FAILING.search(observation):
            return "test_failed"
        if _RE_GO_FAIL.search(observation):
            return "test_failed"

    # Generic Python traceback — script the agent wrote crashed.
    if _RE_TRACEBACK.search(observation[-1500:]):
        return "py_traceback_other"

    return None


def analyze_trajectory(trajectory: list[dict]) -> dict:
    """Return per-trajectory failure stats.

    Returns:
        {
            "n_steps": int,
            "n_failures": int,
            "mode_counts": Counter[mode_key],
            "had_mode": set[mode_key],   # which modes appeared at least once
        }
    """
    mode_counts: Counter = Counter()
    had_mode: set[str] = set()
    n_steps = len(trajectory)
    n_failures = 0
    for step in trajectory:
        action = step.get("action") or ""
        obs = step.get("observation") or ""
        mode = classify_failure(action, obs)
        if mode:
            mode_counts[mode] += 1
            had_mode.add(mode)
            n_failures += 1
    return {
        "n_steps": n_steps,
        "n_failures": n_failures,
        "mode_counts": dict(mode_counts),
        "had_mode": sorted(had_mode),
    }
