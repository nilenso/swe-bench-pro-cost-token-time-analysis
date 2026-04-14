#!/usr/bin/env python3
"""
Deterministic command-level intent classifier for SWE-Agent trajectories.

Rules are implemented from:
  docs/intent-classification-rules.md

Classification is based on literal command/action text (+ filename + limited
observation checks for error/truncation/directory listing), without positional
phase context.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None


CONFIG_FILES = {
    "package.json",
    "pytest.ini",
    "setup.cfg",
    "setup.py",
    "go.mod",
    "makefile",
    "config.json",
}

TEST_FILE_RE = re.compile(r"^(test_.*|.*_test\..*|conftest.*)$")

# Sequence-layer groups (second pass over base intents)
SEQUENCE_VERIFY_INTENTS = {
    "run-test-suite",
    "run-test-specific",
    "run-verify-script",
    "run-custom-script",
    "compile-build",
    "syntax-check",
}
SEQUENCE_REPRO_INTENTS = {
    "run-repro-script",
    "run-inline-snippet",
}
SEQUENCE_EDIT_INTENTS = {
    "edit-source",
    "insert-source",
    "apply-patch",
    "edit-test-or-repro",
    "create-file",
}
SEQUENCE_FAILED_VERIFY_INTENTS = {
    "run-test-suite(failed)",
    "run-script(failed)",
}
SEQUENCE_READ_INTENTS = {
    "read-file-full",
    "read-file-range",
    "read-file-full(truncated)",
    "read-test-file",
    "read-config-file",
    "read-via-bash",
}
SEQUENCE_SEARCH_INTENTS = {
    "search-keyword",
    "search-files-by-name",
    "search-files-by-content",
    "search-keyword(failed)",
}

# Third-layer hierarchical grouping: <high-level>.<base-intent>
INTENT_TO_HIGH_LEVEL = {
    # read
    "read-file-full": "read",
    "read-file-range": "read",
    "read-file-full(truncated)": "read",
    "read-test-file": "read",
    "read-config-file": "read",
    "read-via-bash": "read",

    # search
    "view-directory": "search",
    "list-directory": "search",
    "search-keyword": "search",
    "search-files-by-name": "search",
    "search-files-by-content": "search",
    "inspect-file-metadata": "search",

    # reproduce
    "create-repro-script": "reproduce",
    "run-repro-script": "reproduce",
    "run-inline-snippet": "reproduce",

    # edit
    "edit-source": "edit",
    "insert-source": "edit",
    "apply-patch": "edit",
    "create-file": "edit",

    # verify
    "run-test-suite": "verify",
    "run-test-specific": "verify",
    "create-test-script": "verify",
    "run-verify-script": "verify",
    "create-verify-script": "verify",
    "edit-test-or-repro": "verify",
    "run-custom-script": "verify",
    "syntax-check": "verify",
    "compile-build": "verify",

    # git
    "git-diff": "git",
    "git-status-log": "git",
    "git-stash": "git",

    # housekeeping
    "file-cleanup": "housekeeping",
    "create-documentation": "housekeeping",
    "start-service": "housekeeping",
    "install-deps": "housekeeping",
    "check-tool-exists": "housekeeping",

    # failed
    "search-keyword(failed)": "failed",
    "read-via-bash(failed)": "failed",
    "run-script(failed)": "failed",
    "run-test-suite(failed)": "failed",
    "bash-command(failed)": "failed",

    # other
    "submit": "other",
    "empty": "other",
    "echo": "other",
    "bash-other": "other",
    "undo-edit": "other",
}


def _load_json(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def _safe_first_line(text: str) -> str:
    return (text or "").split("\n", 1)[0]


def _command_signature(action: str) -> str:
    """Normalized command signature for sequence-level rerun detection."""
    cmd = _strip_leading_env_and_timeout(_unwrap_command(action or ""))
    head = _safe_first_line(cmd).lower().strip()
    return re.sub(r"\s+", " ", head)


def _extract_path(action_line: str) -> str:
    """Extract target path token from str_replace_editor commands."""
    try:
        tokens = shlex.split(action_line)
    except ValueError:
        tokens = action_line.split()
    return tokens[2] if len(tokens) >= 3 else ""


def _basename(path: str) -> str:
    p = (path or "").rstrip("/")
    return os.path.basename(p) if p else ""


def _has_no_extension_excluding_leading_dot(name: str) -> bool:
    if not name:
        return False
    check = name[1:] if name.startswith(".") else name
    return "." not in check


def _contains_any(haystack: str, needles: list[str] | tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def _startswith_any(haystack: str, prefixes: list[str] | tuple[str, ...]) -> bool:
    return any(haystack.startswith(p) for p in prefixes)


def _strip_outer_shell_quotes(s: str) -> str:
    s = s.strip()
    if not s:
        return s

    # Handles: "...", '...', $'...', $"...", and common malformed suffixes like "..."}
    m = re.match(r"^\$?([\"'])([\s\S]*)\1\}?$", s)
    if m:
        return m.group(2)

    # Very common malformed wrapper from serialized tool calls: "...}
    if s[0] in {"'", '"'} and s.endswith("}"):
        return s[1:-1]

    return s


def _unwrap_command(action: str) -> str:
    """Unwrap bash -lc and leading cd/source wrappers."""
    cmd = (action or "").strip()
    for _ in range(6):
        changed = False
        s = cmd.strip()
        s_lower = s.lower()

        if s_lower.startswith("bash -lc"):
            inner = s[8:].strip()  # len("bash -lc") == 8
            cmd = _strip_outer_shell_quotes(inner)
            changed = True

        s = cmd.strip()
        # Keep only command after first && for cd/source wrappers
        if (s.startswith("cd ") or s.startswith("source ")) and "&&" in s:
            cmd = s.split("&&", 1)[1].strip()
            changed = True

        if not changed:
            break

    return cmd.strip()


def _strip_leading_env_and_timeout(cmd: str) -> str:
    """Strip common wrappers (env assignments, timeout, set -e) for intent detection."""
    s = cmd.strip()
    for _ in range(8):
        prev = s
        s = re.sub(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+", "", s)
        s = re.sub(r"^env\s+", "", s)
        s = re.sub(r"^timeout\s+\S+\s+", "", s)
        s = re.sub(r"^set\s+-[A-Za-z]+(?:\s+-[A-Za-z]+)*\s*;\s*", "", s)
        s = re.sub(r"^set\s+-o\s+\S+\s*;\s*", "", s)
        if s == prev:
            break
    return s.strip()


def _is_test_runner_cmd(cmd_lower: str) -> bool:
    return _contains_any(
        cmd_lower,
        (
            "pytest",
            "python -m pytest",
            "go test",
            "npm test",
            "npx jest",
            "mocha",
            "python -m unittest",
            "yarn test",
        ),
    )


def _is_search_cmd(cmd_lower: str) -> bool:
    return _startswith_any(cmd_lower, ("grep", "rg ", "ag ", "find "))


def _is_read_cmd(cmd_lower: str) -> bool:
    return _startswith_any(cmd_lower, ("cat ", "head ", "tail ", "sed -n", "nl ", "awk ", "ls "))


def _is_script_cmd(cmd_lower: str) -> bool:
    return _startswith_any(cmd_lower, ("python ", "python3 ", "node ", "go run "))


def _classify_script_name(script_name: str) -> str:
    if _contains_any(script_name, ("repro", "reproduce", "demo")):
        return "run-repro-script"
    if _contains_any(script_name, ("test_", "verify", "check", "validate", "edge_case")):
        return "run-verify-script"
    if script_name:
        return "run-custom-script"
    return "run-inline-snippet"


def _get_git_subcommand(cmd: str) -> str:
    """Extract git subcommand, skipping flags such as -C <path>."""
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()

    if not parts or parts[0] != "git":
        return ""

    i = 1
    while i < len(parts):
        token = parts[i]
        if token in {"-C", "-c", "--git-dir", "--work-tree"}:
            i += 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        return token.lower()
    return ""


def _extract_script_filename(cmd: str) -> str:
    """Extract named script filename from script-running commands."""
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()

    if len(parts) < 2:
        return ""

    exts = (".py", ".js", ".go", ".ts", ".mjs", ".cjs", ".sh")
    for p in parts[1:]:
        if p.startswith("-"):
            continue
        candidate = p.strip("'\"")
        if candidate.lower().endswith(exts):
            return os.path.basename(candidate).lower()

    # Fallback: pick first token that looks like a path with extension
    m = re.search(r"(?:^|\s)([^\s'\"]+\.(?:py|js|go|ts|mjs|cjs|sh))(?:\s|$)", cmd, re.I)
    if m:
        return os.path.basename(m.group(1)).lower()
    return ""


# ---------------------------------------------------------------------------
# Verify outcome detection
# ---------------------------------------------------------------------------
# Intents whose observation can be parsed for pass/fail outcome.
VERIFY_OUTCOME_INTENTS = {
    "run-test-suite",
    "run-test-specific",
    "run-verify-script",
    "run-custom-script",
    "compile-build",
    "syntax-check",
}

# Source-only edits (excludes test/repro edits, excludes create-file which is
# often throwaway scripts like final_verification.py) — used for work-done.
SOURCE_EDIT_INTENTS = {
    "edit-source",
    "insert-source",
    "apply-patch",
}

# Pre-compiled patterns for test-runner summary parsing.
# Verbose: "======= 5 passed in 0.08s ======="
_RE_PYTEST_SUMMARY = re.compile(
    r"=+\s+(.*?\d+\s+(?:passed|failed|error).*?)\s+in\s+[\d.]+s\s*=+",
)
# Quiet (-q): "5 passed in 0.08s" or "1 failed, 5 passed in 0.08s" (no === border)
_RE_PYTEST_SUMMARY_QUIET = re.compile(
    r"^(\d+\s+(?:passed|failed|error)(?:,\s+\d+\s+\w+)*\s+in\s+[\d.]+s)\s*$",
    re.MULTILINE,
)
_RE_PYTEST_FAIL_IN_SUMMARY = re.compile(r"\d+\s+(?:failed|error)")
_RE_PYTEST_PASSED_IN_SUMMARY = re.compile(r"\d+\s+passed")
_RE_MOCHA_PASSING = re.compile(r"(\d+)\s+passing")
_RE_MOCHA_FAILING = re.compile(r"(\d+)\s+failing")
_RE_GO_RESULT = re.compile(r"^(ok|FAIL)\s+\S+", re.MULTILINE)
_RE_GO_TEST_LINE = re.compile(r"--- (PASS|FAIL):", re.MULTILINE)
_RE_JEST_SUMMARY = re.compile(r"Tests:\s+(.+)")
_RE_COMPILE_ERROR = re.compile(
    r"(?:syntax|compile|build)\s+error|cannot\s+find|undefined:|"
    r"cannot\s+use|undeclared\s+name|expected\s+",
    re.IGNORECASE,
)
_RE_EXIT_NONZERO = re.compile(r"^exit status [1-9]", re.MULTILINE)
_RE_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
_RE_NODE_THROW = re.compile(r"throw\s+\w+|Error:.*\n\s+at\s+")
_RE_CUSTOM_SUMMARY = re.compile(r"(\d+)\s+passed.*?(\d+)\s+failed")


def classify_verify_outcome(action: str, observation: str, base_intent: str) -> str:
    """Classify the outcome of a verify step as 'pass', 'fail', or '' (unknown).

    Only attempts classification for VERIFY_OUTCOME_INTENTS.
    Returns '' when the outcome cannot be unambiguously determined.
    """
    if base_intent not in VERIFY_OUTCOME_INTENTS:
        return ""
    if not observation:
        return ""

    tail = observation[-2000:]

    # --- pytest ---
    # Try verbose (=== bordered) first, then quiet (-q) format.
    m = _RE_PYTEST_SUMMARY.search(tail) or _RE_PYTEST_SUMMARY_QUIET.search(tail)
    if m:
        summary = m.group(1)
        if _RE_PYTEST_FAIL_IN_SUMMARY.search(summary):
            return "fail"
        if _RE_PYTEST_PASSED_IN_SUMMARY.search(summary):
            return "pass"

    # pytest: "no tests ran" or "N error in Xs" (collection/setup errors)
    if re.search(r"no tests ran", tail, re.IGNORECASE):
        return "fail"
    if re.search(r"\d+\s+error\s+in\s+[\d.]+s", tail):
        return "fail"

    # --- mocha ---
    mp = _RE_MOCHA_PASSING.search(tail)
    if mp:
        mf = _RE_MOCHA_FAILING.search(tail)
        if mf and int(mf.group(1)) > 0:
            return "fail"
        return "pass"

    # --- go test ---
    go_results = _RE_GO_RESULT.findall(tail)
    if go_results:
        return "fail" if "FAIL" in go_results else "pass"
    go_lines = _RE_GO_TEST_LINE.findall(tail)
    if go_lines:
        return "fail" if "FAIL" in go_lines else "pass"

    # --- jest ---
    jm = _RE_JEST_SUMMARY.search(tail)
    if jm:
        s = jm.group(1)
        if "failed" in s:
            return "fail"
        if "passed" in s:
            return "pass"

    # --- compile/build errors ---
    if base_intent == "compile-build":
        if _RE_COMPILE_ERROR.search(tail):
            return "fail"
        if _RE_EXIT_NONZERO.search(tail):
            return "fail"
        if "panic:" in tail:
            return "fail"
        # Empty or clean output from go build / go vet = pass,
        # but only when the observation is short (< 200 chars of real content).
        stripped = observation.strip()
        if not stripped or len(stripped) < 200:
            # Check the action includes a known build command
            action_lower = (action or "").lower()
            if _contains_any(action_lower, ("go build", "go vet", "make")):
                return "pass"
        return ""

    # --- syntax-check ---
    if base_intent == "syntax-check":
        # Syntax checks with && echo "✓ ..." — if the echo text appears, it passed.
        if "&&" in (action or "") and "\u2713" in tail:
            return "pass"
        # py_compile with no output = pass
        if "py_compile" in (action or "").lower():
            if not observation.strip():
                return "pass"
            if "Error" in tail or "SyntaxError" in tail:
                return "fail"
            return "pass"
        return ""

    # --- custom verify/test scripts ---
    # Only classify when there's an unambiguous structured summary.
    cm = _RE_CUSTOM_SUMMARY.search(tail)
    if cm:
        return "fail" if int(cm.group(2)) > 0 else "pass"

    # Traceback at the end = fail
    if _RE_TRACEBACK.search(tail[-500:]):
        return "fail"

    # Node.js throw/error at the end
    if _RE_NODE_THROW.search(tail[-500:]):
        return "fail"

    # Non-zero exit
    if _RE_EXIT_NONZERO.search(tail):
        return "fail"

    return ""


def classify_step(action: str, observation: str = "") -> str:
    """Classify one trajectory step into deterministic intent label."""
    action = action or ""
    observation = observation or ""

    action_line = _safe_first_line(action).strip()
    action_line_lower = action_line.lower()

    # Empty
    if not action.strip():
        return "empty"

    # Submit
    if action_line_lower.startswith("submit"):
        return "submit"

    # str_replace_editor view
    if action_line_lower.startswith("str_replace_editor view"):
        target = _extract_path(action_line)
        base = _basename(target)
        base_lower = base.lower()
        obs_lower = observation.lower()

        if "--view_range" in action_line_lower:
            return "read-file-range"

        if "files and directories" in obs_lower:
            return "view-directory"

        if _has_no_extension_excluding_leading_dot(base):
            return "view-directory"

        if TEST_FILE_RE.match(base_lower):
            return "read-test-file"

        if base_lower in CONFIG_FILES:
            return "read-config-file"

        if "too large to display" in obs_lower:
            return "read-file-full(truncated)"

        return "read-file-full"

    # str_replace_editor create
    if action_line_lower.startswith("str_replace_editor create"):
        filename = _basename(_extract_path(action_line)).lower()

        if _contains_any(filename, ("repro", "reproduce")):
            return "create-repro-script"
        if _contains_any(filename, ("test_", "test.py", "test.js", "test.go")):
            return "create-test-script"
        if _contains_any(filename, ("verify", "check", "validate", "edge_case")):
            return "create-verify-script"
        if _contains_any(filename, ("summary", "readme", "changes", "implementation")):
            return "create-documentation"
        return "create-file"

    # str_replace_editor str_replace
    if action_line_lower.startswith("str_replace_editor str_replace"):
        filename = _basename(_extract_path(action_line)).lower()
        if _contains_any(filename, ("test_", "repro", "verify", "check")):
            return "edit-test-or-repro"
        return "edit-source"

    # str_replace_editor insert
    if action_line_lower.startswith("str_replace_editor insert"):
        return "insert-source"

    # str_replace_editor undo_edit
    if action_line_lower.startswith("str_replace_editor undo"):
        return "undo-edit"

    # Everything else: bash/direct commands
    cmd = _unwrap_command(action)
    cmd_for_match = _strip_leading_env_and_timeout(cmd)
    cmd_match_lower = cmd_for_match.lower().strip()
    cmd_head_lower = cmd_match_lower.split("\n", 1)[0].strip()
    obs_lower_500 = observation[:500].lower()

    # Failed shell command variants (classify by intended action)
    if _contains_any(
        obs_lower_500,
        (
            "syntax error",
            "unexpected token",
            "command not found",
            "here-document at line",
            "unexpected `}'",
            "invalid number of lines",
            "invalid option",
            "broken pipe",
        ),
    ):
        if _is_search_cmd(cmd_head_lower):
            return "search-keyword(failed)"
        if _is_test_runner_cmd(cmd_head_lower):
            return "run-test-suite(failed)"
        if _is_script_cmd(cmd_head_lower):
            return "run-script(failed)"
        if _is_read_cmd(cmd_head_lower):
            return "read-via-bash(failed)"
        return "bash-command(failed)"

    # applypatch
    if "applypatch" in cmd_head_lower:
        return "apply-patch"

    # Test suite
    if _is_test_runner_cmd(cmd_head_lower):
        if "::" in cmd_head_lower or " -k " in cmd_head_lower:
            return "run-test-specific"
        return "run-test-suite"

    # Syntax / compile check
    if _contains_any(cmd_head_lower, ("py_compile", "compileall", "node -c ")):
        return "syntax-check"

    if _startswith_any(cmd_head_lower, ("go build", "go vet", "make ")):
        return "compile-build"

    if _startswith_any(cmd_head_lower, ("npx tsc", "tsc ", "./node_modules/.bin/tsc")):
        return "compile-build"

    if _contains_any(
        cmd_head_lower,
        ("npm run build", "yarn build", "check-types", "lint:types", "npm run types"),
    ):
        return "compile-build"

    # Search commands
    if _startswith_any(cmd_head_lower, ("grep", "rg ", "ag ")):
        return "search-keyword"

    if _startswith_any(cmd_head_lower, ("find ",)):
        if _contains_any(cmd_head_lower, ("grep", "xargs")):
            return "search-files-by-content"
        return "search-files-by-name"

    # Read commands
    if _startswith_any(cmd_head_lower, ("cat ", "head ", "tail ", "sed -n", "nl ", "awk ")):
        return "read-via-bash"

    # List / navigate
    if _startswith_any(cmd_head_lower, ("ls", "tree ", "pwd")):
        return "list-directory"

    # Run python/node/go script
    if _startswith_any(cmd_head_lower, ("python ", "python3 ", "node ", "go run ")):
        # Inline snippets
        if "-c " in cmd_for_match or "- <<" in cmd_for_match or "-e " in cmd_for_match:
            if "node" in cmd_head_lower and "-c " in cmd_for_match.lower():
                return "syntax-check"
            return "run-inline-snippet"

        return _classify_script_name(_extract_script_filename(cmd_for_match))

    # Shell-invoked scripts, e.g. ./verify.sh, sh repro.sh
    if _startswith_any(cmd_head_lower, ("./", "sh ", "bash ", "zsh ")):
        return _classify_script_name(_extract_script_filename(cmd_for_match))

    # Git (supports git -C /path ...)
    git_sub = _get_git_subcommand(cmd_for_match)
    if git_sub == "diff":
        return "git-diff"
    if git_sub in {"status", "show", "log"}:
        return "git-status-log"
    if git_sub == "stash":
        return "git-stash"

    # File management
    if _startswith_any(cmd_head_lower, ("rm ", "mv ", "cp ", "chmod ")):
        return "file-cleanup"

    # Install / deps
    if _contains_any(cmd_head_lower, ("pip install", "pip list", "npm install", "go get", "apt ")):
        return "install-deps"

    # Service management
    if _contains_any(cmd_head_lower, ("redis-server", "redis-cli", "mongod", "sleep ")):
        return "start-service"

    # Check tool existence
    if _startswith_any(cmd_head_lower, ("which ", "type ")):
        return "check-tool-exists"

    # Inspect metadata
    if _startswith_any(cmd_head_lower, ("wc ", "file ", "stat ")):
        return "inspect-file-metadata"

    # Echo
    if _startswith_any(cmd_head_lower, ("echo ", "printf ")):
        return "echo"

    return "bash-other"


def classify_trajectory(trajectory: list[dict]) -> list[str]:
    return [classify_step(step.get("action", ""), step.get("observation", "")) for step in trajectory]


def classify_trajectory_counts(trajectory: list[dict]) -> Counter:
    c = Counter()
    for step in trajectory:
        c[classify_step(step.get("action", ""), step.get("observation", ""))] += 1
    return c


def to_hierarchical_intent(base_intent: str) -> str:
    high = INTENT_TO_HIGH_LEVEL.get(base_intent, "other")
    return f"{high}.{base_intent}"


def classify_hierarchical_layer(base_intents: list[str]) -> list[str]:
    return [to_hierarchical_intent(i) for i in base_intents]


def classify_hierarchical_counts(base_intents: list[str]) -> Counter:
    return Counter(classify_hierarchical_layer(base_intents))


def classify_verify_outcomes(trajectory: list[dict], base_intents: list[str]) -> list[str]:
    """Per-step verify outcome: 'pass', 'fail', or '' (not a verify-run step or unknown)."""
    outcomes: list[str] = []
    for step, base_intent in zip(trajectory, base_intents):
        action = step.get("action", "") or ""
        observation = step.get("observation", "") or ""
        outcomes.append(classify_verify_outcome(action, observation, base_intent))
    return outcomes


def classify_sequence_layer(
    trajectory: list[dict],
    base_intents: list[str],
    verify_outcomes: list[str] | None = None,
) -> list[str]:
    """Second-layer deterministic sequence intents derived from base intents + nearby history.

    When *verify_outcomes* is provided (list of 'pass'/'fail'/'' per step),
    two additional retrospective markers are emitted:

    - ``seq-first-all-pass``: the first verify-pass that occurs after the last
      source edit (i.e. the first moment the tests confirm the finished
      implementation works).
    - ``seq-work-done``: same step as first-all-pass when the trajectory has no
      further source edits after it (retrospective — requires knowing the full
      trajectory).  Differs from first-all-pass only conceptually: first-all-pass
      marks "tests pass after last edit", work-done marks "this is the point
      after which nothing productive happens".
    """
    n = len(base_intents)
    seq_labels: list[str] = [""] * n

    # --- Pre-scan: find last source edit index (for first-all-pass / work-done) ---
    last_source_edit_idx = -1
    if verify_outcomes is not None:
        for i in range(n - 1, -1, -1):
            if base_intents[i] in SOURCE_EDIT_INTENTS:
                last_source_edit_idx = i
                break

    first_all_pass_emitted = False

    # --- Forward pass ---
    seen_verify = False
    seen_repro = False
    edited_since_verify = False
    edited_since_repro = False
    prev_verify_sig = ""
    prev_repro_sig = ""
    prev_base = ""
    edited_paths: set[str] = set()

    for i, (step, base_intent) in enumerate(zip(trajectory, base_intents)):
        action = step.get("action", "") or ""
        action_line = _safe_first_line(action)
        signature = _command_signature(action)

        seq = "seq-none"

        if base_intent in SEQUENCE_VERIFY_INTENTS:
            if seen_verify and (not edited_since_verify) and signature and signature == prev_verify_sig:
                seq = "seq-verify-rerun-same-command"
            elif edited_since_verify:
                seq = "seq-verify-after-edit"
            elif seen_verify and not edited_since_verify:
                seq = "seq-verify-rerun-no-edit"

            seen_verify = True
            edited_since_verify = False
            prev_verify_sig = signature

            # --- first-all-pass / work-done ---
            if (
                verify_outcomes is not None
                and not first_all_pass_emitted
                and verify_outcomes[i] == "pass"
                and i > last_source_edit_idx
                and last_source_edit_idx >= 0
            ):
                seq = "seq-first-all-pass"
                first_all_pass_emitted = True

        elif base_intent in SEQUENCE_REPRO_INTENTS:
            if seen_repro and (not edited_since_repro) and signature and signature == prev_repro_sig:
                seq = "seq-repro-rerun-same-command"
            elif edited_since_repro:
                seq = "seq-repro-after-edit"
            elif seen_repro and not edited_since_repro:
                seq = "seq-repro-rerun-no-edit"

            seen_repro = True
            edited_since_repro = False
            prev_repro_sig = signature

        elif base_intent in SEQUENCE_EDIT_INTENTS:
            if prev_base in SEQUENCE_FAILED_VERIFY_INTENTS:
                seq = "seq-edit-after-failed-verify"

            edited_since_verify = True
            edited_since_repro = True

            path = _extract_path(action_line)
            if path:
                edited_paths.add(path)

        elif base_intent in SEQUENCE_READ_INTENTS:
            if prev_base in SEQUENCE_FAILED_VERIFY_INTENTS:
                seq = "seq-diagnose-read-after-failed-verify"
            else:
                path = _extract_path(action_line)
                if path and path in edited_paths:
                    seq = "seq-reread-edited-file"

        elif base_intent in SEQUENCE_SEARCH_INTENTS:
            if prev_base in SEQUENCE_FAILED_VERIFY_INTENTS:
                seq = "seq-diagnose-search-after-failed-verify"

        elif base_intent == "submit" and seen_verify:
            seq = "seq-submit-after-verify"

        seq_labels[i] = seq
        prev_base = base_intent

    # --- Retrospective: work-done ---
    # work-done = first-all-pass step, but only if no source edits follow it.
    # Since first-all-pass is already defined as "first verify-pass after last
    # source edit", work-done coincides with it by construction.  We emit it as
    # a separate label on the same step so downstream can use either name.
    if first_all_pass_emitted:
        for i in range(n):
            if seq_labels[i] == "seq-first-all-pass":
                # Verify no source edits after this point (should hold by
                # construction, but be explicit).
                has_later_source_edit = any(
                    base_intents[j] in SOURCE_EDIT_INTENTS for j in range(i + 1, n)
                )
                if not has_later_source_edit:
                    seq_labels[i] = "seq-work-done"
                break

    return seq_labels


def classify_sequence_counts(trajectory: list[dict], base_intents: list[str]) -> Counter:
    outcomes = classify_verify_outcomes(trajectory, base_intents)
    return Counter(classify_sequence_layer(trajectory, base_intents, verify_outcomes=outcomes))


def classify_file(traj_path: str) -> tuple[list[str], dict]:
    data = _load_json(traj_path)
    intents = classify_trajectory(data.get("trajectory", []))
    return intents, data


def summarize_file(traj_path: str) -> Counter:
    data = _load_json(traj_path)
    return classify_trajectory_counts(data.get("trajectory", []))


def _collect_traj_files(paths: list[str]) -> list[Path]:
    traj_files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".traj":
            traj_files.append(path)
        elif path.is_dir():
            traj_files.extend(sorted(path.glob("*/*.traj")))
    return sorted(traj_files)


def _classify_file_summary(task: tuple[str, bool, bool]) -> tuple[str, int, str, dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    traj_path, include_sequence, include_hierarchical = task
    data = _load_json(traj_path)
    trajectory = data.get("trajectory", [])
    base_intents = classify_trajectory(trajectory)
    base_counts = Counter(base_intents)

    sequence_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    if include_sequence:
        outcomes = classify_verify_outcomes(trajectory, base_intents)
        sequence_counts = Counter(classify_sequence_layer(trajectory, base_intents, verify_outcomes=outcomes))
        outcome_counts = Counter(oc for oc in outcomes if oc)

    hierarchical_counts: Counter = Counter()
    if include_hierarchical:
        hierarchical_counts = classify_hierarchical_counts(base_intents)

    exit_status = data.get("info", {}).get("exit_status", "")
    return (
        traj_path,
        len(trajectory),
        exit_status,
        dict(base_counts),
        dict(sequence_counts),
        dict(hierarchical_counts),
        dict(outcome_counts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify SWE-Agent trajectory steps into deterministic intents")
    parser.add_argument("paths", nargs="+", help=".traj files or directories containing .traj files")
    parser.add_argument("--show", type=int, default=0, help="Show first N step classifications per file (sequential mode)")
    parser.add_argument("--sequence-layer", action="store_true", help="Compute sequence-level intent labels on top of base intents")
    parser.add_argument("--hierarchical-layer", action="store_true", help="Compute high-level dot-notation intents (e.g. read-code.read-file-full)")
    default_workers = min(8, os.cpu_count() or 1)
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Parallel workers for per-file classification (default: {default_workers})")
    parser.add_argument("--quiet", action="store_true", help="Only print aggregate counts and timing")
    parser.add_argument("--json-output", default="", help="Write aggregate intent counts JSON to path")
    args = parser.parse_args()

    traj_files = _collect_traj_files(args.paths)
    if not traj_files:
        raise SystemExit("No .traj files found")

    if args.show > 0 and args.workers > 1:
        print("note: --show requested; forcing --workers=1 for deterministic step display")
        args.workers = 1

    total = Counter()
    total_sequence = Counter()
    total_hierarchical = Counter()
    total_outcomes = Counter()
    t0 = time.perf_counter()

    if args.workers <= 1:
        for tf in traj_files:
            data = _load_json(str(tf))
            trajectory = data.get("trajectory", [])
            intents = classify_trajectory(trajectory)
            c = Counter(intents)
            total.update(c)

            outcomes: list[str] = []
            seq_labels: list[str] = []
            if args.sequence_layer:
                outcomes = classify_verify_outcomes(trajectory, intents)
                seq_labels = classify_sequence_layer(trajectory, intents, verify_outcomes=outcomes)
                total_sequence.update(seq_labels)
                total_outcomes.update(oc for oc in outcomes if oc)

            hier_labels: list[str] = []
            if args.hierarchical_layer:
                hier_labels = classify_hierarchical_layer(intents)
                total_hierarchical.update(hier_labels)

            if not args.quiet:
                print(f"\n{tf}")
                print(f"steps={len(intents)}  exit={data.get('info', {}).get('exit_status', '')}")
                print("top_intents:", dict(c.most_common(10)))
                if args.hierarchical_layer:
                    print("top_hierarchical:", dict(Counter(hier_labels).most_common(10)))
                if args.sequence_layer:
                    seq_counts = Counter(seq_labels)
                    print("top_sequence:", dict(seq_counts.most_common(10)))

            if args.show > 0:
                for i, (step, intent) in enumerate(zip(trajectory, intents), start=1):
                    if i > args.show:
                        break
                    action_line = _safe_first_line(step.get("action", ""))
                    outcome_tag = ""
                    if outcomes:
                        oc = outcomes[i - 1]
                        if oc:
                            outcome_tag = f"[{oc}]"
                    if args.sequence_layer and args.hierarchical_layer:
                        seq = seq_labels[i - 1]
                        hier = hier_labels[i - 1]
                        print(f"  {i:>3}  {intent:<28} {outcome_tag:<6} {hier:<44} {seq:<34} {action_line[:80]}")
                    elif args.sequence_layer:
                        seq = seq_labels[i - 1]
                        print(f"  {i:>3}  {intent:<28} {outcome_tag:<6} {seq:<36} {action_line[:100]}")
                    elif args.hierarchical_layer:
                        hier = hier_labels[i - 1]
                        print(f"  {i:>3}  {intent:<28} {hier:<44} {action_line[:110]}")
                    else:
                        print(f"  {i:>3}  {intent:<28} {action_line[:140]}")
    else:
        max_workers = min(max(args.workers, 1), os.cpu_count() or 1)
        tasks = [(str(p), bool(args.sequence_layer), bool(args.hierarchical_layer)) for p in traj_files]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for traj_path, steps, exit_status, counts, sequence_counts, hierarchical_counts, outcome_counts in ex.map(_classify_file_summary, tasks, chunksize=16):
                c = Counter(counts)
                total.update(c)
                if args.sequence_layer:
                    total_sequence.update(sequence_counts)
                    total_outcomes.update(outcome_counts)
                if args.hierarchical_layer:
                    total_hierarchical.update(hierarchical_counts)
                if not args.quiet:
                    print(f"\n{traj_path}")
                    print(f"steps={steps}  exit={exit_status}")
                    print("top_intents:", dict(c.most_common(10)))
                    if args.hierarchical_layer:
                        print("top_hierarchical:", dict(Counter(hierarchical_counts).most_common(10)))
                    if args.sequence_layer:
                        print("top_sequence:", dict(Counter(sequence_counts).most_common(10)))

    elapsed = time.perf_counter() - t0
    total_steps = sum(total.values())

    print("\n=== aggregate ===")
    print(dict(total.most_common()))

    high_level_counts = Counter()
    if args.hierarchical_layer:
        print("\n=== hierarchical aggregate ===")
        print(dict(total_hierarchical.most_common()))
        for label, n in total_hierarchical.items():
            high_level_counts[label.split(".", 1)[0]] += n
        print("\n=== high-level aggregate ===")
        print(dict(high_level_counts.most_common()))

    if args.sequence_layer:
        print("\n=== sequence aggregate ===")
        print(dict(total_sequence.most_common()))
        if total_outcomes:
            print("\n=== verify outcomes ===")
            print(dict(total_outcomes.most_common()))
    print(f"files={len(traj_files)} steps={total_steps} elapsed_sec={elapsed:.3f}")

    if args.json_output:
        payload = {
            "files": len(traj_files),
            "steps": total_steps,
            "elapsed_sec": elapsed,
            "intents": dict(total.most_common()),
        }
        if args.hierarchical_layer:
            payload["hierarchical_intents"] = dict(total_hierarchical.most_common())
            payload["high_level_categories"] = dict(high_level_counts.most_common())
        if args.sequence_layer:
            payload["sequence_intents"] = dict(total_sequence.most_common())
            if total_outcomes:
                payload["verify_outcomes"] = dict(total_outcomes.most_common())
        with open(args.json_output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
