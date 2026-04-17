"""
Layer 1: Single source of truth for the entire taxonomy.

All model definitions, intent categories, phase groupings, display names,
and descriptions live here. Nothing else in the codebase should hardcode
these values.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure scripts/ is importable so we can pull INTENT_TO_HIGH_LEVEL from
# the authoritative classify_intent module.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import classify_intent as _ci  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry -- add new models here and everything else adapts.
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "claude45": {"label": "Sonnet 4.5", "color": "#b8785e"},
    "gpt5": {"label": "GPT-5", "color": "#6a8da8"},
    "gemini25pro": {"label": "Gemini 2.5 Pro", "color": "#5a9a6a"},
    "glm45": {"label": "GLM 4.5", "color": "#8a6a9a"},
}

# ---------------------------------------------------------------------------
# Level 1: Base intents (~46 labels) -- canonical source is classify_intent.py
# ---------------------------------------------------------------------------
INTENT_TO_HIGH_LEVEL: dict[str, str] = dict(_ci.INTENT_TO_HIGH_LEVEL)

# All known base intents, derived from the mapping above.
BASE_INTENTS: list[str] = sorted(INTENT_TO_HIGH_LEVEL.keys())

# ---------------------------------------------------------------------------
# Level 2: High-level categories (9)
# ---------------------------------------------------------------------------
HIGH_LEVEL_CATEGORIES: dict[str, dict] = {
    "read":         {"letter": "R", "color": "#5a7d9a"},
    "search":       {"letter": "S", "color": "#5a7d9a"},
    "reproduce":    {"letter": "P", "color": "#b0956a"},
    "edit":         {"letter": "E", "color": "#4a8a5a"},
    "verify":       {"letter": "V", "color": "#b56a50"},
    "git":          {"letter": "G", "color": "#3a8a8a"},
    "housekeeping": {"letter": "H", "color": "#3a8a8a"},
    "failed":       {"letter": "X", "color": "#a05050"},
    "other":        {"letter": "O", "color": "#888"},
}

# Convenience lookups derived from the above.
HIGH_LEVEL_LETTER: dict[str, str] = {
    k: v["letter"] for k, v in HIGH_LEVEL_CATEGORIES.items()
}
LETTER_TO_NAME: dict[str, str] = {v: k for k, v in HIGH_LEVEL_LETTER.items()}
HIGH_LEVEL_COLORS: dict[str, str] = {
    k: v["color"] for k, v in HIGH_LEVEL_CATEGORIES.items()
}
LETTER_COLORS: dict[str, str] = {
    HIGH_LEVEL_LETTER[k]: v for k, v in HIGH_LEVEL_COLORS.items()
}

# Canonical ordering of letters (for transition matrices, etc.)
ORDERED_LETTERS: list[str] = ["R", "S", "P", "E", "V", "G", "H", "X", "O"]

# ---------------------------------------------------------------------------
# Level 3: Phases (5 groupings of high-level categories)
# ---------------------------------------------------------------------------
PHASES: dict[str, dict] = {
    "understand": {"categories": ["read", "search"],       "color": "#5a7d9a"},
    "reproduce":  {"categories": ["reproduce"],             "color": "#b0956a"},
    "edit":       {"categories": ["edit"],                  "color": "#4a8a5a"},
    "verify":     {"categories": ["verify"],                "color": "#b56a50"},
    "cleanup":    {"categories": ["git", "housekeeping"],   "color": "#3a8a8a"},
}

# ---------------------------------------------------------------------------
# Level 4: Display names and descriptions for every base intent
# ---------------------------------------------------------------------------
INTENT_DISPLAY_NAMES: dict[str, str] = {
    # read
    "read-file-full": "view file",
    "read-file-range": "view lines (range)",
    "read-file-full(truncated)": "view file (truncated)",
    "read-test-file": "view test file",
    "read-config-file": "view config file",
    "read-via-bash": "cat / head / tail",
    # search
    "view-directory": "view directory",
    "list-directory": "ls / tree",
    "search-keyword": "grep / ripgrep",
    "search-files-by-name": "find by filename",
    "search-files-by-content": "find | grep",
    "inspect-file-metadata": "wc / stat",
    # reproduce
    "create-repro-script": "write repro script",
    "run-repro-script": "run repro script",
    "run-inline-snippet": "python -c / node -e",
    # inline snippet sub-intents
    "run-inline-verify": "inline verify snippet",
    "read-via-inline-script": "inline read snippet",
    "edit-via-inline-script": "inline edit snippet",
    "create-file-via-inline-script": "inline create-file snippet",
    "check-version": "check version",
    # edit
    "edit-source": "edit source",
    "insert-source": "insert into source",
    "apply-patch": "apply patch",
    "create-file": "create file",
    # verify
    "run-test-suite": "pytest / go test (broad)",
    "run-test-specific": "pytest -k / :: (targeted)",
    "create-test-script": "write test file",
    "run-verify-script": "run verify script",
    "create-verify-script": "write verify script",
    "edit-test-or-repro": "edit test / repro",
    "run-custom-script": "run custom script",
    "syntax-check": "syntax check",
    "compile-build": "go build / make / tsc",
    # git
    "git-diff": "git diff",
    "git-status-log": "git status / log / show",
    "git-stash": "git stash",
    # housekeeping
    "file-cleanup": "rm / mv / cp",
    "create-documentation": "write docs file",
    "start-service": "start service (redis, etc.)",
    "install-deps": "pip install / npm install",
    "check-tool-exists": "which / type",
    # failed
    "search-keyword(failed)": "grep/find (shell error)",
    "read-via-bash(failed)": "cat/head/sed (shell error)",
    "run-script(failed)": "python/node (shell error)",
    "run-test-suite(failed)": "pytest/test (shell error)",
    "bash-command(failed)": "bash (shell error)",
    # other
    "submit": "submit patch",
    "empty": "empty action (timeout)",
    "echo": "echo / printf",
    "bash-other": "other shell command",
    "undo-edit": "undo edit",
}

INTENT_DESCRIPTIONS: dict[str, str] = {
    "read-file-full": "view an entire source file via str_replace_editor",
    "read-file-range": "view a specific line range (--view_range)",
    "read-file-full(truncated)": "view a file that was too large, got abbreviated",
    "read-test-file": "view a test file (test_*, _test.*, conftest)",
    "read-config-file": "view package.json, pytest.ini, setup.cfg, go.mod, Makefile, etc.",
    "read-via-bash": "cat, head, tail, sed -n, nl, awk",
    "view-directory": "view a directory listing via str_replace_editor",
    "list-directory": "ls, tree, pwd",
    "search-keyword": "grep, rg, ag for a pattern",
    "search-files-by-name": "find ... -name (locating files by name/path)",
    "search-files-by-content": "find ... -exec grep / find | xargs grep",
    "inspect-file-metadata": "wc, file, stat",
    "create-repro-script": "create a file named repro*, reproduce*, demo*",
    "run-repro-script": "run a file named repro*, reproduce*, demo*",
    "run-inline-snippet": "python -c, python - <<, python3 -c, node -e",
    "run-inline-verify": "inline snippet that imports project code or runs assertions",
    "read-via-inline-script": "inline snippet that reads a file and prints content",
    "edit-via-inline-script": "inline snippet that reads, modifies, and writes a file",
    "create-file-via-inline-script": "inline snippet that writes a file without reading first",
    "check-version": "inline snippet that checks python/node version",
    "edit-source": "str_replace on a non-test, non-repro source file",
    "insert-source": "str_replace_editor insert on a source file",
    "apply-patch": "applypatch command (GPT-specific alternative to str_replace)",
    "create-file": "create a file that doesn't match repro/test/verify/doc patterns",
    "run-test-suite": "pytest, go test, npm test, npx jest, mocha (broad)",
    "run-test-specific": "pytest with -k or :: (targeting specific tests)",
    "create-test-script": "create a file named test_*, *test.py, *test.js, *test.go",
    "run-verify-script": "run a file named test_*, verify*, check*, validate*, edge_case*",
    "create-verify-script": "create a file named verify*, check*, validate*",
    "edit-test-or-repro": "str_replace on a test or repro file",
    "run-custom-script": "run a named script that doesn't match repro/test/verify patterns",
    "syntax-check": "py_compile, compileall, node -c",
    "compile-build": "go build, go vet, make, npx tsc, tsc",
    "git-diff": "git diff",
    "git-status-log": "git status, git show, git log",
    "git-stash": "git stash",
    "file-cleanup": "rm, mv, cp, chmod",
    "create-documentation": "create a file named summary, readme, changes, implementation",
    "start-service": "redis-server, redis-cli, mongod, sleep",
    "install-deps": "pip install, pip list, npm install, go get, apt",
    "check-tool-exists": "which, type",
    "search-keyword(failed)": "grep/find that hit shell errors",
    "read-via-bash(failed)": "cat/head/sed that hit shell errors",
    "run-script(failed)": "python/node run that hit shell errors",
    "run-test-suite(failed)": "pytest/test that hit shell errors",
    "bash-command(failed)": "other bash that hit shell errors",
    "submit": "submit the patch",
    "empty": "empty action string (rate limit, context window exit)",
    "echo": "echo, printf",
    "bash-other": "unclassified bash command",
    "undo-edit": "str_replace_editor undo_edit",
}
