# Intent Classification Rules

Classify each step in a SWE-Agent trajectory by what the command **literally is**,
derived from the action string and filename alone. No positional context (before/after
first edit) is used.

## Categories (35 intents)

### Reading code
- `read-file-full` — view an entire source file
- `read-file-range` — view a specific line range of a file
- `read-file-full(truncated)` — view a file that was too large, got abbreviated
- `read-test-file` — view a test file (filename matches test_*, _test.*, conftest)
- `read-config-file` — view a config file (package.json, pytest.ini, setup.cfg, go.mod, Makefile, config.json)
- `read-via-bash` — read file contents using cat, head, tail, sed -n, nl, awk

### Searching / navigating
- `view-directory` — view a directory listing via str_replace_editor
- `list-directory` — ls, tree, pwd
- `search-keyword` — grep, rg, ag for a pattern
- `search-files-by-name` — find ... -name (locating files by name/path)
- `search-files-by-content` — find ... -exec grep / find | xargs grep (locating files by what's inside them)
- `inspect-file-metadata` — wc, file, stat

### Reproducing
- `create-repro-script` — create a file named repro*, reproduce*, demo*
- `run-repro-script` — run a file named repro*, reproduce*, demo*
- `run-inline-snippet` — python -c, python - <<, python3 -c, node -e (inline code)

### Implementing
- `edit-source` — str_replace on a non-test, non-repro source file
- `insert-source` — str_replace_editor insert on a source file
- `apply-patch` — applypatch command (GPT-specific alternative to str_replace)
- `create-file` — create a file that doesn't match repro/test/verify/doc patterns

### Verifying
- `run-test-suite` — pytest, go test, npm test, npx jest, mocha (broad)
- `run-test-specific` — pytest with -k or :: (targeting specific tests)
- `create-test-script` — create a file named test_*, *test.py, *test.js, *test.go
- `run-verify-script` — run a file named test_*, verify*, check*, validate*, edge_case*
- `create-verify-script` — create a file named verify*, check*, validate*
- `edit-test-or-repro` — str_replace on a test or repro file
- `run-custom-script` — run a named script that doesn't match repro/test/verify patterns
- `syntax-check` — py_compile, compileall, node -c (syntax validation)
- `compile-build` — go build, go vet, make

### Git
- `git-diff` — git diff
- `git-inspect` — git status, git show, git log
- `git-stash` — git stash

### Infrastructure
- `file-cleanup` — rm, mv, cp, chmod
- `create-documentation` — create a file named *summary*, *readme*, *changes*, *implementation*
- `start-service` — redis-server, redis-cli, sleep (waiting for services)
- `install-deps` — pip install, npm install, go get, apt
- `check-tool-exists` — which, type

### Failed
- `search-keyword(failed)` — grep/find that hit shell errors
- `read-via-bash(failed)` — cat/head/sed that hit shell errors
- `run-script(failed)` — python/node run that hit shell errors
- `run-test-suite(failed)` — pytest/test that hit shell errors
- `bash-command(failed)` — other bash that hit shell errors

### Other
- `submit` — submit the patch
- `empty` — empty action string (rate limit, context window exit)
- `echo` — echo, printf
- `bash-other` — unclassified bash command
- `undo-edit` — str_replace_editor undo_edit


## Classification rules (pseudocode)

```
function classify_step(action, observation):

    action_line = first_line(action).lowercase()

    # ── Empty ──
    if action is blank:
        return "empty"

    # ── Submit ──
    if action_line starts with "submit":
        return "submit"

    # ── str_replace_editor view ──
    if action_line starts with "str_replace_editor view":
        target = extract_path(action_line)        # third token
        basename = last_component(target)

        if "--view_range" in action_line:
            return "read-file-range"

        if observation contains "files and directories":
            return "view-directory"

        if basename has no "." (excluding leading dot):
            return "view-directory"              # /app, /app/src/database

        if basename matches test_*, _test.*, conftest*:
            return "read-test-file"

        if basename in {package.json, pytest.ini, setup.cfg, setup.py,
                        go.mod, Makefile, config.json}:
            return "read-config-file"

        if observation contains "too large to display":
            return "read-file-full(truncated)"

        return "read-file-full"

    # ── str_replace_editor create ──
    if action_line starts with "str_replace_editor create":
        filename = last_component(extract_path(action_line)).lowercase()

        if filename contains {repro, reproduce}:
            return "create-repro-script"
        if filename contains {test_, test.py, test.js, test.go}:
            return "create-test-script"
        if filename contains {verify, check, validate, edge_case}:
            return "create-verify-script"
        if filename contains {summary, readme, changes, implementation}:
            return "create-documentation"
        return "create-file"

    # ── str_replace_editor str_replace ──
    if action_line starts with "str_replace_editor str_replace":
        filename = last_component(extract_path(action_line)).lowercase()

        if filename contains {test_, repro, verify, check}:
            return "edit-test-or-repro"
        return "edit-source"

    # ── str_replace_editor insert ──
    if action_line starts with "str_replace_editor insert":
        return "insert-source"

    # ── str_replace_editor undo_edit ──
    if action_line starts with "str_replace_editor undo":
        return "undo-edit"

    # ── Everything else: bash / direct commands ──

    # Unwrap bash -lc "..." wrapper
    cmd = action
    if cmd starts with "bash -lc":
        cmd = strip_wrapper(cmd)            # remove bash -lc and outer quotes
    if cmd starts with "cd ... &&":
        cmd = part_after_first("&&", cmd)   # keep only the command after cd
    if cmd starts with "source ... &&":
        cmd = part_after_first("&&", cmd)   # keep only the command after activate

    cmd_lower = cmd.lowercase().strip()

    # ── Check observation for shell-level errors ──
    if observation (first 500 chars) contains any of:
        "syntax error", "unexpected token", "command not found",
        "here-document at line", "unexpected `}'",
        "invalid number of lines", "invalid option", "broken pipe"
    then:
        # Classify by what it was TRYING to do
        if cmd_lower contains {grep, find}:    return "search-keyword(failed)"
        if cmd_lower contains {test, pytest}:  return "run-test-suite(failed)"
        if cmd_lower contains {python, node}:  return "run-script(failed)"
        if cmd_lower contains {cat, head, tail, sed, ls}:
                                               return "read-via-bash(failed)"
        return "bash-command(failed)"

    # ── applypatch ──
    if cmd_lower contains "applypatch":
        return "apply-patch"

    # ── Test suite ──
    if cmd_lower contains {pytest, python -m pytest, go test,
                           npm test, npx jest, mocha, python -m unittest}:
        if cmd_lower contains "::" or cmd_lower contains " -k ":
            return "run-test-specific"
        return "run-test-suite"

    # ── Syntax / compile check ──
    if cmd_lower contains {py_compile, compileall, node -c }:
        return "syntax-check"
    if cmd_lower starts with {go build, go vet, make }:
        return "compile-build"

    # ── Search commands ──
    if cmd_lower starts with {grep, rg , ag }:
        return "search-keyword"
    if cmd_lower starts with "find ":
        if cmd_lower contains {grep, xargs}:
            return "search-files-by-content"
        return "search-files-by-name"

    # ── Read commands ──
    if cmd_lower starts with {cat , head , tail , sed -n, nl , awk }:
        return "read-via-bash"

    # ── List / navigate ──
    if cmd_lower starts with {ls, tree , pwd}:
        return "list-directory"

    # ── Run python/node script ──
    if cmd_lower starts with {python , python3 , node }:

        # Inline snippet
        if cmd contains "-c " or cmd contains "- <<" or cmd contains "-e ":
            if "node" in cmd and "-c " in cmd:     # node -c is syntax check
                return "syntax-check"
            return "run-inline-snippet"

        # Named script — extract filename
        script_name = extract_script_filename(cmd).lowercase()

        if script_name contains {repro, reproduce}:
            return "run-repro-script"
        if script_name contains {test_, verify, check, validate, edge_case}:
            return "run-verify-script"
        if script_name is not empty:
            return "run-custom-script"

        return "run-inline-snippet"           # fallback for python - etc.

    # ── Git ──
    if cmd_lower starts with "git diff":       return "git-diff"
    if cmd_lower starts with {git status, git show, git log}:
                                               return "git-inspect"
    if cmd_lower starts with "git stash":      return "git-stash"

    # ── File management ──
    if cmd_lower starts with {rm , mv , cp , chmod }:
        return "file-cleanup"

    # ── Install / deps ──
    if cmd_lower contains {pip install, pip list, npm install, go get, apt }:
        return "install-deps"

    # ── Service management ──
    if cmd_lower contains {redis-server, redis-cli, mongod, sleep }:
        return "start-service"

    # ── Check tool existence ──
    if cmd_lower starts with {which , type }:
        return "check-tool-exists"

    # ── Inspect metadata ──
    if cmd_lower starts with {wc , file , stat }:
        return "inspect-file-metadata"

    # ── Echo ──
    if cmd_lower starts with {echo , printf }:
        return "echo"

    # ── Fallback ──
    return "bash-other"
```

## Notes

- No positional context is used. Every classification is derived from the command
  string and filename alone.
- The `(failed)` variants classify by intended action, not by outcome. A failed grep
  is still a search attempt.
- `run-inline-snippet` captures python -c, python - <<, node -e. These are ambiguous
  in purpose (could be exploration, reproduction, or verification) but structurally
  distinct from running a named script.
- `run-custom-script` is the catch-all for named scripts that don't match
  repro/test/verify filename patterns.
- `create-file` is the catch-all for created files that don't match any pattern.
  These are often new source files being added as part of the fix.
- `bash-other` should be <2% of steps. If it's higher, inspect examples and add rules.
