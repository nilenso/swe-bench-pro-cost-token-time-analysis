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
- `run-inline-snippet` — python -c, python - <<, node -e (residual: inline code that
  doesn't match any sub-classification below)

### Inline snippet sub-classification (applied to python -c / python - << / node -e)
- `run-inline-verify` — imports project code, runs assertions, or prints logic checks
- `read-via-inline-script` — reads file content, prints/inspects it
- `edit-via-inline-script` — reads file, modifies with .replace()/re.sub(), writes back
- `create-file-via-inline-script` — writes a new file without reading first
- `check-version` — python -V, node -v, sys.version

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

        # Inline snippet — sub-classified by code structure
        if cmd contains "-c " or cmd contains "- <<" or cmd contains "-e ":
            if "node" in cmd and "-c " in cmd:     # node -c is syntax check
                return "syntax-check"
            return classify_inline_snippet(cmd)    # see below

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

### Inline snippet sub-classification (pseudocode)

```
function classify_inline_snippet(cmd):
    # Detect file I/O (python and node patterns)
    has_write = cmd matches (write_text | .write( | open(...'w') |
                             writeFileSync | writeFile()
    has_read  = cmd matches (read_text | .read( | open(...'r') |
                             readFileSync | readFile()
    has_modify = cmd matches (.replace( | re.sub()

    # File editing: reads + modifies + writes
    if has_write and (has_read or has_modify):
        return "edit-via-inline-script"
    if has_write and not has_read:
        return "create-file-via-inline-script"

    # Import detection (python + node)
    has_import = cmd matches (^from|^import | require()
    has_print  = cmd matches (print( | console.log)
    has_assert = cmd matches (assert | assertEqual | expect( | .toBe()

    if has_import and has_assert:  return "run-inline-verify"
    if has_read and not has_write: return "read-via-inline-script"
    if has_import and has_print:   return "run-inline-verify"
    if has_import:                 return "run-inline-verify"    # smoke test
    if cmd matches (--version | -V | sys.version | node -v):
                                   return "check-version"
    if has_print:                  return "run-inline-verify"    # logic check

    return "run-inline-snippet"    # residual
```

## Notes

- No positional context is used. Every classification is derived from the command
  string and filename alone.
- The `(failed)` variants classify by intended action, not by outcome. A failed grep
  is still a search attempt.
- `run-inline-snippet` was originally a catch-all for python -c, python - <<, node -e.
  As of 2026-04-15, these are sub-classified by code structure into
  `run-inline-verify`, `read-via-inline-script`, `edit-via-inline-script`,
  `create-file-via-inline-script`, and `check-version`. The residual
  `run-inline-snippet` (8% of inline snippets) captures code that doesn't match
  any structural pattern.
- `run-custom-script` is the catch-all for named scripts that don't match
  repro/test/verify filename patterns.
- `create-file` is the catch-all for created files that don't match any pattern.
  These are often new source files being added as part of the fix.
- `bash-other` should be <2% of steps. If it's higher, inspect examples and add rules.

## Implementation fixes applied (2026-04-13)

Implemented in: `scripts/classify_intent.py`

After running on random samples (30 Claude + 30 GPT trajectories) and manually checking
step/action pairs against raw trajectory source, I fixed the following inconsistencies:

1. **Malformed `bash -lc` unwrapping**
   - **Issue:** Many actions looked like `bash -lc "..."}` (extra trailing brace from serialized tool calls), causing commands like `grep`/`sed`/`ls` to fall into `bash-other`.
   - **Fix:** Added robust shell-wrapper stripping that handles normal quotes, `$'...'`, and malformed `"..."}` endings.
   - **Why:** Restores the literal command so the intended rule can trigger.

2. **False failed-intent matches from substring checks**
   - **Issue:** `(failed)` routing used broad `contains("test")` / `contains("find")`, so paths like `/app/test/...` and python heredoc bodies could be misclassified.
   - **Fix:** Failed routing now classifies from the **command head** (first line) using command-shape checks (`grep/find`, test-runner commands, python/node/go-run, read commands).
   - **Why:** Keeps failure labels tied to actual attempted shell command type, not incidental substrings in paths/code.

3. **Wrapper prefixes hiding real command type**
   - **Issue:** Prefixes like `timeout 60 ...`, env assignments (`FOO=bar ...`), and `set -e...;` prevented matching of downstream command.
   - **Fix:** Added normalization that strips common leading wrappers before intent matching.
   - **Why:** Preserves literal intent while handling common shell scaffolding.

4. **Git commands with `-C` not recognized**
   - **Issue:** `git -C /app status|show|log|diff` went to `bash-other`.
   - **Fix:** Added git subcommand extraction that skips git flags (including `-C`).
   - **Why:** Correctly maps to `git-inspect` / `git-diff` / `git-stash`.

5. **High `bash-other` from common verification/build invocations**
   - **Issue:** Frequent `npx tsc`, type-check/build script calls, and named script runs via `go run`/`sh` were under-classified.
   - **Fix:**
     - classify `npx tsc` / `tsc` / local `tsc` binaries as `compile-build`
     - classify `yarn test` as `run-test-suite`
     - classify named scripts run via `go run`, `sh`, `bash`, `./script.sh` through existing repro/verify/custom filename rules
   - **Why:** Reduces fallback noise and better matches observed literal command intent.

### Validation snapshot after fixes

- Random sample check: **30 Claude + 30 GPT** trajectories
  - Claude sample: `bash-other` = **1.17%**
  - GPT sample: `bash-other` = **0.89%**
- Full dataset sanity check:
  - Claude: `bash-other` = **1.64%**
  - GPT: `bash-other` = **1.44%**

This keeps fallback usage in the target range while preserving deterministic,
command-literal classification.

## Sequence-layer intents (deterministic, local-history)

Implemented in: `scripts/classify_intent.py` (`--sequence-layer`)

This is a **second pass** on top of base intents. It uses short, deterministic
history (no model inference, no long-chain pattern matching) to capture iteration
patterns such as re-testing/re-verifying.

### Sequence labels

- `seq-none` — no sequence-specific pattern detected
- `seq-verify-after-edit` — a verify/test/build step after one or more edits since last verify
- `seq-verify-rerun-no-edit` — another verify/test/build step with no intervening edit
- `seq-verify-rerun-same-command` — same verify command repeated without edits
- `seq-repro-after-edit` — repro/inline run after one or more edits since last repro
- `seq-repro-rerun-no-edit` — another repro/inline run with no intervening edit
- `seq-repro-rerun-same-command` — same repro command repeated without edits
- `seq-reread-edited-file` — reading a file that was edited earlier in the same trajectory
- `seq-edit-after-failed-verify` — edit immediately after failed verify/script run
- `seq-diagnose-read-after-failed-verify` — read step immediately after failed verify/script run
- `seq-diagnose-search-after-failed-verify` — search step immediately after failed verify/script run
- `seq-submit-after-verify` — submit after at least one verify step has occurred

### Sequence rules (pseudocode)

```
function classify_sequence_layer(trajectory, base_intents):
    seen_verify = false
    seen_repro = false
    edited_since_verify = false
    edited_since_repro = false
    prev_verify_signature = ""
    prev_repro_signature = ""
    edited_paths = set()
    prev_base = ""

    for each step i:
        base = base_intents[i]
        signature = normalized_command_head(step.action)
        seq = "seq-none"

        if base in VERIFY_SET:
            if seen_verify and not edited_since_verify and signature == prev_verify_signature:
                seq = "seq-verify-rerun-same-command"
            elif edited_since_verify:
                seq = "seq-verify-after-edit"
            elif seen_verify and not edited_since_verify:
                seq = "seq-verify-rerun-no-edit"

            seen_verify = true
            edited_since_verify = false
            prev_verify_signature = signature

        elif base in REPRO_SET:
            if seen_repro and not edited_since_repro and signature == prev_repro_signature:
                seq = "seq-repro-rerun-same-command"
            elif edited_since_repro:
                seq = "seq-repro-after-edit"
            elif seen_repro and not edited_since_repro:
                seq = "seq-repro-rerun-no-edit"

            seen_repro = true
            edited_since_repro = false
            prev_repro_signature = signature

        elif base in EDIT_SET:
            if prev_base in FAILED_VERIFY_SET:
                seq = "seq-edit-after-failed-verify"

            edited_since_verify = true
            edited_since_repro = true
            edited_paths.add(extract_path_if_any(step.action))

        elif base in READ_SET:
            if prev_base in FAILED_VERIFY_SET:
                seq = "seq-diagnose-read-after-failed-verify"
            elif extract_path_if_any(step.action) in edited_paths:
                seq = "seq-reread-edited-file"

        elif base in SEARCH_SET and prev_base in FAILED_VERIFY_SET:
            seq = "seq-diagnose-search-after-failed-verify"

        elif base == "submit" and seen_verify:
            seq = "seq-submit-after-verify"

        output seq
        prev_base = base
```

### Manual verification + subset run

I manually inspected sampled trajectories (30 Claude + 30 GPT) and validated
sequence labels against raw action strings.

Examples that matched well:
- repeated `pytest`/`go test` runs after edits → `seq-verify-after-edit`
- consecutive verify runs with no edits → `seq-verify-rerun-no-edit`
- repeated `python /app/repro.py` with no edits → `seq-repro-rerun-same-command`
- opening the same source file after editing it → `seq-reread-edited-file`

Subset sanity:
- sequence layer produced expected re-test/repro loops without requiring long
  sequence templates
- classification remains deterministic and local-history based

## Verify outcome detection

Implemented in: `scripts/classify_intent.py` (`classify_verify_outcome`)

For verify-run intents (`run-test-suite`, `run-test-specific`, `run-verify-script`,
`run-custom-script`, `compile-build`, `syntax-check`, `run-inline-verify`), the
observation field is parsed for unambiguous pass/fail signals. Returns `'pass'`,
`'fail'`, or `''` (unknown).

### Parsed summary formats

| Runner | Pass signal | Fail signal |
|---|---|---|
| pytest | `N passed ... in Xs` summary line | `N failed` or `N error` in summary line |
| mocha | `N passing` | `N failing` (N > 0) |
| go test | `ok  package` | `FAIL  package` or `--- FAIL:` |
| jest | `Tests: N passed` | `Tests: N failed` |
| compile (go build, go vet, make) | Empty/short output | Error text, non-zero exit |
| syntax-check (py_compile) | No output or no error | `SyntaxError` |
| syntax-check (node -c with && echo) | Echo text appears in output | Error before echo |
| custom scripts | `N passed, 0 failed` summary | Traceback, throw/Error, non-zero exit |

### Design constraints

- Only classifies when the signal is **unambiguous from the observation text**.
- Custom verify scripts (agent-created throwaway scripts) have variable output
  formats. Only structured `N passed, M failed` summaries and clear error
  signals (Traceback, throw) are detected. Everything else returns `''`.
- The outcome is per-step, not per-test. A step where 80 tests pass and 1
  fails is classified as `'fail'`.

### Aggregate stats (2026-04-15, after inline snippet reclassification)

| | Claude 4.5 | GPT-5 |
|---|---|---|
| verify-pass | 5,276 | 338 |
| verify-fail | 1,323 | 445 |
| pass rate (of detected) | 80% | 43% |

## Sequence markers: first-all-pass and work-done

Implemented in: `scripts/classify_intent.py` (`classify_sequence_layer` with
`verify_outcomes` parameter)

Two retrospective markers that use verify outcomes + source edit positions:

### `seq-first-all-pass`

The first verify step where:
1. The outcome is `'pass'`, AND
2. The step occurs after the **last source edit** (`edit-source`, `insert-source`,
   `apply-patch`, `edit-via-inline-script` — excludes `create-file` which is often
   a throwaway script)

This marks the first moment the tests confirm the implementation works, after
the agent has stopped modifying source files.

### `seq-work-done`

Same step as `seq-first-all-pass`, but only if no source edits follow it
(verified retrospectively). In practice, `seq-work-done` and
`seq-first-all-pass` coincide by construction, since `first-all-pass`
already requires being after the *last* source edit.

If a trajectory has source edits but never gets a verify-pass after the last
one (e.g. tests keep failing), neither marker is emitted.

### Limitations

- **Partial test suites**: The agent may run only a subset of tests that pass,
  while other tests still fail. `work-done` fires on whatever the agent chose
  to run, not on the full suite.
- **`create-file` excluded from source edits**: Throwaway scripts like
  `final_verification.py` often classify as `create-file`. Including them would
  push `last_source_edit_idx` too late. The trade-off is that genuine new source
  files (e.g. `src/controllers/new.js`) are also excluded.
- **No outcome = no marker**: Trajectories where the agent only runs custom
  verify scripts with ambiguous output won't get `work-done`.

### Aggregate stats (2026-04-15, after inline snippet reclassification)

| | Claude 4.5 | GPT-5 |
|---|---|---|
| Trajectories with `seq-work-done` | 592 / 730 (81%) | 82 / 730 (11%) |

GPT's work-done count increased from 26 to 82 after reclassification because
`run-inline-verify` (formerly `run-inline-snippet`) is now in the verify set
and can trigger work-done.

## Hierarchical intent layer (high-level categories + dot notation)

Implemented in: `scripts/classify_intent.py` (`--hierarchical-layer`)

This third layer groups every base intent into a compact set of high-level
categories. Output labels are emitted as:

- `<high-level>.<base-intent>`
- Example: `read-code.read-file-full`

### High-level categories (9 total)

1. `read-code`
2. `search-navigate`
3. `reproduce`
4. `implement`
5. `verify`
6. `git`
7. `infrastructure`
8. `failed`
9. `other`

### Deterministic mapping

- `read-code.*`
  - `read-file-full`, `read-file-range`, `read-file-full(truncated)`, `read-test-file`, `read-config-file`, `read-via-bash`
- `search-navigate.*`
  - `view-directory`, `list-directory`, `search-keyword`, `search-files-by-name`, `search-files-by-content`, `inspect-file-metadata`
- `reproduce.*`
  - `create-repro-script`, `run-repro-script`, `run-inline-snippet`
- `implement.*`
  - `edit-source`, `insert-source`, `apply-patch`, `create-file`, `edit-via-inline-script`, `create-file-via-inline-script`
- `verify.*`
  - `run-test-suite`, `run-test-specific`, `create-test-script`, `run-verify-script`, `create-verify-script`, `edit-test-or-repro`, `run-custom-script`, `syntax-check`, `compile-build`, `run-inline-verify`
- `read-code.*` (also includes)
  - `read-via-inline-script`
- `search-navigate.*` (also includes)
  - `check-version`
- `git.*`
  - `git-diff`, `git-inspect`, `git-stash`
- `infrastructure.*`
  - `file-cleanup`, `create-documentation`, `start-service`, `install-deps`, `check-tool-exists`
- `failed.*`
  - `search-keyword(failed)`, `read-via-bash(failed)`, `run-script(failed)`, `run-test-suite(failed)`, `bash-command(failed)`
- `other.*`
  - `submit`, `empty`, `echo`, `bash-other`, `undo-edit`

### Hierarchical rules (pseudocode)

```
function hierarchical_intent(base_intent):
    high_level = INTENT_TO_HIGH_LEVEL.get(base_intent, "other")
    return high_level + "." + base_intent
```

This layer is intentionally simple and fully deterministic. It compresses
fine-grained mechanics into higher-level behavioral buckets while preserving
traceability to the original base intent.
