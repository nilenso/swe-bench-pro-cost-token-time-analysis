# Script Refactoring Plan

## Expected layered architecture

### Layer 1: Definitions / Models

Four levels of hierarchy, declaratively defined with descriptions:

1. **Tool** — the literal command (grep, pytest, str_replace_editor view, git diff)
2. **Intent** — what the tool use means (view-file, run-test-suite, edit-source, search-keyword)
3. **High-level category** — the broad action type (read, search, edit, verify, git, housekeeping)
4. **Phase** — the trajectory phase (understand, reproduce, edit, verify, cleanup)

These should be models, clearly defined, declarative, easy to read. With descriptions.

### Layer 2: Heuristic classification

The actual classification of tool invocations into intents. Regex-based pattern matching on action strings and observations. This is the meat, the bulk of the logic, the real value.

### Layer 3: Orchestration

Parsing .traj files, running through all files, parallelising, caching, and producing per-file results.

### Layer 4: Sequential analysis

Making additional intents based on positions, relative positions. E.g. "verify after edit", "rerun same command", "diagnose after failed verify".

### Layer 5: Analysis / Aggregation

Aggregating per-file results into the metrics we need: proportions, bigram matrices, phase profiles, step distributions.

### Layer 6: Presentation

The display / HTML report. CSS, chart rendering, interactive elements.

---

## Current state

### What's clean

- **Layer 2** (heuristic classification) — `classify_step()` in `classify_intent.py` is well-contained (~200 lines). All regex work, command unwrapping, and tool detection lives here.
- **Layer 4** (sequential analysis) — `classify_sequence_layer()` in `classify_intent.py` exists and works, though it's not used by the analytics page.

### What's scattered or duplicated

- **Layer 1** (definitions) — taxonomy is split across `classify_intent.py` (INTENT_TO_HIGH_LEVEL), `build_analytics.py` (HIGH_LEVEL_LETTER, HIGH_LEVEL_COLORS, INTENT_DISPLAY_NAMES), `intent_descriptions.json`, and inline JS in `render_html()`. The phase grouping (understand = read+search, cleanup = git+housekeeping) exists only in JavaScript. There's no single place to read the full four-level hierarchy with descriptions.

- **Layer 3** (orchestration) — split across both files. `classify_intent.py` has its own `main()` with ProcessPoolExecutor for CLI batch classification. `build_analytics.py` has its own parallel pipeline (`_process_one_file` + `build_payload` + disk cache). Two independent orchestration paths doing similar work.

- **Layer 5** (analysis) — `build_payload()` in `build_analytics.py` handles aggregation. Some filtering and sorting also happens in the JavaScript presentation layer.

- **Layer 6** (presentation) — `render_html()` is a ~600-line f-string containing CSS, HTML, and all JavaScript. The JS does additional computation (filtering thresholds, sorting by gap, bar sizing) that arguably belongs in the analysis layer.

### What's missing

- No unified model file. The four-level hierarchy should live in one place.
- Phase grouping is ad-hoc (only in JS).
- `build_analytics.py` doesn't call into `classify_intent.py`'s orchestration, it reimplements it.
- Sequential analysis (layer 4) isn't connected to the analytics page at all.
- The HTML template mixes presentation logic with analysis logic.
