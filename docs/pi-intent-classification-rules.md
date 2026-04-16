# Pi Transcript Intent Classification Rules

This is the Pi-harness counterpart to `docs/intent-classification-rules.md`.

The **taxonomy is unchanged**. We still classify each action into the same base intents and the same high-level buckets:

- read
- search
- reproduce
- edit
- verify
- git
- housekeeping
- failed
- other

## Harness adaptation

Pi transcripts are JSONL session logs, not SWE-Agent trajectory JSON files. The classifier therefore converts Pi tool calls into the equivalent intent semantics:

- `read(path, offset?, limit?)`
  - `read-file-range` when `offset`/`limit` is present
  - otherwise `read-file-full`, `read-test-file`, `read-config-file`, or `read-file-full(truncated)` using the same filename-based rules
- `edit(path, ...)`
  - `edit-source`, `edit-test-or-repro`, or `insert-source`
- `write(path, content)`
  - `create-file`, `create-test-script`, `create-verify-script`, `create-repro-script`, or `create-documentation`
- `bash(command, timeout?)`
  - classified with the original deterministic shell-command rules
- lightweight helper tools
  - `ls` → list-directory
  - `find` → search-files-by-name
  - `grep` → search-keyword
  - `finish_and_exit` → submit analogue
  - planning / greeting helpers (`todo`, `watch_plans_start`, `greet`, `hello`) → `other`

## Observation source

For each Pi tool call, the classifier pairs it with its matching `toolResult` message using `toolCallId`. The result text becomes the step observation, which lets the original verify-outcome parser still detect test/build pass/fail signals for `bash` verification commands.

## Session-level model purity

For the copied Pi analytics pages, a session is treated as **single-model only** when there is exactly one distinct model across both:

- assistant messages: `message.role == "assistant"` → `message.model`
- model-switch events: `type == "model_change"` → `modelId`

Model names are lightly normalized first (for example dot vs hyphen variants like `claude-opus-4.6` / `claude-opus-4-6`). This avoids counting sessions as single-model when the assistant replies all come from one model but the transcript also records a switch in `model_change` events.

## Session-level completion

Pi transcripts do not have SWE-Bench-style benchmark resolution labels. Instead, the copied analysis uses **completed cleanly**:

- last assistant `stopReason == "stop"`, or
- an explicit `finish_and_exit` tool call

This is used anywhere the copied analytics needed a session-outcome field.

## Important caveat

`completed cleanly` is **not** equivalent to benchmark correctness or issue resolution. It only means the session reached a clean terminal state in the Pi harness.
