# High-level intent category renaming

The 9 high-level categories group the 35 base intents into behavioral buckets
for charts and dashboards. Several names were chosen from a systems perspective
("infrastructure", "search-navigate") rather than from the perspective of
someone watching an agentic coding tool work. This doc proposes renaming 4 of
the 9 categories.

## Proposed renames

### infrastructure → housekeeping

Sub-intents: file-cleanup, create-documentation, start-service, install-deps, check-tool-exists

"Infrastructure" reads like cloud infra / Terraform / Docker to most developers.
Looking at the Claude-vs-GPT data, the standout sub-intents are file-cleanup
(GPT 2.7%, Claude 0.0%) and create-documentation (GPT 1.2%, Claude 0.0%) —
these are the agent tidying up files and writing summary docs nobody asked for.
That's housekeeping, not infrastructure. "Housekeeping" is immediately
understood and carries just enough connotation of "not core work" to make the
comparison land without editorializing.

### search-navigate → search

Sub-intents: view-directory, list-directory, search-keyword, search-files-by-name, search-files-by-content, inspect-file-metadata

"search-navigate" is two verbs awkwardly hyphenated. All of these — grep, find,
ls, tree, directory viewing, file metadata — are the agent orienting itself in
the codebase. Developers naturally say "I searched for the function" even when
they mean `ls` or `tree`. Just **search**.

### implement → edit

Sub-intents: edit-source, insert-source, apply-patch, create-file

"Implement" implies grand purposeful construction. What's actually happening is
str_replace, insert, and patch — literal edits. **"edit"** is more honest and
reads naturally: "the agent spent 15% editing."

### read-code → read

Sub-intents: read-file-full, read-file-range, read-file-full(truncated), read-test-file, read-config-file, read-via-bash

"read-code" is slightly misleading since it includes config files and test files.
The sub-intents already specify *what* is being read. Shortening to **"read"**
is simpler and reads naturally in a sentence: "the agent spent 35% reading."

## No change needed

| Category | Why it's fine |
|---|---|
| reproduce | Create/run repro scripts, run inline snippets — all the agent trying to see the bug. Clear. |
| verify | Big bucket (tests, syntax checks, builds, verify scripts) but unified by "did my change work?" |
| git | Self-explanatory, small category. |
| failed | Clear what it means. |
| other | Catch-all, as expected. |

## Summary

| Current | Proposed |
|---|---|
| read-code | **read** |
| search-navigate | **search** |
| reproduce | reproduce (no change) |
| implement | **edit** |
| verify | verify (no change) |
| git | git (no change) |
| infrastructure | **housekeeping** |
| failed | failed (no change) |
| other | other (no change) |
