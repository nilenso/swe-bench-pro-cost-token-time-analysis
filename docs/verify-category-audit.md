# Audit: Is Claude's "verify" category actually implementation?

The heatmap shows verify growing to 37–49% in the last quarter of Claude 4.5
trajectories while edit fades to 2%. This doc investigates whether "verify" is
misclassified implementation work.

## How "verify" is classified

The `verify` high-level category maps from these base intents
(see `docs/intent-classification-rules.md`):

- `run-test-suite` — pytest, go test, npm test
- `run-test-specific` — pytest with -k or ::
- `run-verify-script` — running files named test_*, verify*, check*
- `create-test-script` — **creating** files named test_*, *test.py
- `create-verify-script` — **creating** files named verify*, check*
- `edit-test-or-repro` — **str_replace on** test/repro files
- `run-custom-script` — running any named script (catch-all)
- `syntax-check` — py_compile, node -c
- `compile-build` — go build, go vet, make

Three of these (`create-test-script`, `create-verify-script`,
`edit-test-or-repro`) involve writing or editing code, not running checks.

## Sub-intent breakdown in the last 30% of Claude trajectories

Across all 730 Claude trajectories, 6,799 steps in the final 30% are classified
as "verify":

| Sub-intent | Count | % of verify |
|---|---|---|
| `run-test-suite` | 2,708 | 39.8% |
| `run-verify-script` | 1,304 | 19.2% |
| **`create-test-script`** | **895** | **13.2%** |
| `compile-build` | 594 | 8.7% |
| `run-test-specific` | 425 | 6.3% |
| `run-custom-script` | 327 | 4.8% |
| **`create-verify-script`** | **217** | **3.2%** |
| **`edit-test-or-repro`** | **203** | **3.0%** |
| `syntax-check` | 126 | 1.9% |

**~19.4% of "verify" is writing code** (create-test-script +
create-verify-script + edit-test-or-repro = 1,315 / 6,799). But it's writing
*throwaway* test/verify scripts, not the actual fix.

All high-level intents in the last 30% for context:

| High-level | Count | % |
|---|---|---|
| verify | 6,799 | 39.3% |
| read | 2,721 | 15.7% |
| housekeeping | 1,859 | 10.7% |
| other | 1,856 | 10.7% |
| search | 1,618 | 9.4% |
| edit | 974 | 5.6% |
| reproduce | 906 | 5.2% |
| git | 508 | 2.9% |
| failed | 62 | 0.4% |


## Example trajectories

### Example 1: Ansible icx_ping module

**Trajectory:** `instance_ansible__ansible-622a493ae03bd5e5cf517d336fc426e9d12208c7`
**84 steps total. Last 30% starts at step 58.**
**Late verify: 11 steps. Late edit: 0 steps.**

The issue involves implementing a ping module for ICX network devices. Claude
finishes the main implementation by step ~57. From step 58 onward:

- Creates `verify_icx_ping.py` → runs it → deletes it
- Runs `pytest test/units/modules/network/icx/` (3 separate times)
- Creates `final_test.py` → runs it → deletes it
- Runs `py_compile` syntax check
- Creates `test_edge_cases.py` → runs it → deletes it
- Runs pytest again

**Caveat:** I did not read the test output (observation field) for this
trajectory. The claim below is based on the action strings, not on whether
tests actually passed.

**Surface read:** Looks like verification — running tests, creating throwaway
check scripts, syntax-checking. But without reading the observations, I can't
confirm whether these tests pass or fail.

Late-phase steps:

```
 58 [verify       | run-verify-script        ] python test_pr_requirements.py
 59 [housekeeping | file-cleanup             ] rm -f test_icx_ping_*.py test_pr_requirements.py
 60 [search       | list-directory           ] ls -la .../icx/
 61 [read         | read-file-range          ] view icx_ping.py :125-165
 62 [read         | read-file-range          ] view icx_ping.py :315-330
 63 [verify       | create-verify-script     ] create verify_icx_ping.py
 64 [verify       | run-verify-script        ] python verify_icx_ping.py
 65 [verify       | run-test-suite           ] pytest test/units/modules/network/icx/ -v
 66 [housekeeping | file-cleanup             ] rm verify_icx_ping.py
 67 [verify       | create-test-script       ] create final_test.py
 68 [verify       | run-custom-script        ] python final_test.py
 69 [housekeeping | file-cleanup             ] rm final_test.py
 70 [reproduce    | run-inline-snippet       ] python -c "..."
 71 [verify       | run-test-suite           ] pytest test/units/modules/network/icx/ -q
 72 [housekeeping | create-documentation     ] create IMPLEMENTATION_SUMMARY.md
 73 [search       | inspect-file-metadata    ] wc -l icx_ping.py
 74 [verify       | syntax-check             ] py_compile icx_ping.py
 75 [reproduce    | run-inline-snippet       ] python -c "..."
 76 [verify       | create-test-script       ] create test_edge_cases.py
 77 [verify       | run-verify-script        ] python test_edge_cases.py
 78 [housekeeping | file-cleanup             ] rm test_edge_cases.py
 79 [reproduce    | run-inline-snippet       ] python -c "..."
 80 [verify       | run-test-suite           ] pytest ...icx/ -v | grep
 81 [read         | read-via-bash            ] head -n 120 icx_ping.py
 82 [other        | empty                    ]
 83 [other        | submit                   ] submit
```


### Example 2: Navidrome channel count support

**Trajectory:** `instance_navidrome__navidrome-e12a14a87d392ac70ee4cc8079e3c3e0103dbcb2`
**76 steps total. Last 30% starts at step 53.**
**Late verify: 11 steps. Late edit: 0 source edits.**

Issue about adding audio channel count parsing from ffmpeg output. Implementation
done by ~step 52. Then:

- Runs `test_edge_cases.py` → **edits the test script** (fixing a regex in the
  test, not in source) → re-runs it
- Runs `go test ./scanner/metadata/...`
- Runs `verify_implementation.py`
- Runs `go test ./...` (full suite)
- Runs `go build ./...`
- Creates `test_channels_demo.go` → **edits it** (removes unused import) →
  runs it → deletes it
- Runs `go test ./...` twice more
- Creates `IMPLEMENTATION_SUMMARY.md`, `CHANGES_VERIFICATION.md`,
  `IMPLEMENTATION_COMPLETE.md` (housekeeping)

The two `edit-test-or-repro` steps were fixing the throwaway test scripts
themselves (removing an unused import, fixing a regex pattern in the test), not
implementing anything in the actual source.

**Caveat:** Test output not checked — same limitation as Example 1.

Late-phase steps:

```
 53 [verify       | run-verify-script        ] python3 test_edge_cases.py
 54 [verify       | edit-test-or-repro       ] str_replace test_edge_cases.py (fix regex)
 55 [verify       | run-verify-script        ] python3 test_edge_cases.py
 56 [verify       | run-test-suite           ] go test ./scanner/metadata/... -v
 57 [verify       | run-verify-script        ] python3 verify_implementation.py
 58 [housekeeping | file-cleanup             ] rm verify_implementation.py test_edge_cases.py
 59 [verify       | run-test-suite           ] go test ./...
 60 [housekeeping | create-documentation     ] create IMPLEMENTATION_SUMMARY.md
 61 [other        | echo                     ] echo "=== Checking all modified files ==="
 62 [verify       | compile-build            ] go build -v ./...
 63 [housekeeping | create-documentation     ] create CHANGES_VERIFICATION.md
 64 [verify       | create-test-script       ] create test_channels_demo.go
 65 [reproduce    | run-repro-script         ] go run test_channels_demo.go
 66 [verify       | edit-test-or-repro       ] str_replace test_channels_demo.go (remove unused import)
 67 [reproduce    | run-repro-script         ] go run test_channels_demo.go
 68 [housekeeping | file-cleanup             ] rm test_channels_demo.go ...
 69 [verify       | run-test-suite           ] go test ./... -v | grep PASS/FAIL
 70 [verify       | run-test-suite           ] go test ./scanner/... -v
 71 [housekeeping | create-documentation     ] create IMPLEMENTATION_COMPLETE.md
 72 [read         | read-via-bash            ] cat > /tmp/final_verification.sh
 73 [housekeeping | file-cleanup             ] rm IMPLEMENTATION_COMPLETE.md ...
 74 [other        | empty                    ]
 75 [other        | submit                   ] submit
```


### Example 3: Ansible Galaxy API caching (FAILED — resolved=false)

**Trajectory:** `instance_ansible__ansible-de5858f48dc9e1ce9117034e0d7e76806f420ca8`
**117 steps total. Last 30% starts at step 81.**
**Late verify: 13 steps. Late edit: 2 steps.**
**Eval result: FAILED (resolved=false)**

Issue: add caching support for Galaxy API requests, with `--no-cache` and
`--clear-response-cache` CLI flags, cache invalidation, thread-safe file
access, etc. A large feature with many requirements.

Claude makes 10 source edits between steps 30–53 (adding config, imports,
cache logic, CLI flags). But implementation isn't done — Claude is still
toggling the `no_cache` default back and forth:

- **Step 53** (edit): changes `no_cache=True` → `no_cache=False` ("caching
  should be enabled by default")
- **Step 71** (edit): changes it back `no_cache=False` → `no_cache=True`
  ("maybe for backward compatibility with tests, I should default to
  no_cache=True")
- **Step 90** (edit): tweaks the cache directory creation error handling

Then in the late phase, **tests are failing and Claude can't fix them**:

- Step 84: `pytest test/units/galaxy/test_api.py` — **41 passed** (API tests ok)
- Step 85: `pytest test/units/cli/test_galaxy.py` — **1 failed, 80 passed**
  (`test_collection_install_with_names` fails: `assert mock_warning.call_count == 1`
  but got 2 — Claude's cache code emits an extra warning)
- Steps 86, 91, 95, 96, 103: Runs that same failing test **5 more times** with
  different `grep`/`--tb` filters. It fails identically every time.
- Steps 102–104: Stashes changes, re-runs test (still fails), pops stash —
  trying to determine if the test was already broken upstream
- Step 105: Runs full cli test suite — now **4 tests fail**
- Steps 110–112: Creates `final_verification.py`, runs it, runs pytest again.
  Still failing.
- Submits anyway.

**This is the most important example.** The "verify" classification is correct —
Claude IS running tests. But the verify steps aren't "confirming the fix works."
They're part of an **unresolved debug loop**: test fails → re-run test with
different output format → test still fails → edit something → test still fails
→ give up and submit. The verify steps here represent failed problem-solving,
not successful confirmation.

Late-phase steps:

```
 81 [verify       | create-test-script       ] create test_cli_args.py
 82 [verify       | run-verify-script        ] python test_cli_args.py
 83 [reproduce    | run-repro-script         ] python test_reproduce.py && ...
 84 [verify       | run-test-suite           ] pytest test/units/galaxy/test_api.py -xvs
 85 [verify       | run-test-suite           ] pytest test/units/cli/test_galaxy.py -x
 86 [verify       | run-test-specific        ] pytest ...::test_collection_install_with_names -xvs
 87 [read         | read-file-range          ] view test_galaxy.py :730-760
 88 [search       | search-keyword           ] grep "display.warning" galaxy/api.py
 89 [reproduce    | run-inline-snippet       ] python -c "..."
 90 [edit         | edit-source              ] str_replace galaxy/api.py  ← ACTUAL SOURCE EDIT
 91 [verify       | run-test-specific        ] pytest ...::test_collection_install -xvs
 92 [reproduce    | run-inline-snippet       ] python -c "..."
 93 [reproduce    | run-inline-snippet       ] python -c "..."
 94 [read         | read-file-range          ] view test_galaxy.py :734-760
 95 [verify       | run-test-specific        ] pytest ... | grep -A5 -B5 "call_count"
 96 [verify       | run-test-specific        ] pytest ... --tb=short | grep "FAILED"
 97 [reproduce    | run-inline-snippet       ] python3 << 'EOF' ...
 98 [search       | search-keyword           ] grep -r "running the development version"
 99 [search       | search-keyword           ] grep -B5 -A5 "running the development version"
100 [git          | git-status-log           ] git status test_galaxy.py
101 [git          | git-status-log           ] git log --oneline -10
102 [git          | git-stash                ] git stash
103 [verify       | run-test-specific        ] pytest ...::test_collection_install -xvs
104 [git          | git-stash                ] git stash pop
105 [verify       | run-test-suite           ] pytest test/units/cli/test_galaxy.py -v
106 [verify       | run-test-suite           ] pytest test/units/galaxy/ -v
107 [reproduce    | run-repro-script         ] python test_reproduce.py
108 [housekeeping | create-documentation     ] create IMPLEMENTATION_SUMMARY.md
109 [housekeeping | file-cleanup             ] rm -f test_*.py
110 [edit         | create-file              ] create final_verification.py
111 [verify       | run-custom-script        ] python final_verification.py
112 [verify       | run-test-suite           ] pytest test/units/galaxy/test_api.py -v --tb=no
113 [reproduce    | run-inline-snippet       ] python -c "..."
114 [housekeeping | file-cleanup             ] rm final_verification.py IMPLEMENTATION_SUMMARY.md
115 [other        | empty                    ]
116 [other        | submit                   ] submit
```


### Example 4: Flipt config context propagation

**Trajectory:** `instance_flipt-io__flipt-d966559200183b713cdf3ea5007a7e0ba86a5afb`
**92 steps total. Last 30% starts at step 64.**
**Late verify: 7 steps. Late edit: 2 steps (editing a throwaway demo file).**

- Runs `go test` (passes) → searches code → creates a demo file → runs it →
  edits demo file → deletes it
- Runs `go test` 3 more times → `go build` → creates docs
- Runs more `go test` → git diffs to review own changes

Genuine verification but repetitive. The edits are to a throwaway demo file.

**Caveat:** Test output not checked — same limitation as Examples 1 and 2.


## Conclusions

### Is "verify" actually implementation?

The classification is mechanically correct — test runs are test runs, build
commands are build commands. **~80% of late-phase "verify" is running checks,
not writing code.** The ~19% that writes code is creating throwaway test/verify
scripts, not implementing the actual fix.

But "correctly classified as verify" doesn't mean "successfully verifying the
fix works." The heatmap shows "verify" growing in the late phase, but that
could mean any of:

1. **Redundant re-verification** — tests pass, Claude keeps re-running them
2. **Stuck debug loops** — tests fail, Claude re-runs hoping for a different
   result or trying to read the output differently
3. **Genuine iterative debugging** — test fails, Claude edits source, re-runs,
   edits again (but the edits are sparse enough that "verify" dominates the
   bin percentages)

Example 3 (Galaxy API caching) is a clear case of #2 — Claude runs the same
failing test 5+ times, never fixes it, submits anyway. The verify steps look
like confirmation on the heatmap but are actually failed problem-solving.


## Inline snippet reclassification (2026-04-15)

The original classifier put all `python -c` / `python - <<` / `node -e`
commands into `run-inline-snippet` under "reproduce." Auditing these revealed
they serve very different purposes:

| Sub-intent | High-level | Count | % |
|---|---|---|---|
| `run-inline-verify` | verify | 1,923 | 61% |
| `read-via-inline-script` | read | 490 | 16% |
| `edit-via-inline-script` | edit | 303 | 10% |
| `create-file-via-inline-script` | edit | 89 | 3% |
| `check-version` | search | 71 | 2% |
| `run-inline-snippet` (residual) | reproduce | 253 | 8% |

The reclassification reduced "reproduce" by half (Claude 2,111 → 1,004,
GPT 3,080 → 1,723). Most of what appeared as late-phase reproduction was
actually `python -c` spot-checks (verification) or GPT editing source files
via Python string replacement instead of the editor tool.

GPT uses `edit-via-inline-script` 3.6x more than Claude (245 vs 68). This
is GPT's alternative to `str_replace_editor` — it reads a file with
`Path.read_text()`, calls `.replace()`, and writes it back. The classifier
previously counted this as "reproduce," inflating GPT's reproduction band
in the late phase of the trajectory shape chart.


## When does each model stop writing code?

The cleanest structural marker is **last source edit** — the last step where
the agent modifies a source file, whether via `str_replace`, `insert`, or
`python -c` file editing (now classified as `edit-via-inline-script`). This is
purely mechanical, requires no outcome parsing, and has an unambiguous meaning:
the agent stopped modifying the actual code at this point.

Everything after the last source edit — whether it's verification, cleanup,
documentation, or more verification — is post-implementation activity.

616 paired instances where both models submitted a patch.

### Last source edit position

| | Claude 4.5 | GPT-5 |
|---|---|---|
| Median | **61%** | **89%** |
| P25–P75 | 47%–77% | 78%–95% |
| n (trajectories with source edits) | 600 | 607 |

Claude stops modifying source code at the **61% mark**. The remaining 39% of
the trajectory is post-implementation: running tests, creating throwaway verify
scripts, writing `IMPLEMENTATION_SUMMARY.md`, running tests again, cleaning up,
running tests one more time.

GPT stops at **89%**. It spends most of its trajectory understanding the
codebase, makes its edits late, and submits shortly after.

### What this means for cost

Both models resolve at similar rates (Claude 44.5%, GPT 42.5%). But Claude's
trajectory shape means it spends roughly a third of its steps — and a third
of its tokens and cost — on activity that happens after it has stopped
changing the code. GPT spends about 10%.

The report found Claude costs 5.5x more per resolve than GPT ($25.45 vs
$4.65). The post-implementation tail isn't the only reason for that gap
(Claude's trajectories are also longer overall, and it uses more tokens per
step), but it's a visible contributor: a substantial fraction of Claude's
spend goes to re-running passing tests, creating and deleting throwaway
scripts, and writing documentation files that the benchmark doesn't evaluate.

### Verify outcome detection (supplementary)

We also built a verify outcome parser (`classify_verify_outcome` in
`scripts/classify_intent.py`) that reads test-runner output from the
observation field — pytest summary lines, mocha, go test, jest, compile
errors — returning `pass`, `fail`, or unknown per step.

**Definitions:**

- **verify-pass / verify-fail**: parsed from structured test-runner output
  (pytest `N passed in Xs`, mocha `N passing`/`N failing`, go test `ok`/`FAIL`,
  jest, compile errors). Returns unknown when format is unrecognized.
- **work-done**: first verify-pass after the last source edit. A constructed
  metric — not a benchmark definition. Depends on outcome parsing and our
  choice of which intents count as source edits.
- **resolved**: the benchmark's ground-truth. The agent's patch is applied and
  the benchmark's hidden test suite is run. Independent of whatever tests the
  agent chose to run.

**Verify outcomes across all trajectories (after inline snippet reclassification):**

| | Claude 4.5 | GPT-5 |
|---|---|---|
| Total verify steps | 16,879 (30% of steps) | 2,242 (5.2% of steps) |
| verify-pass | 5,276 | 338 |
| verify-fail | 1,323 | 445 |
| pass rate (of detected) | 80% | 43% |

Verify step counts increased after reclassifying `run-inline-snippet` —
many `python -c` spot-checks that were previously "reproduce" are now
correctly classified as `run-inline-verify` under "verify." Claude still
runs ~7.5x more verify commands than GPT. 80% of Claude's detectable
verify outcomes are passes. This sounds good until you cross-reference with
the benchmark: in 280 trajectories, Claude's chosen tests pass but the issue
isn't resolved. The agent picks which tests to run, and it picks ones that
pass.

**Work-done vs benchmark resolution (616 paired instances):**

```
Claude 4.5 — 274 resolved (44.5%)

                                    │  Count │ Resolve rate
────────────────────────────────────┼────────┼─────────────
Work-done fired                     │    521 │        46.3%
No work-done                        │     95 │        34.7%

GPT-5 — 262 resolved (42.5%)

                                    │  Count │ Resolve rate
────────────────────────────────────┼────────┼─────────────
Work-done fired                     │     76 │        57.9%
No work-done                        │    540 │        40.4%
```

Work-done is a weak predictor for Claude (46% vs 35% baseline) and a
slightly stronger one for GPT (58% vs 40%), but GPT's fires so rarely
(76/616) that it covers little of the dataset. The agent's own test
results are not a reliable proxy for the benchmark's ground truth.
