# SWE-Bench Pro Analysis: GPT-5 vs Claude Sonnet 4.5

This repository is the **data + methodology deep dive** behind the SWE-Bench Pro operating-profile analysis (cost, tokens, tool time, and behavior), not just a headline benchmark comparison.

It is designed so you can:

- reproduce paired-task analysis from raw trajectories,
- inspect exactly how metrics are computed,
- verify caveats (especially token accounting),
- and regenerate all report variants used in the write-up.

---

## What this repo answers

Beyond resolve rate, this analysis focuses on:

1. **Cost per task** (from benchmark logs),
2. **Token usage patterns** (input, output, total),
3. **Tool execution time** (not full wall-clock),
4. **Agent behavior** (steps/actions/content volume),
5. **Paired per-task comparisons** (Sonnet 4.5 / GPT-5 ratios).

The central design choice is **pairing**: compare models only on instances where both submitted on the same task.

---

## Dataset and scope

- Benchmark: **SWE-Bench Pro** public trajectories
- Models: **GPT-5** and **Claude Sonnet 4.5**
- Run slice: October 2025 leaderboard runs
- Raw trajectories: ~1460 total (730 per model)
- Paired submitted tasks used in main comparisons: **616**

Data sources:

- S3 trajectories/evals under `s3://scaleapi-results/swe-bench-pro/`
- Eval result maps from `scaleapi/SWE-bench_Pro-os`

See `RESEARCH_NOTES.md` for full provenance details.

---

## Repository map

### Core scripts

- `scripts/extract_stats_fast.py` — parallel extractor (orjson + multiprocessing)
- `scripts/extract_stats.py` — reference extractor (single-process)
- `scripts/build_report.py` — builds main interactive report (`report.html`)
- `scripts/build_text_report.py` — builds plain-text summary (`report.txt`)
- `scripts/build_unsubmitted_report.py` — analysis of non-submitted cases
- `scripts/build_report_parity.py` — parity scatter report variant
- `scripts/build_white_ratio_charts.py` — white editorial ratio chart page

### Templates

- `scripts/report_template.html` — main report template
- `scripts/report_template_parity.html` — parity report template

### Methodology docs

- `RESEARCH_NOTES.md` — long-form research notes, caveats, source context
- `docs/intent-classification-rules.md` — deterministic command-level intent taxonomy

### Generated artifacts

- `report.html` / `docs/index.html` — main interactive report
- `report_parity.html` / `docs/parity.html` — parity scatter variant
- `ratio-charts-white.html` / `docs/ratio-charts-white.html` — publication-style ratio charts
- `unsubmitted.html` / `docs/unsubmitted.html` — non-submission analysis
- `report.txt` / `docs/report.txt` — text summary for quick review/agents

---

## Methodology (important)

### 1) Pairing and filtering

- Group by `instance_id`.
- Keep only tasks with both model trajectories present.
- For paired analyses, require both to have `submitted=true`.
- This avoids cross-task composition bias from comparing different task subsets.

### 2) Ratio convention

All per-task ratio charts use:

- **ratio = Sonnet 4.5 / GPT-5**
- `1×` = equal
- `<1×` means Sonnet is lower
- `>1×` means Sonnet is higher

### 3) Metric definitions

From each trajectory:

- **Cost:** `info.model_stats.instance_cost`
- **Input tokens:** `info.model_stats.tokens_sent`
- **API calls:** `info.model_stats.api_calls`
- **Tool time:** sum/stats of `trajectory[].execution_time`
- **Steps:** `len(trajectory)`

Derived/additional fields (extractor):

- **Output tokens (recounted):** `output_tokens.total`
  - counted with `tiktoken` (`cl100k_base`) over:
    - `message.content`
    - `message.thought` (if distinct)
    - `tool_calls[].function.arguments`
- **Action breakdown:** bash/view/edit/create/search/submit/other
- **Content volume:** observation/action/thought/response char totals

### 4) Aggregation style

- Use **per-task ratios** first, then summarize with **median + IQR**.
- Do not infer direction from pooled means alone (heavy tails can mislead).

### 5) Outcome decomposition

Paired outcomes are split into:

- both solved,
- GPT-only solved,
- Sonnet-only solved,
- neither solved.

This is used alongside headline resolve rates.

---

## Critical caveats

1. **`tokens_received` undercounts output** in upstream SWE-Agent context because it does not fully account for tool-call arguments.
   - This is why recounted `output_tokens.total` exists in extracted stats.
2. **Costs are benchmark-environment proxy costs** (litellm path), not guaranteed to match current public list pricing.
3. **Tool time is not full wall-clock latency**; model inference latency is not directly captured.
4. **Hidden reasoning tokens are not visible** in trajectories.

For caveat details and links, see `RESEARCH_NOTES.md`.

---

## Quickstart

### Prerequisites

- Python 3.10+
- AWS CLI (for S3 download)
- Python deps:

```bash
pip install orjson tiktoken
```

### Download data

```bash
bash download.sh
```

This downloads both model directories into `data/gpt5` and `data/claude45`.

### Run full pipeline

```bash
./run_analysis.sh
```

Outputs:

- `stats.json`
- `report.html`
- `report.txt`
- `unsubmitted.html`
- `docs/index.html` + related docs copies

Open report:

```bash
./run_analysis.sh --open
```

Optional workers:

```bash
./run_analysis.sh -w 8
```

---

## Build optional report variants

### Parity scatter report

```bash
python3 scripts/build_report_parity.py stats.json -o report_parity.html
cp report_parity.html docs/parity.html
```

### White ratio charts page

```bash
python3 scripts/build_white_ratio_charts.py stats.json -o docs/ratio-charts-white.html
cp docs/ratio-charts-white.html ratio-charts-white.html
```

---

## Notes on token metrics across artifacts

Different artifacts may intentionally show different token views:

- Main report/parity pages emphasize extracted metrics including recounted output-token fields.
- White ratio charts currently include input/output/total views from run-stat fields used in that page builder.

If you are validating token-accounting caveats, always check field provenance in:

- `scripts/extract_stats_fast.py`
- report template metric definitions
- `RESEARCH_NOTES.md`

---

## Session-derived evolution (from Pi coding sessions)

Reviewing all repo sessions (via `pi` session transcripts) surfaced a few decisions that materially changed results and interpretation:

1. **Method baseline was locked early** in `RESEARCH_NOTES.md` (token-accounting caveats, pricing caveats, measurement limits).
2. **Pairing policy switched to “both submitted”** (instead of “both resolved”) to reduce survivorship bias in efficiency comparisons.
3. **Extraction was rebuilt for full-run iteration speed** (`scripts/extract_stats_fast.py`: orjson + multiprocessing + progress).
4. **A key eval-mapping bug was fixed** (model eval maps were colliding by identical instance IDs), which changed reported resolve-rate comparisons.
5. **Report generator was split into data + template layers** (`scripts/build_report.py` + `scripts/report_template.html`) to make visualization iteration safer.
6. **Variant visual products were added** for paired interpretability:
   - parity scatter report (`build_report_parity.py`, `report_template_parity.html`),
   - white publication-style ratio page (`build_white_ratio_charts.py`).
7. **Token interpretation was explicitly tightened**: `input + output` is useful but not equal to full billed token load in all cases.

This repo keeps all those artifacts side-by-side so readers can audit decisions, not just final charts.

---

## Related writing

Companion blog post (published separately):

- **Checking my model vibes against SWE-Bench Pro**

This repo is the technical appendix + reproducibility base for that narrative.
