# Coding-agent trajectory analysis

This repo contains analysis of **SWE-Bench Pro trajectories** and **other coding-agent trajectories** as well, including Pi transcripts.

The main analysis families here are:

1. **SWE-Bench Pro operating-profile analysis** — cost, tokens, tool time, and paired task comparisons.
2. **Trajectory-shape analysis** — workflow patterns, intent frequencies, and reference tables across both SWE-Bench Pro and Pi transcripts.

The published HTML artifacts live in `docs/`.

---

## Start here

### Trajectory shapes / intent analysis

#### SWE-Bench Pro
- [Analytics — all 4 models](docs/analytics.html)
- [Analytics — Sonnet 4.5 vs GPT-5](docs/analytics-sonnet-gpt5.html)
- [Reference tables — all 4 models](docs/reference.html)
- [Reference tables — Sonnet 4.5 vs GPT-5](docs/reference-sonnet-gpt5.html)

#### Pi transcripts
- [Analytics — all available Pi models](docs/pi-analytics.html)
- [Analytics — Opus 4.5/4.6 vs GPT-5.4](docs/pi-analytics-opus-gpt54.html)
- [Reference tables — all available Pi models](docs/pi-references.html)
- [Reference tables — Opus 4.5/4.6 vs GPT-5.4](docs/pi-references-opus-gpt54.html)

#### Shared appendix / viewers
- [Trajectory sequence viewer](docs/trajectory-sequences.html)
- [Intent taxonomy tree](docs/intent-taxonomy.html)
- [Intent classification rules (SWE-Agent)](docs/intent-classification-rules.md)
- [Intent classification rules (Pi)](docs/pi-intent-classification-rules.md)

### Cost / token / time analysis
- [Main interactive report](docs/index.html)
- [Parity scatter variant](docs/parity.html)
- [White ratio charts](docs/ratio-charts-white.html)
- [Unsubmitted-task report](docs/unsubmitted.html)
- [Text summary](docs/report.txt)

---

## Repo layout

The repo is intentionally split by responsibility:

- `analysis/` — reusable SWE-Bench Pro analysis package
  - ingestion, classification, aggregation, failure-mode helpers
- `analysis_pi/` — reusable Pi transcript analysis package
  - session filtering, classification, aggregation, user-message analysis
- `scripts/` — thin entrypoints and HTML builders
  - `build_*.py` page generators
  - `extract_stats*.py` cost/token/time extraction
- `docs/` — committed, published outputs
  - this is the canonical location for rendered HTMLs linked above
- `generated/` — local scratch outputs created by build scripts
  - not committed
- `data/` — downloaded benchmark and Pi transcript inputs

In short: **analysis logic lives in `analysis*/`, page-building entrypoints live in `scripts/`, published artifacts live in `docs/`.**

---

## What each analysis family answers

### 1) SWE-Bench Pro operating profile
Focuses on paired-task comparisons for October 2025 benchmark runs:

- cost per task
- input / output / total token behavior
- tool execution time
- steps, actions, and content volume
- paired Sonnet 4.5 / GPT-5 ratios

Main methodology notes live in:

- [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md)
- [`scripts/extract_stats_fast.py`](scripts/extract_stats_fast.py)
- [`scripts/report_template.html`](scripts/report_template.html)

### 2) Trajectory shapes / intent taxonomy
Focuses on workflow structure rather than only success rate:

- high-level action frequencies
- low-level intent frequencies
- trajectory-shape phase profiles
- deterministic intent taxonomies
- cross-harness comparison between SWE-Agent and Pi

Main source files:

- [`analysis/`](analysis)
- [`analysis_pi/`](analysis_pi)
- [`scripts/build_analytics.py`](scripts/build_analytics.py)
- [`scripts/build_reference_tables.py`](scripts/build_reference_tables.py)
- [`scripts/build_pi_analytics.py`](scripts/build_pi_analytics.py)
- [`scripts/build_pi_reference_tables.py`](scripts/build_pi_reference_tables.py)

---

## Quickstart

### Prerequisites

- Python 3.10+
- AWS CLI
- Python deps:

```bash
pip install orjson tiktoken
```

### Download benchmark data

```bash
bash download.sh
```

This populates `data/gpt5/` and `data/claude45/`.

---

## Build the cost / token / time report set

```bash
./run_analysis.sh
```

This writes:

- local scratch outputs to `generated/`
- published copies to `docs/`

Key outputs:

- `docs/index.html`
- `docs/unsubmitted.html`
- `docs/report.txt`

Open the local build:

```bash
./run_analysis.sh --open
```

Use fewer workers if needed:

```bash
./run_analysis.sh -w 8
```

---

## Build trajectory-shape / reference pages

### SWE-Bench Pro analytics

```bash
python3 scripts/build_analytics.py --data-root data --output docs/analytics.html
python3 scripts/build_analytics.py --data-root data --models claude45,gpt5 --output docs/analytics-sonnet-gpt5.html
```

### SWE-Bench Pro reference tables

```bash
python3 scripts/build_reference_tables.py --data-root data --output docs/reference.html
python3 scripts/build_reference_tables.py --data-root data --models claude45,gpt5 --output docs/reference-sonnet-gpt5.html
```

### Pi analytics

All available Pi models:

```bash
python3 scripts/build_pi_analytics.py --data-root data/pi-mono --output docs/pi-analytics.html
```

Opus 4.5/4.6 vs GPT-5.4:

```bash
python3 scripts/build_pi_analytics.py \
  --data-root data/pi-mono \
  --models claude-opus-4-5 claude-opus-4-6 gpt-5.4 \
  --merge-models claude-opus-4-5,claude-opus-4-6=claude-opus-4-5-6:Opus\ 4.5/4.6 \
  --output docs/pi-analytics-opus-gpt54.html
```

### Pi reference tables

All available Pi models:

```bash
python3 scripts/build_pi_reference_tables.py --data-root data/pi-mono --output docs/pi-references.html
```

Opus 4.5/4.6 vs GPT-5.4:

```bash
python3 scripts/build_pi_reference_tables.py \
  --data-root data/pi-mono \
  --models claude-opus-4-5 claude-opus-4-6 gpt-5.4 \
  --merge-models claude-opus-4-5,claude-opus-4-6=claude-opus-4-5-6:Opus\ 4.5/4.6 \
  --output docs/pi-references-opus-gpt54.html
```

### Other supporting pages

```bash
python3 scripts/build_report_parity.py stats.json -o docs/parity.html
python3 scripts/build_white_ratio_charts.py stats.json -o docs/ratio-charts-white.html
python3 scripts/build_trajectory_sequence_viewer.py --output docs/trajectory-sequences.html
```

---

## Notes on methodology

### Pairing policy
For the cost / token / time analysis, the main paired comparisons require:

- same `instance_id`
- both model trajectories present
- both trajectories submitted

This avoids cross-task composition bias.

### Ratio convention
Where ratio charts are used:

- **ratio = Sonnet 4.5 / GPT-5**
- `1×` = equal
- `<1×` means Sonnet is lower
- `>1×` means Sonnet is higher

### Token caveat
`tokens_received` from upstream SWE-Agent undercounts some output, especially tool-call arguments. Use the recounted `output_tokens.total` fields when validating token-accounting conclusions.

For details, see [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).

---

## Related writing

- Blog post: **Checking my model vibes against SWE-Bench Pro**
- Blog post: **Trajectory shapes**
