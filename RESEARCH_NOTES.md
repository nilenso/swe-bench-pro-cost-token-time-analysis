# SWE-Bench Pro: Comparative Analysis of GPT-5 vs Claude Sonnet 4.5 Trajectories

## What This Is

A comparative study of how GPT-5 and Claude Sonnet 4.5 behave on the SWE-Bench Pro benchmark, focusing on tokens, cost, tool execution time, and agent behavior. The goal is to understand the structural differences in how these models approach software engineering tasks, independent of pricing.

## The Benchmark

**SWE-Bench Pro** is an enterprise-level benchmark by Scale AI for evaluating AI agents on long-horizon software engineering tasks.

- **Paper**: `swe_bench_pro_paper.pdf` (downloaded) / `swe_bench_pro_paper.txt` (text version)
- **Dataset**: https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro (731 public instances)
- **Eval harness**: https://github.com/scaleapi/SWE-bench_Pro-os
- **Leaderboard**: https://labs.scale.com/leaderboard/swe_bench_pro_public

Key properties:
- 731 public instances across 11 repos, 4 languages (Python, JavaScript, TypeScript, Go)
- Minimum 10-line patches, average 107 lines across 4.1 files
- Each instance: a repo at a specific commit + problem statement + requirements + interface spec + test suite
- Human-verified and augmented to ensure resolvability
- Uses GPL/copyleft repos for contamination resistance

The 11 public repos: ansible/ansible, internetarchive/openlibrary, flipt-io/flipt, qutebrowser/qutebrowser, gravitational/teleport, protonmail/webclients, future-architect/vuls, navidrome/navidrome, element-hq/element, NodeBB/NodeBB, tutao/tutanota

## The Comparison We're Making

**GPT-5 vs Claude Sonnet 4.5**, both run on the **leaderboard configuration** (250 turn limit, no cost limit) using the **SWE-Agent scaffold** with the same prompt.

These are the best apples-to-apples trajectories available:
- Both from October 13, 2025
- 729 shared instances (730 Claude, 729 GPT-5, off by 1 flipt instance)
- Same SWE-Agent version (1.1.0), same tools, same evaluation

Model configs (from `replay_config.agent.model` in trajectories):
- GPT-5: `litellm_proxy/openai/gpt-5`, temperature=1.0, `completion_kwargs: {reasoning_effort: "high"}`
- Claude: `litellm_proxy/anthropic/claude-sonnet-4-5`, temperature=1.0, no special kwargs

Release dates: GPT-5 released August 7, 2025. Claude Sonnet 4.5 released September 29, 2025 (~2 months newer).

Overall resolve rates (from eval_results.json on GitHub):
- GPT-5: 265/729 = 36.4%
- Claude Sonnet 4.5: 319/729 = 43.8%
- Both solved: 206, Only GPT-5: 59, Only Claude: 113, Neither: 351

## Trajectory Data Location

**S3 bucket**: `s3://scaleapi-results/swe-bench-pro/` (public, use `--no-sign-request`)

Two directories per model:
- `traj/{instance_id}/{instance_id}.traj` — the full agent trajectory (JSON, ~5-50MB each)
- `eval/{instance_id}/_output.json` + `_patch.diff` — test results and generated patch

Full dataset sizes:
- `gpt-5-250-turns-10132025/`: 2,186 objects, 10.8 GB
- `claude-45sonnet-10132025/`: 2,180 objects, 12.5 GB
- **Total: ~23 GB, ~1,460 .traj files**

Download command:
```bash
aws s3 cp s3://scaleapi-results/swe-bench-pro/gpt-5-250-turns-10132025/ data/gpt5/ --recursive --no-sign-request
aws s3 cp s3://scaleapi-results/swe-bench-pro/claude-45sonnet-10132025/ data/claude45/ --recursive --no-sign-request
```

Eval results (for resolved status cross-reference):
```bash
curl -o eval_results_gpt5.json https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/traj/gpt-5-250-turns-10132025/eval_results.json
curl -o eval_results_claude45.json https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/traj/claude-45sonnet-10132025/eval_results.json
```

## Trajectory File Format

Each `.traj` file is a JSON dict with keys:

```
trajectory    — list of steps, each with: action, observation, response, thought, execution_time, state, query, extra_info
history       — list of chat messages: {role, content, agent, message_type, tool_calls?, tool_call_ids?, action?, thought?}
info          — {swe_agent_hash, swe_agent_version, submission, exit_status, model_stats, edited_files30/50/70}
replay_config — JSON string with full agent/model config
environment   — instance_id string
```

### Key Fields

**`info.model_stats`** — the primary cost/token data:
```json
{
  "instance_cost": 1.103788,      // total API cost, reliable, from litellm.cost_calculator.completion_cost(response)
  "tokens_sent": 872960,          // total input tokens, reliable, from litellm.utils.token_counter(messages)
  "tokens_received": 0,           // UNDERCOUNTS — see below
  "api_calls": 42                 // number of LLM roundtrips
}
```

**`trajectory[].execution_time`** — seconds for tool/command execution (NOT LLM inference). The only timing data available. No wall-clock time, no LLM latency, no timestamps anywhere.

**`trajectory[].action`** — raw action string, classifiable by prefix into: bash, view, edit, create, search_find, submit.

**`info.submission`** — the generated patch (string). Empty if agent didn't submit.

**`info.exit_status`** — "submitted", "submitted (exit_command_timeout)", or other.

## The tokens_received Bug (Critical Finding)

`tokens_received` is computed in SWE-Agent source (`sweagent/agent/models.py`, lines 761-767 of upstream SWE-agent/SWE-agent):

```python
output = choices[i].message.content or ""
output_tokens += litellm.utils.token_counter(
    text=output,
    model=...,
)
```

It counts **only `message.content` tokens**. Tool call arguments (`choices[i].message.tool_calls[].function.arguments`) are extracted separately but **never added to the token count**.

Impact:
- **GPT-5** responds with almost pure tool calls (empty content). `tokens_received` reports 0 or near-0 on most instances. Actual output via tiktoken: ~8,500 tokens/instance average.
- **Claude** responds with content + thought + tool calls. `tokens_received` captures the content portion but misses tool call args. Actual output: ~17,400 tokens/instance average.
- **Undercount factor**: ~7-8× for both models.

`instance_cost` is NOT affected — it uses `litellm.cost_calculator.completion_cost(response)` which operates on the full API response.

**Fix**: Use tiktoken (`cl100k_base`) to count tokens across all assistant output: `content` + `thought` (when different from content) + `tool_calls[].function.arguments`. This is what `extract_stats.py` does in the `output_tokens` field.

## The Extract Script

`extract_stats.py` processes trajectory files and outputs per-trajectory JSON with:

- **Cost**: `model_stats.instance_cost` (passthrough, reliable)
- **Tokens**: `model_stats.tokens_sent`, `.api_calls`, derived `tokens_per_call`
- **Output tokens (corrected)**: `output_tokens.{content, thought, tool_call_args, total}` via tiktoken
- **Tool time**: `tool_time.{total_seconds, mean_seconds, median_seconds, max_seconds, steps_over_10s, steps_over_60s}`
- **Actions**: `actions.{bash, view, edit, create, search_find, submit, other}`
- **Content volume**: `content.{observation_chars, action_chars, thought_chars, response_chars}`
- **Outcome**: `exit_status`, `submitted`, `patch_chars`, `resolved` (from eval_results.json)
- **Config**: `config.{model_name, temperature, per_instance_call_limit, completion_kwargs}`

Usage:
```bash
python3 extract_stats.py \
    data/gpt5 data/claude45 \
    --eval-results eval_results_gpt5.json eval_results_claude45.json \
    -o full_stats.json
```

Performance: ~70ms per trajectory. Full dataset (1,460 files, 23GB): ~2 minutes processing, 10-15 minutes download.

## Sample Results (8 instances × 2 models)

Sample selection: 2 instances each of {both solved, GPT-5 only, Claude only, neither}, across NodeBB (JS) and ansible (Python). Trajectories in `sample/gpt5/` and `sample/claude45/`.

Results in `sample_stats.json` (raw) and `sample_summary.txt` (tabular). Key numbers:

### Cost
- GPT-5 mean: $3.05/instance, Claude: $12.41 (4.1× more)
- Cost per resolve: GPT-5 $6.09, Claude $24.81

### Tokens
- Input tokens (mean): GPT-5 5.3M, Claude 3.0M — GPT-5 sends 1.8× more
- Tokens per call: GPT-5 53K, Claude 36K — GPT-5 carries heavier context
- Output tokens tiktoken (mean): GPT-5 8,540, Claude 17,443 — Claude produces 2× more
- tokens_received (mean): GPT-5 1,114, Claude 3,545 — broken metric, don't use

### Tool Execution Time
- Mean total: GPT-5 201s, Claude 64s — GPT-5 3.2× more
- Median per step: both ~0.5s
- GPT-5 has heavy tail: 2 calls >120s (recursive greps, full test suites)

### Actions
- GPT-5: 48 bash, 30 view, 12 edit, 0 search per instance
- Claude: 34 bash, 17 view, 15 edit, 11 search per instance
- GPT-5 always starts with `str_replace_editor view /app` (broad directory listing)
- Claude always starts with `find /app -type f -name "*.py" | grep -E "(relevant_terms)"` (targeted search)

### Content Volume
- Observations consumed: GPT-5 309K chars, Claude 204K (GPT-5 reads 1.5× more)
- Actions produced: GPT-5 33K chars, Claude 53K (Claude writes 1.6× more)
- Thoughts produced: GPT-5 5K chars, Claude 15K (Claude reasons 3× more visibly)
- Patch size: GPT-5 27K chars, Claude 56K (Claude produces 2× larger patches)

## Cost vs List Pricing (Important Caveat)

The observed costs **do not match public list pricing** for either model. Both models are accessed through Scale's internal litellm proxy (`litellm.ml.scaleinternal.com`), which likely has negotiated enterprise rates or miscalibrated cost calculators.

- GPT-5 observed: $24.37 total. At list $2.50/1M input: would be $107. **Paying ~23% of list.**
- Claude observed: $99.26 total. At Sonnet list $3/1M input: would be $74. **Paying ~133% of list.**

The 4.1× cost ratio is **an artifact of Scale's internal pricing**, not generalizable. At list prices ($2.50/$15 vs $3/$15), the ratio would flip to ~0.7× (Claude cheaper, because GPT-5 sends 1.8× more input).

**For a published study: report token volumes, not dollar costs.** Token volumes are intrinsic to model behavior. Costs are a function of opaque pricing that changes.

## What We Can't Measure

1. **Wall-clock time per instance** — not recorded anywhere. No timestamps on messages, no start/end times. S3 file modification dates are bulk upload times, not completion times.
2. **LLM inference latency per call** — not recorded. We only have tool execution time.
3. **GPT-5 hidden reasoning tokens** — with `reasoning_effort: high`, GPT-5 does internal chain-of-thought that's billed but never appears in the response. These tokens are invisible: not in `content`, not in `tool_calls`, not anywhere in the trajectory. `instance_cost` includes them, but we can't count them. This means GPT-5's true output token count is higher than what tiktoken measures.
4. **Input/output cost breakdown** — `instance_cost` is a blended total. We know input tokens and visible output tokens, but can't decompose the dollar amount.

## What Matters for the Full Analysis

The token and tool-time differences compound. GPT-5's explore-everything approach (bash + view heavy) generates more observation output, which accumulates in conversation history, which inflates input tokens on every subsequent call. It's quadratic-ish growth. Claude's targeted search-then-edit approach keeps context leaner.

For the full 729-instance analysis, the interesting cuts are:
- **By repo/language**: The sample only covers NodeBB (JS) and ansible (Python). The full set has Go repos (flipt, vuls, navidrome, teleport) and TypeScript (element, protonmail, tutao) which may show different patterns.
- **By outcome category**: Both-solved vs one-only vs neither — do the models spend differently when they succeed vs fail?
- **By instance difficulty**: Proxy via patch size, step count, or tokens spent.
- **Distribution shapes**: Means can hide bimodal behavior. GPT-5's tool time is bimodal (mostly <1s, occasional >100s).

## Files in This Directory

```
swe_bench_pro_paper.pdf          — the paper
swe_bench_pro_paper.txt          — paper converted to text
extract_stats.py                 — main extraction script
sample_stats.json                — extracted stats for 8 instances × 2 models
sample_summary.txt               — tabular summary of sample stats
sample_instances.json            — which 8 instances were sampled and why
eval_results_gpt5.json           — resolved/not for all 729 GPT-5 instances
eval_results_claude45.json       — resolved/not for all 730 Claude instances
sample/gpt5/                     — 8 GPT-5 sample trajectories + eval files
sample/claude45/                 — 8 Claude sample trajectories + eval files
trajectories/                    — earlier single-instance exploration (can ignore)
```
