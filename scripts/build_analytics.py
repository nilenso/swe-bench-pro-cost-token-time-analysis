#!/usr/bin/env python3
"""
Build an HTML analytics page comparing GPT-5 vs Claude 4.5 trajectories.

Charts:
  1. High-level letter frequencies (bar chart, side-by-side)
  2. Low-level intent frequencies (bar chart, side-by-side)
  3. Top 2-letter transitions / bigrams (bar chart, side-by-side)
  4. Typical trajectory shape visualisations

Usage:
  python scripts/build_analytics.py --data-root data --output docs/analytics.html
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import classify_intent as ci

# Single-letter codes for transition matrices and sequence analysis only
HIGH_LEVEL_LETTER = {
    "read": "R",
    "search": "S",
    "reproduce": "P",
    "edit": "E",
    "verify": "V",
    "git": "G",
    "housekeeping": "H",
    "failed": "X",
    "other": "O",
}
# Reverse: letter → short display name (used in transition matrices)
LETTER_TO_NAME = {v: k for k, v in HIGH_LEVEL_LETTER.items()}

# Colors keyed by the short name (not letter)
HIGH_LEVEL_COLORS = {
    "read": "#5a7d9a",
    "search": "#5a7d9a",
    "reproduce": "#b0956a",
    "edit": "#4a8a5a",
    "verify": "#b56a50",
    "git": "#3a8a8a",
    "housekeeping": "#3a8a8a",
    "failed": "#a05050",
    "other": "#888",
}
# Also keyed by letter for transition matrix rendering
LETTER_COLORS = {HIGH_LEVEL_LETTER[k]: v for k, v in HIGH_LEVEL_COLORS.items()}


def parse_repo(instance_id: str) -> str:
    core = instance_id
    if core.startswith("instance_"):
        core = core[len("instance_"):]
    if "__" not in core:
        return "unknown/unknown"
    org, rest = core.split("__", 1)
    m = re.search(r"-(?:[0-9a-f]{7,40})(?:-v.*)?$", rest)
    repo = rest[: m.start()] if m else rest
    return f"{org}/{repo}"


def collect_files(data_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for model in ("gpt5", "claude45"):
        base = data_root / model / "traj"
        if not base.exists():
            continue
        for p in sorted(base.glob("*/*.traj")):
            out.append((model, p))
    return out


CACHE_DIR = Path(__file__).parent.parent / ".cache" / "analytics"


def _cache_key(path_str: str) -> str:
    """Fast cache key from file path, size, and mtime."""
    st = os.stat(path_str)
    return f"{path_str}:{st.st_size}:{int(st.st_mtime)}"


def _process_one_file(args: tuple[str, str]) -> dict:
    """Process a single .traj file. Top-level function for pickling."""
    model, path_str = args

    # Check cache
    import hashlib
    key = hashlib.md5(_cache_key(path_str).encode()).hexdigest()
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_bytes())
        cached["model"] = model
        return cached

    data = ci._load_json(path_str)
    trajectory = data.get("trajectory", [])
    if not trajectory:
        return {"model": model, "empty": True}

    base_intents = ci.classify_trajectory(trajectory)
    hierarchical = ci.classify_hierarchical_layer(base_intents)

    highs = [h.split(".", 1)[0] for h in hierarchical]
    high_letters = [HIGH_LEVEL_LETTER.get(h, "?") for h in highs]
    high_seq = "".join(high_letters)

    high_c = Counter(high_letters)
    low_c = Counter(base_intents)
    bigram_c = Counter()
    for i in range(len(high_letters) - 1):
        bigram_c[high_letters[i] + high_letters[i + 1]] += 1

    # Phase profile: 20 bins
    phase: dict[str, list[float]] = {}
    n = len(high_letters)
    if n >= 5:
        bins = 20
        for letter in HIGH_LEVEL_LETTER.values():
            counts_in_bin = []
            for b in range(bins):
                start = int(b * n / bins)
                end = int((b + 1) * n / bins)
                segment = high_letters[start:end]
                counts_in_bin.append(segment.count(letter) / len(segment) if segment else 0.0)
            phase[letter] = counts_in_bin

    result = {
        "empty": False,
        "high_c": dict(high_c),
        "low_c": dict(low_c),
        "bigram_c": dict(bigram_c),
        "high_seq": high_seq,
        "steps": len(base_intents),
        "phase": phase,
    }

    # Write cache
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result))
    except OSError:
        pass

    result["model"] = model
    return result


INTENT_DISPLAY_NAMES = {
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
    # other
    "submit": "submit patch",
    "empty": "empty action (timeout)",
    "echo": "echo / printf",
    "bash-other": "other shell command",
    "undo-edit": "undo edit",
}


def _load_intent_descriptions() -> dict[str, str]:
    desc_path = Path(__file__).parent / "intent_descriptions.json"
    with open(desc_path) as f:
        return json.load(f)


def build_payload(data_root: Path) -> dict:
    files = collect_files(data_root)

    # Per-model accumulators
    high_counts: dict[str, Counter] = {"gpt5": Counter(), "claude45": Counter()}
    low_counts: dict[str, Counter] = {"gpt5": Counter(), "claude45": Counter()}
    bigram_counts: dict[str, Counter] = {"gpt5": Counter(), "claude45": Counter()}

    all_high_seqs: dict[str, list[str]] = {"gpt5": [], "claude45": []}
    step_counts: dict[str, list[int]] = {"gpt5": [], "claude45": []}
    phase_profiles: dict[str, dict[str, list[list[float]]]] = {
        "gpt5": {l: [] for l in HIGH_LEVEL_LETTER.values()},
        "claude45": {l: [] for l in HIGH_LEVEL_LETTER.values()},
    }

    tasks = [(model, str(path)) for model, path in files]
    max_workers = min(8, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_process_one_file, tasks, chunksize=32):
            if result["empty"]:
                continue
            model = result["model"]
            high_counts[model].update(result["high_c"])
            low_counts[model].update(result["low_c"])
            bigram_counts[model].update(result["bigram_c"])
            all_high_seqs[model].append(result["high_seq"])
            step_counts[model].append(result["steps"])
            for letter, bins in result["phase"].items():
                phase_profiles[model][letter].append(bins)

    # Aggregate phase profiles: average across trajectories
    avg_phase: dict[str, dict[str, list[float]]] = {}
    for model in ("gpt5", "claude45"):
        avg_phase[model] = {}
        for letter in HIGH_LEVEL_LETTER.values():
            profiles = phase_profiles[model][letter]
            if not profiles:
                avg_phase[model][letter] = [0.0] * 20
                continue
            bins = len(profiles[0])
            avg_phase[model][letter] = [
                sum(p[b] for p in profiles) / len(profiles) for b in range(bins)
            ]

    # Normalize counts to proportions for fair comparison
    def to_proportions(counter: Counter) -> dict[str, float]:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counter.items()}

    # Build transition matrices (row=from, col=to) as proportions
    ordered_letters = ["R", "S", "P", "E", "V", "G", "H", "X", "O"]
    bigram_matrix: dict[str, list[list[float]]] = {}
    for model in ("gpt5", "claude45"):
        total_bg = sum(bigram_counts[model].values())
        if total_bg == 0:
            bigram_matrix[model] = [[0.0] * len(ordered_letters) for _ in ordered_letters]
            continue
        matrix = []
        for fr in ordered_letters:
            row = []
            for to in ordered_letters:
                row.append(bigram_counts[model].get(fr + to, 0) / total_bg)
            matrix.append(row)
        bigram_matrix[model] = matrix

    # Also keep top bigrams list for reference
    top_bigrams_set = set()
    for model in ("gpt5", "claude45"):
        for bg, _ in bigram_counts[model].most_common(20):
            top_bigrams_set.add(bg)
    top_bigrams = sorted(top_bigrams_set, key=lambda b: -(bigram_counts["gpt5"].get(b, 0) + bigram_counts["claude45"].get(b, 0)))

    # Get all intents from displayed categories (exclude failed/X)
    displayed_categories = {"read", "search", "reproduce", "edit", "verify", "git", "housekeeping", "other"}
    top_low_set = set()
    for intent, high in ci.INTENT_TO_HIGH_LEVEL.items():
        if high in displayed_categories:
            if low_counts["gpt5"].get(intent, 0) + low_counts["claude45"].get(intent, 0) > 0:
                top_low_set.add(intent)
    top_low = sorted(top_low_set, key=lambda i: -(low_counts["gpt5"].get(i, 0) + low_counts["claude45"].get(i, 0)))

    # Step count distributions (binned)
    def bin_steps(counts, bin_size=5):
        if not counts:
            return {}
        bins = Counter()
        for c in counts:
            b = (c // bin_size) * bin_size
            bins[b] += 1
        return dict(sorted(bins.items()))

    return {
        "high_counts": {m: dict(high_counts[m].most_common()) for m in ("gpt5", "claude45")},
        "high_proportions": {
            m: {LETTER_TO_NAME.get(k, k): v for k, v in to_proportions(high_counts[m]).items()}
            for m in ("gpt5", "claude45")
        },
        "low_proportions": {m: to_proportions(low_counts[m]) for m in ("gpt5", "claude45")},
        "top_low_intents": top_low,
        "bigram_matrix": bigram_matrix,
        "bigram_letters": ordered_letters,
        "avg_phase": avg_phase,
        "step_dist": {m: bin_steps(step_counts[m]) for m in ("gpt5", "claude45")},
        "num_trajs": {m: len(all_high_seqs[m]) for m in ("gpt5", "claude45")},
        "high_level_letter": HIGH_LEVEL_LETTER,
        "letter_to_name": LETTER_TO_NAME,
        "name_to_letter": HIGH_LEVEL_LETTER,
        "letter_colors": LETTER_COLORS,
        "name_colors": HIGH_LEVEL_COLORS,
        "intent_to_category": dict(ci.INTENT_TO_HIGH_LEVEL),
        "intent_descriptions": _load_intent_descriptions(),
        "intent_display_names": INTENT_DISPLAY_NAMES,
    }


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>GPT-5 vs Claude 4.5 — Trajectory Analytics</title>
  <style>
    :root {{
      --bg: #fffff8;
      --panel: #fffff8;
      --muted: #777;
      --text: #333;
      --accent: #5a7d9a;
      --border: #ddd;
      --claude: #b8785e;
      --gpt: #6a8da8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; font-weight: 400; letter-spacing: -0.3px; }}
    .subtitle {{ color: var(--muted); font-size: 14px; margin-bottom: 36px; font-style: italic; }}
    h2 {{
      font-size: 16px;
      font-weight: 400;
      font-style: italic;
      margin: 44px 0 6px 0;
      padding-top: 22px;
      border-top: 1px solid #e0e0e0;
      color: var(--text);
    }}
    .chart-desc {{ color: var(--muted); font-size: 12.5px; margin-bottom: 16px; }}
    .legend {{
      display: flex; gap: 20px; margin-bottom: 16px; font-size: 12.5px;
    }}
    .legend-item {{
      display: flex; align-items: center; gap: 6px;
    }}
    .legend-swatch {{
      width: 12px; height: 12px; border-radius: 2px;
    }}
    .chart-wrapper {{
      padding: 16px 0;
      margin-bottom: 6px;
      overflow-x: auto;
    }}
    canvas {{ display: block; }}

    /* Phase heatmap */
    .heatmap-grid {{
      display: grid;
      grid-template-columns: 80px repeat(10, 1fr);
      gap: 2px;
      font-size: 12px;
      margin-bottom: 10px;
    }}
    .heatmap-label {{
      display: flex; align-items: center; color: var(--text);
      font-size: 11px;
    }}
    .heatmap-cell {{
      height: 28px;
      border-radius: 3px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      color: rgba(255,255,255,0.7);
    }}
    .heatmap-header {{
      font-size: 11px; color: var(--muted); text-align: center;
    }}
    .model-tag {{
      display: inline-block;
      font-size: 12px;
      font-style: italic;
    }}
    .model-tag.gpt {{ color: var(--gpt); }}
    .model-tag.claude {{ color: var(--claude); }}

    .side-by-side {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 800px) {{
      .side-by-side {{ grid-template-columns: 1fr; }}
    }}
    .side-by-side .chart-wrapper {{ margin-bottom: 0; }}
    .side-label {{
      font-size: 13px; font-style: italic; margin-bottom: 10px; color: var(--text);
    }}

    /* Dumbbell chart */
    .dumbbell-row {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 8px;
      align-items: center;
      padding: 4px 0;
      border-bottom: 1px solid #eee;
    }}
    .dumbbell-row:last-child {{ border-bottom: none; }}
    .dumbbell-label {{
      text-align: right;
      font-size: 12px;
      color: var(--text);
    }}
    .dumbbell-track {{
      position: relative;
      height: 22px;
    }}
    .dumbbell-line {{
      position: absolute;
      top: 10px;
      height: 2px;
      border-radius: 1px;
    }}
    .dumbbell-dot {{
      position: absolute;
      top: 3px;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .dumbbell-val {{
      position: absolute;
      top: 2px;
      font-size: 10px;
      white-space: nowrap;
    }}
    .dumbbell-header {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 8px;
      font-size: 11px;
      color: var(--muted);
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 4px;
    }}
    .dumbbell-scale {{
      position: relative;
      height: 16px;
    }}
    .dumbbell-scale-tick {{
      position: absolute;
      top: 0;
      font-size: 9px;
      color: var(--muted);
      transform: translateX(-50%);
    }}

    /* Paired bar table */
    .paired-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
    }}
    .paired-table th {{
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 400;
      font-style: italic;
      color: var(--muted);
    }}
    .paired-table td {{
      padding: 0;
    }}
    .paired-table tr.paired-row {{
      border-bottom: 8px solid transparent;
    }}
    .paired-table tr.paired-row.zebra td {{
      background: rgba(0,0,0,0.015);
    }}
    .paired-table td.paired-name {{
      text-align: right;
      padding-right: 12px;
      color: var(--text);
      white-space: nowrap;
      width: 1%;
      vertical-align: middle;
    }}
    .paired-table td.paired-bars {{
      padding: 4px 4px;
    }}
    .paired-bar-row {{
      display: flex;
      align-items: center;
      height: 11px;
      gap: 4px;
    }}
    .paired-bar {{
      height: 11px;
      border-radius: 2px;
      opacity: 0.85;
    }}
    .paired-bar-val {{
      font-size: 9.5px;
      min-width: 24px;
      color: var(--muted);
    }}
    .paired-table .cat-header td {{
      padding: 14px 0 4px 0;
      font-weight: 400;
      font-style: italic;
      font-size: 13px;
      color: var(--text);
      letter-spacing: 0.3px;
    }}
    .cat-annotation {{
      font-style: normal;
      font-size: 11.5px;
      color: var(--muted);
      margin-top: 2px;
      line-height: 1.4;
    }}

    /* Stacked area legend */
    .stacked-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin-top: 10px;
      font-size: 11.5px;
      color: var(--text);
    }}
    .stacked-legend .item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .stacked-legend .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 2px;
      flex-shrink: 0;
    }}

    /* Transition matrix */
    .tmatrix {{
      display: inline-grid;
      gap: 2px;
      font-size: 11px;
      font-family: ui-monospace, monospace;
    }}
    .tmatrix .corner {{ }}
    .tmatrix .col-hdr {{
      text-align: center;
      color: var(--text);
      font-weight: 700;
      padding: 2px 0;
    }}
    .tmatrix .row-hdr {{
      display: flex; align-items: center; justify-content: flex-end;
      color: var(--text);
      font-weight: 700;
      padding-right: 6px;
    }}
    .tmatrix .cell {{
      width: 48px; height: 32px;
      border-radius: 3px;
      display: flex; align-items: center; justify-content: center;
      font-size: 10px;
      color: rgba(255,255,255,0.85);
      transition: transform 0.1s;
    }}
    .tmatrix .cell:hover {{
      transform: scale(1.15);
      z-index: 2;
      outline: 1px solid var(--text);
    }}
  </style>
</head>
<body>
<div class="container">
  <h1>GPT-5 vs Claude 4.5</h1>
  <div class="subtitle">Trajectory behaviour analytics &mdash; SWE-Bench Pro</div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--gpt)"></div>
      <span>GPT-5</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--claude)"></div>
      <span>Claude 4.5</span>
    </div>
  </div>

  <h2>1. High-Level Action Frequencies</h2>
  <p class="chart-desc">Proportion of steps in each high-level category. Normalised so models are comparable despite different step counts.</p>
  <div class="chart-wrapper">
    <div id="highPaired"></div>
  </div>

  <h2>2. Intent Comparison</h2>
  <p class="chart-desc">Frequency per 100 steps. <span style="color:var(--claude)">Claude 4.5</span> / <span style="color:var(--gpt)">GPT-5</span>.</p>
  <div class="chart-wrapper">
    <div id="heatTable"></div>
  </div>

  <h2>3. Trajectory Length Distribution</h2>
  <p class="chart-desc">How many steps each model typically takes per task.</p>
  <div class="chart-wrapper">
    <canvas id="stepDistChart" height="280"></canvas>
  </div>

  <h2>4. Typical Trajectory Shape</h2>
  <p class="chart-desc">Stacked area chart: how the mix of actions evolves from start to end of the average trajectory.</p>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag gpt">GPT-5</span></div>
    <canvas id="stackedGpt" height="200"></canvas>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag claude">Claude 4.5</span></div>
    <canvas id="stackedClaude" height="200"></canvas>
  </div>

  <h2>4b. Phase Profile — When Does Each Action Happen?</h2>
  <p class="chart-desc">Each trajectory is divided into 20 equal time-slices (0%–100%). The heatmap shows what proportion of steps in each slice belong to each category. Read left-to-right as "beginning → end of trajectory".</p>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag gpt">GPT-5</span></div>
    <div id="heatmapGpt"></div>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag claude">Claude 4.5</span></div>
    <div id="heatmapClaude"></div>
  </div>

</div>

<script>
const D = {payload_json};

// ── Helpers ──────────────────────────────────────────────
function getCtx(id) {{
  const c = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const cssW = c.parentElement.clientWidth - 40;
  const cssH = c.height;
  c.width = cssW * dpr;
  c.height = cssH * dpr;
  c.style.width = cssW + 'px';
  c.style.height = cssH + 'px';
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  return {{ canvas: c, ctx, w: cssW, h: cssH }};
}}

const CLAUDE_COLOR = '#b8785e';
const GPT_COLOR = '#6a8da8';
const MUTED = '#6b7280';
const TEXT = '#1a1a1a';

// Names for display (everywhere except transition matrices)
const CATEGORY_NAMES = ['read','search','reproduce','edit','verify','git','housekeeping'];
const NAME_COLORS = D.name_colors;

// Letters for transition matrices only
const LETTERS = Object.values(D.name_to_letter);
const LETTER_COLORS = D.letter_colors;
const LETTER_TO_NAME = D.letter_to_name;

// ── 1. High-Level Bar Chart ──────────────────────────────
function drawGroupedBar(canvasId, labels, gptVals, claudeVals, labelMap) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 50, right = 20, top = 20, bot = 60;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const n = labels.length;
  const groupW = plotW / n;
  const barW = groupW * 0.35;
  const maxVal = Math.max(...gptVals, ...claudeVals, 0.01);

  // Y axis
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = top + plotH - (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText((maxVal * i / 4 * 100).toFixed(1) + '%', left - 6, y + 4);
  }}

  for (let i = 0; i < n; i++) {{
    const x = left + i * groupW + groupW * 0.1;
    const gH = (gptVals[i] / maxVal) * plotH;
    const cH = (claudeVals[i] / maxVal) * plotH;

    ctx.fillStyle = GPT_COLOR;
    ctx.fillRect(x, top + plotH - gH, barW, gH);

    ctx.fillStyle = CLAUDE_COLOR;
    ctx.fillRect(x + barW + 2, top + plotH - cH, barW, cH);

    // Label
    ctx.fillStyle = TEXT;
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    const lx = x + barW + 1;
    const label = labels[i];
    ctx.fillText(label, lx, top + plotH + 16);
    if (labelMap && labelMap[label]) {{
      ctx.fillStyle = MUTED;
      ctx.font = '10px sans-serif';
      ctx.fillText(labelMap[label], lx, top + plotH + 30);
    }}
  }}
}}

// ── 2. Horizontal grouped bar (for long labels) ─────────
function drawHorizontalGroupedBar(canvasId, labels, gptVals, claudeVals) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 160, right = 30, top = 10, bot = 30;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const n = labels.length;
  const rowH = plotH / n;
  const barH = rowH * 0.35;
  const maxVal = Math.max(...gptVals, ...claudeVals, 0.001);

  // Vertical grid
  ctx.strokeStyle = '#e0e0e0'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const x = left + (i / 4) * plotW;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    ctx.fillText((maxVal * i / 4 * 100).toFixed(1) + '%', x, top + plotH + 16);
  }}

  for (let i = 0; i < n; i++) {{
    const y = top + i * rowH + rowH * 0.15;
    const gW = (gptVals[i] / maxVal) * plotW;
    const cW = (claudeVals[i] / maxVal) * plotW;

    ctx.fillStyle = GPT_COLOR;
    ctx.fillRect(left, y, gW, barH);

    ctx.fillStyle = CLAUDE_COLOR;
    ctx.fillRect(left, y + barH + 1, cW, barH);

    ctx.fillStyle = TEXT; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(labels[i], left - 8, y + rowH * 0.45);
  }}
}}

// ── Draw charts ──────────────────────────────────────────

// 1. High-level paired bars
(function() {{
  const el = document.getElementById('highPaired');
  const cats = CATEGORY_NAMES;
  const maxVal = Math.max(...cats.map(c => Math.max(
    D.high_proportions.gpt5[c] || 0,
    D.high_proportions.claude45[c] || 0
  )));

  function barPct(v) {{ return (v / maxVal * 100).toFixed(1); }}

  let html = `<table class="paired-table">
    <thead><tr>
      <th></th>
      <th style="text-align:left;padding-left:4px">
        <span style="color:${{CLAUDE_COLOR}}">Claude</span>
        <span style="color:var(--muted);padding:0 6px">/</span>
        <span style="color:${{GPT_COLOR}}">GPT</span>
        <span style="color:var(--muted);font-weight:400;padding-left:8px">% of steps</span>
      </th>
    </tr></thead><tbody>`;

  let rowIdx = 0;
  for (const cat of cats) {{
    const g = (D.high_proportions.gpt5[cat] || 0) * 100;
    const c = (D.high_proportions.claude45[cat] || 0) * 100;
    const gap = Math.abs(g - c);
    const gBold = g > c && gap >= 0.3 ? 'font-weight:700' : '';
    const cBold = c > g && gap >= 0.3 ? 'font-weight:700' : '';
    const zebra = rowIdx % 2 === 1 ? ' zebra' : '';

    html += `<tr class="paired-row${{zebra}}">
      <td class="paired-name">${{cat}}</td>
      <td class="paired-bars">
        <div class="paired-bar-row">
          <div class="paired-bar" style="width:${{barPct(c / 100)}}%;background:${{CLAUDE_COLOR}}"></div>
          <span class="paired-bar-val" style="color:${{CLAUDE_COLOR}};${{cBold}}">${{c.toFixed(1)}}</span>
        </div>
        <div class="paired-bar-row">
          <div class="paired-bar" style="width:${{barPct(g / 100)}}%;background:${{GPT_COLOR}}"></div>
          <span class="paired-bar-val" style="color:${{GPT_COLOR}};${{gBold}}">${{g.toFixed(1)}}</span>
        </div>
      </td>
    </tr>`;
    rowIdx++;
  }}

  html += '</tbody></table>';
  el.innerHTML = html;
}})();

// 2. Paired horizontal bars
(function() {{
  const el = document.getElementById('heatTable');
  const intents = D.top_low_intents;
  const catMap = D.intent_to_category || {{}};
  const catOrder = ['read','search','reproduce','edit','verify','git','housekeeping','other'];

  const catAnnotations = {{
    'read': 'GPT reads 1.6x more, especially full files. Over a third of its steps are reading.',
    'search': 'Both grep at similar rates. Claude also runs find | grep and finds files by name (9 per 100 steps vs 0.1 for GPT).',
    'reproduce': 'GPT writes and runs reproduction scripts 4x more often.',
    'edit': 'Similar edit rates overall. GPT also uses insert (1.9 per 100). Claude edits exclusively via str_replace.',
    'verify': 'Claude spends 28% of steps verifying (8x GPT), running test suites, writing test files, and compiling.',
    'git': 'Claude regularly runs git diff and git log to review its own changes. GPT uses git in 0.1% of steps.',
    'housekeeping': 'Claude cleans up with rm/mv/cp (2.7 per 100) and writes documentation files. GPT does 0.1 per 100.',
  }};

  const rows = intents.map(intent => {{
    const g = (D.low_proportions.gpt5[intent] || 0) * 100;
    const c = (D.low_proportions.claude45[intent] || 0) * 100;
    const cat = catMap[intent] || '?';
    return {{ intent, g, c, gap: Math.abs(g - c), cat }};
  }}).filter(r => r.g > 1.5 || r.c > 1.5);

  {{

    const grouped = {{}};
    for (const r of rows) {{
      if (!grouped[r.cat]) grouped[r.cat] = [];
      grouped[r.cat].push(r);
    }}
    for (const cat of Object.keys(grouped)) {{
      grouped[cat].sort((a, b) => b.gap - a.gap);
    }}

    const maxVal = Math.max(...rows.map(r => Math.max(r.g, r.c)), 1);
    function barPct(v) {{ return (v / maxVal * 100).toFixed(1); }}

    let html = `<table class="paired-table">
      <tbody>`;

    let rowIdx = 0;
    for (const cat of catOrder) {{
      if (!grouped[cat] || grouped[cat].length === 0) continue;
      const ann = catAnnotations[cat] || '';
      html += `<tr class="cat-header"><td colspan="2">${{cat}}${{ann ? `<div class="cat-annotation">${{ann}}</div>` : ''}}</td></tr>`;

      for (const r of grouped[cat]) {{
        const gBold = r.g > r.c && r.gap >= 0.3 ? 'font-weight:700' : '';
        const cBold = r.c > r.g && r.gap >= 0.3 ? 'font-weight:700' : '';
        const displayName = (D.intent_display_names || {{}})[r.intent] || r.intent;
        const zebra = rowIdx % 2 === 1 ? ' zebra' : '';

        html += `<tr class="paired-row${{zebra}}">
          <td class="paired-name" title="${{r.intent}}">${{displayName}}</td>
          <td class="paired-bars">
            <div class="paired-bar-row">
              <div class="paired-bar" style="width:${{barPct(r.c)}}%;background:${{CLAUDE_COLOR}}"></div>
              <span class="paired-bar-val" style="color:${{CLAUDE_COLOR}};${{cBold}}">${{r.c.toFixed(1)}}</span>
            </div>
            <div class="paired-bar-row">
              <div class="paired-bar" style="width:${{barPct(r.g)}}%;background:${{GPT_COLOR}}"></div>
              <span class="paired-bar-val" style="color:${{GPT_COLOR}};${{gBold}}">${{r.g.toFixed(1)}}</span>
            </div>
          </td>
        </tr>`;
        rowIdx++;
      }}
    }}

    html += '</tbody></table>';
    el.innerHTML = html;
  }}
}})();

// 3. Step distribution
(function() {{
  const {{ ctx, w, h }} = getCtx('stepDistChart');
  const left = 50, right = 20, top = 20, bot = 50;
  const allBins = new Set([
    ...Object.keys(D.step_dist.gpt5).map(Number),
    ...Object.keys(D.step_dist.claude45).map(Number),
  ]);
  const bins = [...allBins].sort((a,b) => a - b);
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const n = bins.length;
  const groupW = plotW / n;
  const barW = groupW * 0.38;

  const gptTotal = Object.values(D.step_dist.gpt5).reduce((a,b) => a+b, 0);
  const claudeTotal = Object.values(D.step_dist.claude45).reduce((a,b) => a+b, 0);

  const gptVals = bins.map(b => (D.step_dist.gpt5[b] || 0) / gptTotal);
  const claudeVals = bins.map(b => (D.step_dist.claude45[b] || 0) / claudeTotal);
  const maxVal = Math.max(...gptVals, ...claudeVals, 0.01);

  ctx.strokeStyle = '#e0e0e0'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = top + plotH - (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText((maxVal * i / 4 * 100).toFixed(0) + '%', left - 6, y + 4);
  }}

  for (let i = 0; i < n; i++) {{
    const x = left + i * groupW + groupW * 0.08;
    const gH = (gptVals[i] / maxVal) * plotH;
    const cH = (claudeVals[i] / maxVal) * plotH;
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = GPT_COLOR;
    ctx.fillRect(x, top + plotH - gH, barW, gH);
    ctx.fillStyle = CLAUDE_COLOR;
    ctx.fillRect(x + barW + 1, top + plotH - cH, barW, cH);
    ctx.globalAlpha = 1;

    if (i % 2 === 0 || n < 20) {{
      ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
      ctx.fillText(bins[i], x + barW, top + plotH + 14);
    }}
  }}
  ctx.fillStyle = MUTED; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('steps per trajectory', w / 2, h - 6);
}})();

// Color interpolation helper: blend between panel bg and target color
// t=0 → background, t=1 → full color
function lerpColor(hex, t) {{
  const bg = [255, 255, 248]; // #fffff8 (page bg)
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  const mr = Math.round(bg[0] + (r - bg[0]) * t);
  const mg = Math.round(bg[1] + (g - bg[1]) * t);
  const mb = Math.round(bg[2] + (b - bg[2]) * t);
  return `rgb(${{mr}},${{mg}},${{mb}})`;
}}

// Render a heatmap. normalize='row' or 'col'.
function drawHeatmap(containerId, model, normalize) {{
  const el = document.getElementById(containerId);
  const letters = ['R','S','P','E','V','G','H'];
  const bins = 20;

  // Renormalize: compute per-bin sum across shown letters
  const binSums = new Array(bins).fill(0);
  for (const l of letters) {{
    const v = D.avg_phase[model][l];
    for (let b = 0; b < bins; b++) binSums[b] += v[b];
  }}

  // Compute renormalized values (proportion of shown letters only)
  const renormed = {{}};
  for (const l of letters) {{
    const raw = D.avg_phase[model][l];
    renormed[l] = raw.map((v, b) => binSums[b] > 0 ? v / binSums[b] : 0);
  }}

  // Compute max per row or per column for scaling
  const maxPerRow = {{}};
  const maxPerCol = new Array(bins).fill(0);
  for (const l of letters) {{
    maxPerRow[l] = Math.max(...renormed[l]);
    for (let b = 0; b < bins; b++) {{
      if (renormed[l][b] > maxPerCol[b]) maxPerCol[b] = renormed[l][b];
    }}
  }}

  let html = '<div class="heatmap-grid" style="grid-template-columns: 80px repeat(20, 1fr)">';
  html += '<div></div>';
  for (let b = 0; b < bins; b++) {{
    html += `<div class="heatmap-header">${{b % 5 === 0 ? (b*5)+'%' : ''}}</div>`;
  }}

  for (const letter of letters) {{
    const name = LETTER_TO_NAME[letter] || letter;
    const color = LETTER_COLORS[letter];
    html += `<div class="heatmap-label">${{name}}</div>`;
    const vals = renormed[letter];
    for (let b = 0; b < bins; b++) {{
      const maxV = normalize === 'col' ? (maxPerCol[b] || 0.001) : (maxPerRow[letter] || 0.001);
      const ratio = vals[b] / maxV;
      // Use sqrt for better spread, but map 0 → 0 exactly
      const pctVal = vals[b] * 100;
      const t = pctVal < 0.5 ? 0 : 0.15 + Math.sqrt(ratio) * 0.85;
      const bg = lerpColor(color, t);
      const textColor = t > 0.5 ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.5)';
      html += `<div class="heatmap-cell" style="background:${{bg}};color:${{textColor}}">${{(vals[b]*100).toFixed(0)}}%</div>`;
    }}
  }}
  html += '</div>';
  el.innerHTML = html;
}}

// 5. Row-normalised
drawHeatmap('heatmapGpt', 'gpt5', 'row');
drawHeatmap('heatmapClaude', 'claude45', 'row');

// 5b. Column-normalised

// 6. Stacked area charts — 5 grouped bands, inline labels
function drawStackedArea(canvasId, model, annotations, markers) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 40, right = 20, top = 10, bot = 20;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const bins = 20;

  // 5 grouped bands: understand (R+S), reproduce (P), edit (E), verify (V), cleanup (G+H)
  const groups = [
    {{ name: 'understand', letters: ['R','S'], color: '#5a7d9a' }},
    {{ name: 'reproduce', letters: ['P'], color: '#b0956a' }},
    {{ name: 'edit',      letters: ['E'], color: '#4a8a5a' }},
    {{ name: 'verify',    letters: ['V'], color: '#b56a50' }},
    {{ name: 'cleanup',   letters: ['G','H'], color: '#3a8a8a' }},
  ];

  // Sum letters per group per bin
  const groupVals = groups.map(g => {{
    const summed = new Array(bins).fill(0);
    for (const l of g.letters) {{
      const vals = D.avg_phase[model][l];
      if (vals) for (let b = 0; b < bins; b++) summed[b] += vals[b];
    }}
    return summed;
  }});

  // Build stacked layers
  const stacked = [];
  let cumulative = new Array(bins).fill(0);
  for (let gi = 0; gi < groups.length; gi++) {{
    const layer = groupVals[gi].map((v, i) => cumulative[i] + v);
    stacked.push({{ group: groups[gi], bottom: [...cumulative], top: layer }});
    cumulative = layer;
  }}

  const maxes = cumulative;

  function xAt(i) {{ return left + (i / (bins - 1)) * plotW; }}
  function yAt(v, binIdx) {{
    const norm = maxes[binIdx] > 0 ? v / maxes[binIdx] : 0;
    return top + plotH - norm * plotH;
  }}

  // Draw from top layer down (so bottom layers paint over top)
  for (let s = stacked.length - 1; s >= 0; s--) {{
    const layer = stacked[s];
    ctx.fillStyle = layer.group.color;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (let i = 0; i < bins; i++) {{
      const x = xAt(i), y = yAt(layer.top[i], i);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    for (let i = bins - 1; i >= 0; i--) {{
      ctx.lineTo(xAt(i), yAt(layer.bottom[i], i));
    }}
    ctx.closePath();
    ctx.fill();
  }}
  ctx.globalAlpha = 1;

  // 50% vertical reference line
  const halfX = xAt(10); // bin 10 = 50%
  ctx.strokeStyle = 'rgba(0,0,0,0.15)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(halfX, top);
  ctx.lineTo(halfX, top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Vertical markers
  if (markers) {{
    for (const m of markers) {{
      const mx = xAt(m.at / 5); // at% maps to bin at/5
      ctx.strokeStyle = 'rgba(0,0,0,0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(mx, top);
      ctx.lineTo(mx, top + plotH);
      ctx.stroke();
      // Small label at top
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.font = '9px Palatino, Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(m.label, mx, top - 3);
    }}
  }}

  // X labels
  ctx.fillStyle = MUTED; ctx.font = '10px Palatino, Georgia, serif'; ctx.textAlign = 'center';
  for (let i = 0; i < bins; i++) {{
    if (i % 5 === 0) ctx.fillText(i * 5 + '%', xAt(i), top + plotH + 16);
  }}

  // Annotations: brackets inside the chart area
  if (annotations) {{
    ctx.lineWidth = 0.8;

    for (let ai = 0; ai < annotations.length; ai++) {{
      const a = annotations[ai];
      const fromBin = a.from / 5;
      const toBin = a.to / 5;
      const midBin = (fromBin + toBin) / 2;
      const x0 = xAt(fromBin);
      const x1 = xAt(toBin);

      // Find the vertical midpoint of the target band at the annotation's center
      const gi = groups.findIndex(g => g.color === a.color);
      const midBinInt = Math.round(midBin);
      const bandTop = yAt(stacked[gi].top[midBinInt], midBinInt);
      const bandBot = yAt(stacked[gi].bottom[midBinInt], midBinInt);
      const bandH = Math.abs(bandBot - bandTop);
      const midY = (bandTop + bandBot) / 2;

      // Bracket
      ctx.strokeStyle = '#fff';
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.moveTo(x0, midY); ctx.lineTo(x1, midY);
      ctx.moveTo(x0, midY - 4); ctx.lineTo(x0, midY + 4);
      ctx.moveTo(x1, midY - 4); ctx.lineTo(x1, midY + 4);
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Label: if band is thick enough, place inside. Otherwise, place above with a connecting line.
      ctx.font = '9.5px Palatino, Georgia, serif';
      ctx.textAlign = 'center';
      const labelX = (x0 + x1) / 2;

      if (bandH > 30) {{
        // Inside the band
        ctx.fillStyle = '#fff';
        ctx.globalAlpha = 0.9;
        ctx.fillText(a.label, labelX, midY - 7);
        ctx.globalAlpha = 1;
      }} else {{
        // Outside the band with a small connecting line
        // Place below if near the top edge, above otherwise
        const placeBelow = bandTop < top + 30;
        const labelY = placeBelow ? bandBot + 4 : bandTop - 8;
        const lineEnd = placeBelow ? bandBot : bandTop;
        const lineStart = placeBelow ? labelY - 10 : labelY + 3;
        ctx.fillStyle = '#fff';
        ctx.globalAlpha = 0.9;
        ctx.fillText(a.label, labelX, labelY);
        // Thin connecting line
        ctx.strokeStyle = 'rgba(0,0,0,0.2)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(labelX, lineStart);
        ctx.lineTo(labelX, lineEnd);
        ctx.stroke();
        ctx.lineWidth = 0.8;
      }}
    }}
  }}
}}

const gptAnnotations = [
  {{ from: 2, to: 50, label: 'understand', color: '#5a7d9a' }},
  {{ from: 50, to: 80, label: 'edit', color: '#4a8a5a' }},
  {{ from: 85, to: 98, label: 'verify', color: '#b56a50' }},
  {{ from: 78, to: 98, label: 'reproduce', color: '#b0956a' }},
];
const claudeAnnotations = [
  {{ from: 2, to: 30, label: 'understand', color: '#5a7d9a' }},
  {{ from: 30, to: 55, label: 'edit', color: '#4a8a5a' }},
  {{ from: 60, to: 80, label: 'verify', color: '#b56a50' }},
  {{ from: 80, to: 98, label: 'cleanup', color: '#3a8a8a' }},
];

drawStackedArea('stackedGpt', 'gpt5', gptAnnotations,
  [{{ at: 89, label: 'median last edit (89%)' }}]);
drawStackedArea('stackedClaude', 'claude45', claudeAnnotations,
  [{{ at: 61, label: 'median last edit (61%)' }}]);

</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", "-o", default="docs/analytics.html")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    payload = build_payload(data_root)
    html = render_html(payload)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out}")
    print(f"gpt5={payload['num_trajs']['gpt5']} claude45={payload['num_trajs']['claude45']}")


if __name__ == "__main__":
    main()
