#!/usr/bin/env python3
"""
Build an HTML analytics page comparing model trajectories.

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
import sys
from pathlib import Path

# Add project root to path for analysis package
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.orchestrate import process_all
from analysis.aggregate import build_analytics_payload


def build_payload(data_root: Path) -> dict:
    results = process_all(data_root)
    for model, data in sorted(results.items()):
        print(f"  {model}: {len(data)} trajectories")
    return build_analytics_payload(results)


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Trajectory Analytics — SWE-Bench Pro</title>
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
      --glm: #9a6a9a;
      --gemini: #6a9a6a;
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

    .model-tag {{
      display: inline-block;
      font-size: 12px;
      font-style: italic;
    }}
    .model-tag.gpt {{ color: var(--gpt); }}
    .model-tag.claude {{ color: var(--claude); }}
    .model-tag.glm {{ color: var(--glm); }}
    .model-tag.gemini {{ color: var(--gemini); }}

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
    .stacked-panel-header {{
      display: flex;
      align-items: baseline;
      gap: 14px;
      margin-bottom: 2px;
    }}
    .stacked-panel-header .model-tag {{
      font-size: 13px;
    }}
    .stacked-panel-header .panel-subhead {{
      font-size: 11.5px;
      font-style: italic;
      color: var(--muted);
    }}
    #stackedPanels .chart-wrapper {{
      padding: 4px 0;
      margin-bottom: 0;
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
  <h1>Trajectory Analytics</h1>
  <div class="subtitle">SWE-Bench Pro &mdash; multi-model comparison</div>

  <div class="legend" id="topLegend"></div>

  <h2>1. High-Level Action Frequencies</h2>
  <p class="chart-desc">Proportion of steps in each high-level category. Normalised so models are comparable despite different step counts.</p>
  <div class="chart-wrapper">
    <div id="highPaired"></div>
  </div>

  <h2>2. Intent Comparison</h2>
  <p class="chart-desc">Frequency per 100 steps, compared across all models.</p>
  <div class="chart-wrapper">
    <div id="heatTable"></div>
  </div>

  <h2>3. Steps per trajectory, by model</h2>
  <p class="chart-desc">Cumulative share of runs that finished within N steps. Dashed line marks the 250-step cap.</p>
  <p class="chart-desc" id="stepDistDesc"></p>
  <div class="chart-wrapper">
    <canvas id="stepDistChart" height="320"></canvas>
  </div>

  <h2>4. Typical Trajectory Shape</h2>
  <p class="chart-desc">Stacked area chart: how the mix of actions evolves from start to end of the average trajectory. Panels ordered by resolve rate (descending).</p>
  <div id="stackedPanels"></div>


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

const MODEL_COLORS = D.model_colors;
const MODEL_NAMES = D.model_display_names;
// Single source of truth for model order: resolve rate, descending.
// Every section consumes ALL_MODELS, so they all line up.
const ALL_MODELS = [...D.models].sort(
  (a, b) => (D.resolve_rate[b] ?? -Infinity) - (D.resolve_rate[a] ?? -Infinity)
);
const tagClass = {{
  'gpt5': 'gpt',
  'claude45': 'claude',
  'glm45': 'glm',
  'gemini25pro': 'gemini',
}};

(function() {{
  const el = document.getElementById('topLegend');
  if (!el) return;
  el.innerHTML = ALL_MODELS.map(m => (
    `<div class="legend-item">` +
      `<div class="legend-swatch" style="background:${{MODEL_COLORS[m]}}"></div>` +
      `<span>${{MODEL_NAMES[m]}}</span>` +
    `</div>`
  )).join('');
}})();
const CLAUDE_COLOR = '#b8785e';
const GPT_COLOR = '#6a8da8';
const GLM_COLOR = '#6a9a6a';
const GEMINI_COLOR = '#9a6a9a';
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
  const maxVal = Math.max(...cats.map(c =>
    Math.max(...ALL_MODELS.map(m => D.high_proportions[m][c] || 0))
  ));

  function barPct(v) {{ return (v / maxVal * 100).toFixed(1); }}

  let html = `<table class="paired-table">
    <thead><tr>
      <th></th>
      <th style="text-align:left;padding-left:4px">
        ${{ALL_MODELS.map(m => `<span style="color:${{MODEL_COLORS[m]}}">${{MODEL_NAMES[m]}}</span>`).join(' <span style="color:var(--muted);padding:0 4px">/</span> ')}}
        <span style="color:var(--muted);font-weight:400;padding-left:8px">% of steps</span>
      </th>
    </tr></thead><tbody>`;

  let rowIdx = 0;
  for (const cat of cats) {{
    const vals = ALL_MODELS.map(m => (D.high_proportions[m][cat] || 0) * 100);
    const best = Math.max(...vals);
    const zebra = rowIdx % 2 === 1 ? ' zebra' : '';

    html += `<tr class="paired-row${{zebra}}">
      <td class="paired-name">${{cat}}</td>
      <td class="paired-bars">`;
    for (let mi = 0; mi < ALL_MODELS.length; mi++) {{
      const m = ALL_MODELS[mi];
      const v = vals[mi];
      const bold = v === best && best >= 0.3 ? 'font-weight:700' : '';
      html += `<div class="paired-bar-row">
          <div class="paired-bar" style="width:${{barPct(v / 100)}}%;background:${{MODEL_COLORS[m]}}"></div>
          <span class="paired-bar-val" style="color:${{MODEL_COLORS[m]}};${{bold}}">${{v.toFixed(1)}}</span>
        </div>`;
    }}
    html += `</td>
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

  const rows = intents.map(intent => {{
    const vals = {{}};
    let maxV = 0;
    for (const m of ALL_MODELS) {{
      vals[m] = (D.low_proportions[m][intent] || 0) * 100;
      if (vals[m] > maxV) maxV = vals[m];
    }}
    const cat = catMap[intent] || '?';
    return {{ intent, vals, maxV, cat }};
  }}).filter(r => r.maxV > 1.5);

  {{

    const grouped = {{}};
    for (const r of rows) {{
      if (!grouped[r.cat]) grouped[r.cat] = [];
      grouped[r.cat].push(r);
    }}
    for (const cat of Object.keys(grouped)) {{
      grouped[cat].sort((a, b) => b.maxV - a.maxV);
    }}

    const maxVal = Math.max(...rows.map(r => r.maxV), 1);
    function barPct(v) {{ return (v / maxVal * 100).toFixed(1); }}

    let html = `<table class="paired-table">
      <tbody>`;

    let rowIdx = 0;
    for (const cat of catOrder) {{
      if (!grouped[cat] || grouped[cat].length === 0) continue;
      html += `<tr class="cat-header"><td colspan="2">${{cat}}</td></tr>`;

      for (const r of grouped[cat]) {{
        const displayName = (D.intent_display_names || {{}})[r.intent] || r.intent;
        const zebra = rowIdx % 2 === 1 ? ' zebra' : '';
        const best = Math.max(...ALL_MODELS.map(m => r.vals[m]));

        html += `<tr class="paired-row${{zebra}}">
          <td class="paired-name" title="${{r.intent}}">${{displayName}}</td>
          <td class="paired-bars">`;
        for (const m of ALL_MODELS) {{
          const v = r.vals[m];
          const bold = v === best && best >= 0.3 ? 'font-weight:700' : '';
          html += `<div class="paired-bar-row">
              <div class="paired-bar" style="width:${{barPct(v)}}%;background:${{MODEL_COLORS[m]}}"></div>
              <span class="paired-bar-val" style="color:${{MODEL_COLORS[m]}};${{bold}}">${{v.toFixed(1)}}</span>
            </div>`;
        }}
        html += `</td>
        </tr>`;
        rowIdx++;
      }}
    }}

    html += '</tbody></table>';
    el.innerHTML = html;
  }}
}})();

// 3. Step distribution — overlaid ECDFs
(function() {{
  const {{ ctx, w, h }} = getCtx('stepDistChart');
  const left = 56, top = 22, bot = 46;
  const xMax = 250;

  // Measure right-edge labels so the right margin always fits them
  ctx.font = '11px sans-serif';
  const sampleLabels = ALL_MODELS.map(m => `${{MODEL_NAMES[m]}} · 999 steps`);
  const maxLabelW = Math.max(...sampleLabels.map(s => ctx.measureText(s).width));
  const right = Math.ceil(maxLabelW) + 24;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const xPx = x => left + (x / xMax) * plotW;
  const yPx = y => top + (1 - y) * plotH;

  // Build ECDF points and censored fraction per model from binned step counts
  const cdfs = {{}};
  const censored = {{}};
  for (const m of ALL_MODELS) {{
    const entries = Object.entries(D.step_dist[m])
      .map(([k, v]) => [Number(k), v])
      .sort((a, b) => a[0] - b[0]);
    const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
    let cum = 0, beforeCap = 0;
    const pts = [[0, 0]];
    for (const [bin, count] of entries) {{
      if (bin < xMax) beforeCap += count;
      cum += count;
      pts.push([Math.min(bin, xMax), cum / total]);
    }}
    if (pts[pts.length - 1][1] < 1) pts.push([xMax, 1]);
    cdfs[m] = pts;
    censored[m] = 1 - beforeCap / total;
  }}

  // Faint guides at 25% and 75%; stronger rule at 50%
  ctx.strokeStyle = '#e8e8e8'; ctx.lineWidth = 0.5;
  for (const yv of [0.25, 0.75]) {{
    const py = yPx(yv);
    ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(left + plotW, py); ctx.stroke();
  }}
  ctx.strokeStyle = '#bdbdbd'; ctx.lineWidth = 0.8;
  ctx.beginPath(); ctx.moveTo(left, yPx(0.5)); ctx.lineTo(left + plotW, yPx(0.5)); ctx.stroke();

  // Y-axis labels in percent
  ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'right';
  for (const yv of [0, 0.25, 0.5, 0.75, 1]) {{
    ctx.fillText((yv * 100).toFixed(0) + '%', left - 6, yPx(yv) + 3);
  }}

  // Baseline
  ctx.strokeStyle = '#cfcfcf'; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(left, yPx(0)); ctx.lineTo(left + plotW, yPx(0)); ctx.stroke();

  // X-axis ticks
  ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  for (let x = 0; x <= xMax; x += 50) {{
    ctx.fillText(x, xPx(x), top + plotH + 14);
  }}

  // Vertical dashed rule at the step cap; label sits at the bottom-right of the rule
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(xPx(xMax), top); ctx.lineTo(xPx(xMax), top + plotH); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = MUTED; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  ctx.fillText('step cap', xPx(xMax) - 4, top + plotH - 4);

  // Step curves (staircase — keep it sharp, do not smooth)
  for (const m of ALL_MODELS) {{
    const pts = cdfs[m];
    ctx.strokeStyle = MODEL_COLORS[m];
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    let prevY = yPx(0);
    ctx.moveTo(xPx(0), prevY);
    for (const [x, y] of pts) {{
      const px = xPx(x);
      ctx.lineTo(px, prevY);
      const py = yPx(y);
      ctx.lineTo(px, py);
      prevY = py;
    }}
    ctx.stroke();
  }}

  // Medians via linear interpolation across the cdf
  const medians = {{}};
  for (const m of ALL_MODELS) {{
    const pts = cdfs[m];
    let med = null;
    for (let i = 1; i < pts.length; i++) {{
      if (pts[i][1] >= 0.5) {{
        const [x0, y0] = pts[i - 1];
        const [x1, y1] = pts[i];
        med = y1 === y0 ? x1 : x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0);
        break;
      }}
    }}
    medians[m] = med;
  }}

  // Filled dot on each curve where it crosses y=50%
  for (const m of ALL_MODELS) {{
    if (medians[m] == null) continue;
    const px = xPx(medians[m]);
    const py = yPx(0.5);
    ctx.fillStyle = MODEL_COLORS[m];
    ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fffff8'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.stroke();
  }}

  // Censoring shelf annotation: small label hung below the plateau, near the cap
  ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  for (const m of ALL_MODELS) {{
    const pct = censored[m] * 100;
    if (pct < 1) continue;
    const pts = cdfs[m];
    let yShelf = 1;
    for (let i = pts.length - 1; i >= 0; i--) {{
      if (pts[i][0] < xMax) {{ yShelf = pts[i][1]; break; }}
    }}
    const px = xPx(xMax) - 6;
    const py = yPx(yShelf) + 14;
    ctx.fillStyle = MODEL_COLORS[m];
    ctx.fillText(`~${{pct.toFixed(0)}}% hit cap`, px, py);
  }}

  // Direct labels at right end, sorted by median ascending so top→bottom
  // mirrors the curves left→right at y=50%.
  const labels = ALL_MODELS
    .map(m => ({{ m, med: medians[m] }}))
    .filter(d => d.med != null)
    .sort((a, b) => a.med - b.med);
  ctx.textAlign = 'left';
  const lineH = 16;
  const labelX = left + plotW + 10;
  // Caption above the label stack — explains what the numbers mean once
  ctx.font = 'italic 10px sans-serif'; ctx.fillStyle = MUTED;
  ctx.fillText('half its runs finish in:', labelX, top + 4);
  ctx.font = '11px sans-serif';
  let py0 = top + 20;
  for (let i = 0; i < labels.length; i++) {{
    const item = labels[i];
    const py = py0 + i * lineH;
    ctx.fillStyle = MODEL_COLORS[item.m];
    ctx.fillText(
      `${{MODEL_NAMES[item.m]}} · ${{Math.round(item.med)}} steps`,
      labelX, py
    );
  }}

  // Axis labels
  ctx.fillStyle = MUTED; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('steps per trajectory', left + plotW / 2, h - 6);
  ctx.save();
  ctx.translate(14, top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('% of runs finished in ≤ N steps', 0, 0);
  ctx.restore();

  // Findings-led second subhead, generated from the data
  const desc = document.getElementById('stepDistDesc');
  if (desc && labels.length) {{
    const fastest = labels[0];
    const slowest = labels[labels.length - 1];
    const capHits = ALL_MODELS
      .map(m => ({{ m, pct: censored[m] * 100 }}))
      .filter(d => d.pct >= 1)
      .sort((a, b) => b.pct - a.pct);
    const parts = [
      `${{MODEL_NAMES[fastest.m]}} finishes fastest (median ${{Math.round(fastest.med)}} steps)`,
      `${{MODEL_NAMES[slowest.m]}} runs longest (median ${{Math.round(slowest.med)}})`,
    ];
    if (capHits.length) {{
      parts.push(
        capHits
          .map(d => `~${{d.pct.toFixed(0)}}% of ${{MODEL_NAMES[d.m]}} runs hit the 250-step cap`)
          .join('; ')
      );
    }}
    desc.textContent = parts.join('; ') + '.';
  }}
}})();

// 6. Stacked area charts — 5 grouped bands, inline labels
function drawStackedArea(canvasId, model, annotations, markers) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 40, right = 130, top = 22, bot = 10;
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

  // X-axis ticks and labels ABOVE the chart
  ctx.fillStyle = MUTED; ctx.font = '9px Palatino, Georgia, serif'; ctx.textAlign = 'center';
  ctx.strokeStyle = 'rgba(0,0,0,0.1)'; ctx.lineWidth = 0.5;
  for (let i = 0; i < bins; i++) {{
    if (i % 5 === 0) {{
      const x = xAt(i);
      ctx.fillText(i * 5 + '%', x, top - 8);
      // Small tick down into chart
      ctx.beginPath();
      ctx.moveTo(x, top - 3);
      ctx.lineTo(x, top + 4);
      ctx.stroke();
    }}
  }}

  // 50% vertical reference line (dashed, subtle — keep below the marker line)
  const halfX = xAt(10);
  ctx.strokeStyle = 'rgba(0,0,0,0.07)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(halfX, top);
  ctx.lineTo(halfX, top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Vertical marker line for last code change — medium charcoal, thin, opaque.
  // No axis label; right-margin annotation describes it instead.
  if (markers) {{
    for (const m of markers) {{
      const mx = xAt(m.at / 5);
      ctx.strokeStyle = MUTED;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(mx, top);
      ctx.lineTo(mx, top + plotH);
      ctx.stroke();
    }}

    // Right-margin annotation: two italic muted lines describing the marker.
    const m0 = markers[0];
    if (m0 != null) {{
      const post = 100 - m0.at;
      ctx.fillStyle = MUTED;
      ctx.font = 'italic 11px Palatino, Georgia, serif';
      ctx.textAlign = 'left';
      const ax = left + plotW + 12;
      const ay = top + plotH / 2;
      ctx.fillText(`last edit at ${{m0.at}}%`, ax, ay - 4);
      ctx.fillText(`${{post}}% post-edit`, ax, ay + 12);
    }}
  }}

  // Inline band labels: place name inside band, avoiding edges
  for (let s = 0; s < stacked.length; s++) {{
    const layer = stacked[s];
    // Search for thickest bin within the middle range (bins 2 to bins-3) to avoid edge clipping
    const searchFrom = 2;
    const searchTo = bins - 3;
    let bestBin = Math.floor(bins / 2), bestH = 0;
    for (let i = searchFrom; i <= searchTo; i++) {{
      const h = Math.abs(yAt(layer.bottom[i], i) - yAt(layer.top[i], i));
      if (h > bestH) {{ bestH = h; bestBin = i; }}
    }}
    if (bestH < 16) continue; // too thin to label
    const midY = (yAt(layer.top[bestBin], bestBin) + yAt(layer.bottom[bestBin], bestBin)) / 2;
    ctx.fillStyle = '#fff';
    ctx.globalAlpha = 0.85;
    ctx.font = '10px Palatino, Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText(layer.group.name, xAt(bestBin), midY + 4);
    ctx.globalAlpha = 1;
  }}
}}

(function() {{
  const container = document.getElementById('stackedPanels');
  if (!container) return;
  for (const m of ALL_MODELS) {{
    const wrap = document.createElement('div');
    wrap.className = 'chart-wrapper';
    const cls = tagClass[m] || '';
    const rr = D.resolve_rate[m];
    const sub = rr != null
      ? `<span class="panel-subhead">${{rr.toFixed(1)}}% resolved</span>`
      : '';
    wrap.innerHTML =
      `<div class="stacked-panel-header">` +
        `<span class="model-tag ${{cls}}">${{MODEL_NAMES[m]}}</span>` +
        sub +
      `</div>` +
      `<canvas id="stacked_${{m}}" height="150"></canvas>`;
    container.appendChild(wrap);
    const mle = D.median_last_edit[m];
    const markers = mle != null ? [{{ at: mle }}] : null;
    drawStackedArea(`stacked_${{m}}`, m, null, markers);
  }}
}})();

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
    counts = " ".join(f"{m}={payload['num_trajs'][m]}" for m in payload["models"])
    print(counts)


if __name__ == "__main__":
    main()
