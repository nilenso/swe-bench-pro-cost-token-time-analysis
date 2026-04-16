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

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--gpt)"></div>
      <span>GPT-5</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--claude)"></div>
      <span>Sonnet 4.5</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--gemini)"></div>
      <span>Gemini 2.5 Pro</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--glm)"></div>
      <span>GLM-4.5</span>
    </div>
  </div>

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
    <div class="side-label"><span class="model-tag claude">Sonnet 4.5</span></div>
    <canvas id="stackedClaude" height="200"></canvas>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag gemini">Gemini 2.5 Pro</span></div>
    <canvas id="stackedGemini" height="200"></canvas>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag glm">GLM-4.5</span></div>
    <canvas id="stackedGlm" height="200"></canvas>
  </div>

  <h2>4b. Phase Profile — When Does Each Action Happen?</h2>
  <p class="chart-desc">Each trajectory is divided into 20 equal time-slices (0%–100%). The heatmap shows what proportion of steps in each slice belong to each category. Read left-to-right as "beginning → end of trajectory".</p>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag gpt">GPT-5</span></div>
    <div id="heatmapGpt"></div>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag claude">Sonnet 4.5</span></div>
    <div id="heatmapClaude"></div>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag gemini">Gemini 2.5 Pro</span></div>
    <div id="heatmapGemini"></div>
  </div>
  <div class="chart-wrapper">
    <div class="side-label"><span class="model-tag glm">GLM-4.5</span></div>
    <div id="heatmapGlm"></div>
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

const MODEL_COLORS = D.model_colors;
const MODEL_NAMES = D.model_display_names;
const ALL_MODELS = D.models;
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
  const left = 54, right = 150, top = 20, bot = 46;
  const xMax = 250;
  const plotW = w - left - right;
  const plotH = h - top - bot;
  const xPx = x => left + (x / xMax) * plotW;
  const yPx = y => top + (1 - y) * plotH;

  // Build ECDF points per model from binned step counts
  const cdfs = {{}};
  for (const m of ALL_MODELS) {{
    const entries = Object.entries(D.step_dist[m])
      .map(([k, v]) => [Number(k), v])
      .sort((a, b) => a[0] - b[0]);
    const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
    let cum = 0;
    const pts = [[0, 0]];
    for (const [bin, count] of entries) {{
      cum += count;
      pts.push([Math.min(bin, xMax), cum / total]);
    }}
    if (pts[pts.length - 1][1] < 1) pts.push([xMax, 1]);
    cdfs[m] = pts;
  }}

  // Horizontal guides at 0.25, 0.5, 0.75
  ctx.strokeStyle = '#e0e0e0'; ctx.lineWidth = 0.5;
  ctx.font = '10px monospace'; ctx.textAlign = 'right';
  for (const yv of [0, 0.25, 0.5, 0.75, 1]) {{
    const py = yPx(yv);
    if (yv === 0.25 || yv === 0.5 || yv === 0.75) {{
      ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(left + plotW, py); ctx.stroke();
    }}
    ctx.fillStyle = MUTED;
    ctx.fillText(yv.toFixed(2), left - 6, py + 3);
  }}

  // Baseline + cap rule
  ctx.strokeStyle = '#cfcfcf'; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(left, yPx(0)); ctx.lineTo(left + plotW, yPx(0)); ctx.stroke();

  // X-axis ticks
  ctx.fillStyle = MUTED; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  for (let x = 0; x <= xMax; x += 50) {{
    ctx.fillText(x, xPx(x), top + plotH + 14);
  }}

  // Vertical dashed rule at the step cap
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(xPx(xMax), top); ctx.lineTo(xPx(xMax), top + plotH); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = MUTED; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  ctx.fillText('step cap', xPx(xMax) - 4, top + 10);

  // Step curves
  for (const m of ALL_MODELS) {{
    const pts = cdfs[m];
    ctx.strokeStyle = MODEL_COLORS[m];
    ctx.lineWidth = 1.6;
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

  // Median ticks on each curve at y=0.5
  for (const m of ALL_MODELS) {{
    if (medians[m] == null) continue;
    const px = xPx(medians[m]);
    const py = yPx(0.5);
    ctx.strokeStyle = MODEL_COLORS[m];
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(px, py - 5); ctx.lineTo(px, py + 5); ctx.stroke();
  }}

  // Direct labels at right end with median annotation; offset to avoid overlap
  const labels = ALL_MODELS.map(m => {{
    const pts = cdfs[m];
    let yAtEdge = 1;
    for (let i = pts.length - 1; i >= 0; i--) {{
      if (pts[i][0] < xMax) {{ yAtEdge = pts[i][1]; break; }}
    }}
    return {{ m, y: yAtEdge }};
  }});
  labels.sort((a, b) => b.y - a.y);
  let lastPy = -Infinity;
  ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  for (const item of labels) {{
    let py = yPx(item.y);
    if (py - lastPy < 14) py = lastPy + 14;
    lastPy = py;
    ctx.fillStyle = MODEL_COLORS[item.m];
    const med = medians[item.m];
    const text = med != null
      ? `${{MODEL_NAMES[item.m]}} — median ${{Math.round(med)}}`
      : MODEL_NAMES[item.m];
    ctx.fillText(text, left + plotW + 8, py + 3);
  }}

  // Axis labels
  ctx.fillStyle = MUTED; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('steps per trajectory', left + plotW / 2, h - 6);
  ctx.save();
  ctx.translate(14, top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('cumulative share of trajectories', 0, 0);
  ctx.restore();
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
drawHeatmap('heatmapGlm', 'glm45', 'row');
drawHeatmap('heatmapGemini', 'gemini25pro', 'row');

// 5b. Column-normalised

// 6. Stacked area charts — 5 grouped bands, inline labels
function drawStackedArea(canvasId, model, annotations, markers) {{
  const {{ ctx, w, h }} = getCtx(canvasId);
  const left = 40, right = 20, top = 30, bot = 10;
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

  // 50% vertical reference line (dashed, subtle)
  const halfX = xAt(10);
  ctx.strokeStyle = 'rgba(0,0,0,0.12)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(halfX, top);
  ctx.lineTo(halfX, top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Vertical markers above chart with ticks down
  if (markers) {{
    for (const m of markers) {{
      const mx = xAt(m.at / 5);
      // Solid line through chart
      ctx.strokeStyle = 'rgba(0,0,0,0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(mx, top);
      ctx.lineTo(mx, top + plotH);
      ctx.stroke();
      // Label above
      ctx.fillStyle = 'rgba(0,0,0,0.45)';
      ctx.font = '9px Palatino, Georgia, serif';
      ctx.textAlign = 'center';
      ctx.fillText(m.label, mx, top - 8);
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

const canvasMap = {{
  'gpt5': 'stackedGpt',
  'claude45': 'stackedClaude',
  'glm45': 'stackedGlm',
  'gemini25pro': 'stackedGemini',
}};
for (const m of ALL_MODELS) {{
  const id = canvasMap[m];
  if (!id) continue;
  const mle = D.median_last_edit[m];
  const markers = mle != null ? [{{ at: mle, label: `last code change (${{mle}}%)` }}] : null;
  drawStackedArea(id, m, null, markers);
}}

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
