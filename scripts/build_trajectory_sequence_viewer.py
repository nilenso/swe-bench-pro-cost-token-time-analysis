#!/usr/bin/env python3
"""
Build an interactive HTML viewer for trajectory intent sequences.

Features:
- Filter by model (gpt5 / claude45 / both)
- Filter by repo
- Filter by task (instance id)
- Layer view: high-level letters, low-level letters, or both
- Expand each row for additional details

Usage:
  python scripts/build_trajectory_sequence_viewer.py \
      --data-root data \
      --output docs/trajectory-sequences.html
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import classify_intent as ci


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

# 46+ low-level intents need a stable single-char alphabet
LOW_LEVEL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def parse_repo(instance_id: str) -> str:
    core = instance_id
    if core.startswith("instance_"):
        core = core[len("instance_") :]

    if "__" not in core:
        return "unknown/unknown"

    org, rest = core.split("__", 1)

    # Strip commit-ish suffix while allowing hyphens in org/repo names.
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


def build_payload(data_root: Path) -> dict:
    rows: list[dict] = []
    all_base_intents: set[str] = set()

    files = collect_files(data_root)

    for model, path in files:
        data = ci._load_json(str(path))
        trajectory = data.get("trajectory", [])
        base_intents = ci.classify_trajectory(trajectory)
        hierarchical = ci.classify_hierarchical_layer(base_intents)

        highs = [h.split(".", 1)[0] for h in hierarchical]
        high_seq = "".join(HIGH_LEVEL_LETTER.get(h, "?") for h in highs)

        instance_id = path.stem
        repo = parse_repo(instance_id)
        info = data.get("info", {})

        all_base_intents.update(base_intents)

        rows.append(
            {
                "model": model,
                "repo": repo,
                "task": instance_id,
                "steps": len(base_intents),
                "exit_status": info.get("exit_status", ""),
                "submitted": bool(info.get("submission")),
                "high_seq": high_seq,
                "_base_intents": base_intents,
                "path": str(path),
            }
        )

    sorted_intents = sorted(all_base_intents)
    if len(sorted_intents) > len(LOW_LEVEL_ALPHABET):
        raise RuntimeError(
            f"Need larger low-level alphabet: intents={len(sorted_intents)} alphabet={len(LOW_LEVEL_ALPHABET)}"
        )

    intent_to_char = {intent: LOW_LEVEL_ALPHABET[i] for i, intent in enumerate(sorted_intents)}
    char_to_intent = {v: k for k, v in intent_to_char.items()}

    for row in rows:
        base_intents = row.pop("_base_intents")
        row["low_seq"] = "".join(intent_to_char[i] for i in base_intents)

    rows.sort(key=lambda r: (r["repo"], r["task"], r["model"]))

    return {
        "rows": rows,
        "high_level_letter": HIGH_LEVEL_LETTER,
        "low_level_intent_char": intent_to_char,
        "char_to_low_level_intent": char_to_intent,
        "meta": {
            "num_rows": len(rows),
            "num_tasks": len({r['task'] for r in rows}),
            "num_repos": len({r['repo'] for r in rows}),
        },
    }


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))

    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Trajectory Sequence Viewer</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #121922;
      --muted: #9fb0c0;
      --text: #e8eef5;
      --accent: #6cb6ff;
      --border: #2a3645;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .controls {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 10px 14px;
      display: grid;
      grid-template-columns: repeat(7, minmax(120px, 1fr));
      gap: 8px;
      align-items: end;
    }}
    .controls label {{ font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }}
    .controls select, .controls input {{
      width: 100%;
      background: #0f141c;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 13px;
    }}
    .controls .check {{ display:flex; gap:8px; align-items:center; font-size:12px; color: var(--muted); }}
    .summary {{ padding: 10px 14px; color: var(--muted); font-size: 13px; }}
    .legend {{ padding: 0 14px 10px 14px; font-size: 12px; color: var(--muted); }}
    .legend code {{ color: var(--text); }}
    .rows {{ padding: 0 10px 30px 10px; }}
    .row {{
      background: #0f151d;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin: 8px 4px;
      padding: 8px 10px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .badge {{
      border: 1px solid var(--border);
      background: #111b28;
      color: var(--accent);
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 11px;
    }}
    .seq {{
      width: 100%;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: clamp(8px, 1vw, 12px);
      line-height: 1.25;
      letter-spacing: 0.2px;
      white-space: pre-wrap;
      word-break: break-all;
      color: #d4e2ef;
      background: #0b1118;
      border: 1px solid #1f2a37;
      border-radius: 6px;
      padding: 6px;
    }}
    details {{ margin-top: 6px; }}
    details summary {{ cursor: pointer; color: var(--accent); font-size: 12px; }}
    .small {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <div class=\"controls\">
    <div>
      <label>Model</label>
      <select id=\"model\"></select>
    </div>
    <div>
      <label>Repo</label>
      <select id=\"repo\"></select>
    </div>
    <div>
      <label>Task</label>
      <select id=\"task\"></select>
    </div>
    <div>
      <label>Layer</label>
      <select id=\"layer\">
        <option value=\"high\">High-level letters</option>
        <option value=\"low\">Low-level letters</option>
        <option value=\"both\">Both (expanded)</option>
      </select>
    </div>
    <div>
      <label>Sort</label>
      <select id=\"sort\">
        <option value=\"repo_task\">Repo → Task → Model</option>
        <option value=\"steps_desc\">Steps (desc)</option>
        <option value=\"steps_asc\">Steps (asc)</option>
      </select>
    </div>
    <div>
      <label>Contains letters</label>
      <input id=\"contains\" placeholder=\"e.g. IVV or SX\" />
    </div>
    <div class=\"check\">
      <input id=\"submittedOnly\" type=\"checkbox\" />
      <span>Submitted only</span>
    </div>
  </div>

  <div class=\"summary\" id=\"summary\"></div>
  <div class=\"legend\" id=\"legend\"></div>
  <div class=\"rows\" id=\"rows\"></div>

  <script>
    const DATA = {payload_json};

    const el = (id) => document.getElementById(id);
    const rowsEl = el('rows');

    const modelEl = el('model');
    const repoEl = el('repo');
    const taskEl = el('task');
    const layerEl = el('layer');
    const sortEl = el('sort');
    const containsEl = el('contains');
    const submittedOnlyEl = el('submittedOnly');

    const summaryEl = el('summary');
    const legendEl = el('legend');

    const rows = DATA.rows;
    const highMap = DATA.high_level_letter;
    const lowMap = DATA.low_level_intent_char;
    const charToLow = DATA.char_to_low_level_intent;

    function unique(arr) {{ return [...new Set(arr)].sort(); }}

    function fillSelect(select, options, current='all') {{
      const prev = current || select.value || 'all';
      select.innerHTML = '';
      for (const o of options) {{
        const op = document.createElement('option');
        op.value = o.value;
        op.textContent = o.label;
        select.appendChild(op);
      }}
      const hasPrev = options.some(o => o.value === prev);
      select.value = hasPrev ? prev : options[0].value;
    }}

    function buildFilterOptions() {{
      fillSelect(modelEl, [
        {{value:'both', label:'both'}},
        {{value:'gpt5', label:'gpt5'}},
        {{value:'claude45', label:'claude45'}},
      ], modelEl.value || 'both');

      const model = modelEl.value;
      const base = rows.filter(r => model === 'both' ? true : r.model === model);

      const repos = unique(base.map(r => r.repo));
      fillSelect(repoEl, [{{value:'all',label:'all'}}].concat(repos.map(r => ({{value:r,label:r}}))), repoEl.value || 'all');

      const repo = repoEl.value;
      const base2 = base.filter(r => repo === 'all' ? true : r.repo === repo);
      const tasks = unique(base2.map(r => r.task));
      fillSelect(taskEl, [{{value:'all',label:'all'}}].concat(tasks.map(t => ({{value:t,label:t}}))), taskEl.value || 'all');
    }}

    function getFilteredRows() {{
      const model = modelEl.value;
      const repo = repoEl.value;
      const task = taskEl.value;
      const contains = (containsEl.value || '').trim();
      const submittedOnly = submittedOnlyEl.checked;
      const layer = layerEl.value;

      let out = rows.filter(r =>
        (model === 'both' || r.model === model) &&
        (repo === 'all' || r.repo === repo) &&
        (task === 'all' || r.task === task) &&
        (!submittedOnly || r.submitted)
      );

      if (contains) {{
        const upper = contains.toUpperCase();
        out = out.filter(r => {{
          const seq = layer === 'low' ? r.low_seq : r.high_seq;
          return seq.toUpperCase().includes(upper);
        }});
      }}

      const sortMode = sortEl.value;
      if (sortMode === 'steps_desc') out.sort((a,b) => b.steps - a.steps);
      else if (sortMode === 'steps_asc') out.sort((a,b) => a.steps - b.steps);
      else out.sort((a,b) => (a.repo + '|' + a.task + '|' + a.model).localeCompare(b.repo + '|' + b.task + '|' + b.model));

      return out;
    }}

    function runLength(seq) {{
      if (!seq) return '';
      let out = [];
      let cur = seq[0], n = 1;
      for (let i = 1; i < seq.length; i++) {{
        if (seq[i] === cur) n++;
        else {{ out.push(cur + (n>1 ? n : '')); cur = seq[i]; n = 1; }}
      }}
      out.push(cur + (n>1 ? n : ''));
      return out.join(' ');
    }}

    function renderLegend() {{
      const high = Object.entries(highMap)
        .map(([k,v]) => `<code>${{v}}</code>=${{k}}`)
        .join(' · ');

      const lowPairs = Object.entries(lowMap)
        .sort((a,b) => a[1].localeCompare(b[1]))
        .map(([intent,ch]) => `<code>${{ch}}</code>=${{intent}}`)
        .join(' · ');

      legendEl.innerHTML = `
        <div><strong>High-level letters:</strong> ${{high}}</div>
        <details style=\"margin-top:6px\"><summary>Low-level letter map</summary><div class=\"small mono\">${{lowPairs}}</div></details>
      `;
    }}

    function render() {{
      const filtered = getFilteredRows();
      const layer = layerEl.value;

      summaryEl.textContent = `showing ${{filtered.length}} trajectories (of ${{rows.length}})`;
      rowsEl.innerHTML = '';

      for (const r of filtered) {{
        const wrap = document.createElement('div');
        wrap.className = 'row';

        const seqMain = (layer === 'high') ? r.high_seq : (layer === 'low' ? r.low_seq : r.high_seq);
        const seqAlt = (layer === 'both') ? r.low_seq : '';

        wrap.innerHTML = `
          <div class=\"meta\">
            <span class=\"badge\">${{r.model}}</span>
            <span>${{r.repo}}</span>
            <span>${{r.task}}</span>
            <span>steps=${{r.steps}}</span>
            <span>submitted=${{r.submitted ? 'yes' : 'no'}}</span>
            <span>${{r.exit_status || ''}}</span>
          </div>
          <div class=\"seq\">${{seqMain}}</div>
          ${{layer === 'both' ? `<div class=\"small\" style=\"margin-top:4px\">low-level:</div><div class=\"seq\">${{seqAlt}}</div>` : ''}}
          <details>
            <summary>expand</summary>
            <div class=\"small mono\">high RLE: ${{runLength(r.high_seq)}}</div>
            <div class=\"small mono\">low  RLE: ${{runLength(r.low_seq)}}</div>
            <div class=\"small mono\">path: ${{r.path}}</div>
          </details>
        `;

        rowsEl.appendChild(wrap);
      }}
    }}

    function refreshOptionsAndRender() {{
      buildFilterOptions();
      render();
    }}

    modelEl.addEventListener('change', refreshOptionsAndRender);
    repoEl.addEventListener('change', refreshOptionsAndRender);
    taskEl.addEventListener('change', render);
    layerEl.addEventListener('change', render);
    sortEl.addEventListener('change', render);
    containsEl.addEventListener('input', render);
    submittedOnlyEl.addEventListener('change', render);

    buildFilterOptions();
    renderLegend();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive trajectory sequence viewer HTML")
    parser.add_argument("--data-root", default="data", help="Path to data root (default: data)")
    parser.add_argument("--output", "-o", default="docs/trajectory-sequences.html", help="Output HTML path")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    payload = build_payload(data_root)
    html = render_html(payload)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote {out_path}")
    print(
        f"rows={payload['meta']['num_rows']} tasks={payload['meta']['num_tasks']} repos={payload['meta']['num_repos']}"
    )


if __name__ == "__main__":
    main()
