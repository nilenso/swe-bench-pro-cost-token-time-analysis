"""
Build a self-contained HTML report with paired parity scatter plots.

Pairs instances (both models must have submitted), embeds as JSON,
all filtering/aggregation is client-side JS for instant interactivity.

Usage:
    python scripts/build_report_parity.py stats.json -o report_parity.html
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import orjson


def pair_instances(data):
    """Group by instance_id, keep only pairs where both models submitted."""
    by_inst = defaultdict(dict)
    for s in data:
        name = s.get('config', {}).get('model_name', '')
        if 'gpt' in name.lower():
            model = 'gpt5'
        elif 'claude' in name.lower():
            model = 'claude'
        else:
            continue
        by_inst[s['instance_id']][model] = s

    pairs = []
    skipped_unpaired = 0
    skipped_unsubmitted = 0
    for iid in sorted(by_inst):
        models = by_inst[iid]
        if 'gpt5' not in models or 'claude' not in models:
            skipped_unpaired += 1
            continue
        g, c = models['gpt5'], models['claude']
        if not g.get('submitted') or not c.get('submitted'):
            skipped_unsubmitted += 1
            continue
        pairs.append({
            'instance_id': iid,
            'repo': g['repo'],
            'gpt5': g,
            'claude': c,
        })

    print(
        f"  Paired: {len(pairs)}, skipped unpaired: {skipped_unpaired}, "
        f"skipped unsubmitted: {skipped_unsubmitted}",
        file=sys.stderr,
    )
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file", help="JSON stats file from extract_stats_fast.py")
    parser.add_argument("-o", "--output", default="report_parity.html")
    args = parser.parse_args()

    with open(args.stats_file, 'rb') as f:
        data = orjson.loads(f.read())

    pairs = pair_instances(data)

    # Slim down the data for embedding in HTML
    slim_pairs = []
    keep_keys = [
        'instance_id',
        'repo',
        'resolved',
        'submitted',
        'steps',
        'patch_chars',
        'patch_tokens',
        'model_stats',
        'tokens_per_call',
        'output_tokens',
        'tool_time',
        'actions',
        'content',
    ]
    for p in pairs:
        sp = {'instance_id': p['instance_id'], 'repo': p['repo']}
        for model in ('gpt5', 'claude'):
            sp[model] = {k: p[model][k] for k in keep_keys if k in p[model]}
        slim_pairs.append(sp)

    json_data = orjson.dumps(slim_pairs).decode('utf-8')

    template_path = Path(__file__).parent / 'report_template_parity.html'
    template = template_path.read_text()
    html = template.replace('__DATA_PLACEHOLDER__', json_data)

    with open(args.output, 'w') as f:
        f.write(html)

    print(
        f"  Wrote {args.output} ({len(html) / 1024:.0f} KB, {len(slim_pairs)} paired instances)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
