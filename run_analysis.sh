#!/usr/bin/env bash
set -euo pipefail

# SWE-Bench Pro analysis pipeline: traj files → stats JSON → HTML report
#
# Usage:
#   ./run_analysis.sh                     # default: data/{gpt5,claude45}/traj
#   ./run_analysis.sh -w 4                # limit workers
#   ./run_analysis.sh --open              # open report in browser after build

cd "$(dirname "$0")"

WORKERS=""
OPEN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -w) WORKERS="-w $2"; shift 2 ;;
    --open) OPEN=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TRAJ_DIRS="data/gpt5/traj data/claude45/traj"
EVAL_FILES="eval_results_gpt5.json eval_results_claude45.json"
STATS_FILE="stats.json"
REPORT_FILE="report.html"
UNSUB_FILE="unsubmitted.html"
DOCS_DIR="docs"

echo "═══ SWE-Bench Pro Analysis Pipeline ═══"
echo ""

# Step 1: Extract stats
echo "▸ Step 1: Extract trajectory stats"
python3 scripts/extract_stats_fast.py $TRAJ_DIRS \
    --eval-results $EVAL_FILES \
    $WORKERS \
    -o "$STATS_FILE"
echo ""

# Step 2: Build HTML report
echo "▸ Step 2: Build HTML report"
python3 scripts/build_report.py "$STATS_FILE" -o "$REPORT_FILE"
echo ""

# Step 3: Build plain text report (for AI agents)
echo "▸ Step 3: Build plain text report"
python3 scripts/build_text_report.py "$STATS_FILE" -o report.txt
echo ""

# Step 4: Build unsubmitted report
echo "▸ Step 3: Build unsubmitted report"
PAIRED=$(python3 -c "import json; d=json.load(open('$STATS_FILE')); print(len(d))")
python3 scripts/build_unsubmitted_report.py "$STATS_FILE" -o "$UNSUB_FILE" --paired-count "$PAIRED"
echo ""

# Step 5: Copy to docs/ for GitHub Pages
mkdir -p "$DOCS_DIR"
cp "$REPORT_FILE" "$DOCS_DIR/index.html"
cp "$UNSUB_FILE" "$DOCS_DIR/unsubmitted.html"
cp report.txt "$DOCS_DIR/report.txt"
echo ""

echo "═══ Done ═══"
echo "  Stats: $STATS_FILE"
echo "  Report: $REPORT_FILE"
echo "  GH Pages: $DOCS_DIR/index.html"
echo "  Open: file://$(pwd)/$REPORT_FILE"

if $OPEN; then
  open "$REPORT_FILE" 2>/dev/null || xdg-open "$REPORT_FILE" 2>/dev/null || echo "  (open manually)"
fi
