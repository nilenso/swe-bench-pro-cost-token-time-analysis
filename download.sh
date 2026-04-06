#!/bin/bash
# Resume-safe download script. Run as: bash download.sh
# aws s3 cp skips already-downloaded files automatically.

set -e

echo "Starting GPT-5 download..."
aws s3 cp s3://scaleapi-results/swe-bench-pro/gpt-5-250-turns-10132025/ data/gpt5/ \
    --recursive --no-sign-request --only-show-errors &
PID1=$!

echo "Starting Claude download..."
aws s3 cp s3://scaleapi-results/swe-bench-pro/claude-45sonnet-10132025/ data/claude45/ \
    --recursive --no-sign-request --only-show-errors &
PID2=$!

echo "Both running (PIDs: $PID1, $PID2). Monitoring..."
while kill -0 $PID1 2>/dev/null || kill -0 $PID2 2>/dev/null; do
    G=$(find data/gpt5 -name "*.traj" 2>/dev/null | wc -l | tr -d ' ')
    C=$(find data/claude45 -name "*.traj" 2>/dev/null | wc -l | tr -d ' ')
    GS=$(du -sh data/gpt5 2>/dev/null | cut -f1)
    CS=$(du -sh data/claude45 2>/dev/null | cut -f1)
    echo "$(date +%H:%M:%S)  GPT-5: $G/729 traj ($GS)  Claude: $C/730 traj ($CS)"
    sleep 60
done

echo "Done!"
G=$(find data/gpt5 -name "*.traj" | wc -l | tr -d ' ')
C=$(find data/claude45 -name "*.traj" | wc -l | tr -d ' ')
echo "Final: GPT-5 $G/729, Claude $C/730"
