#!/bin/sh
# Keep the long reproduction alive across a detached-session failure.
# Recovery is only allowed from an immutable checkpoint boundary.
set -u

session_name="odh_baseline_20260922"
output_dir="artifacts/reproduction/odh_baseline_20260922"
checkpoint="$output_dir/checkpoint_final.pt"
trajectory="$output_dir/trajectory_freeze.json"

while :; do
    if [ -f "$trajectory" ]; then
        exit 0
    fi
    if ! screen -ls 2>/dev/null | rg -q "\.${session_name}[[:space:]]"; then
        if [ ! -f "$checkpoint" ]; then
            printf '%s\n' "Training session disappeared before the first checkpoint; manual review required." >&2
            exit 2
        fi
        screen -dmS "$session_name" sh -c ".venv-reproduction/bin/python scripts/reproduce_odh_baseline.py --mode train --device cpu --threads 4 --resume --output $output_dir"
    fi
    sleep 60
done
