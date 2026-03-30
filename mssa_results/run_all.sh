#!/usr/bin/env bash
# Run all mSSA runs defined in runs.toml sequentially.
# Usage:
#   ./run_all.sh              # with movies
#   ./run_all.sh --no-movies  # skip movie generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS="$@"

mkdir -p "$SCRIPT_DIR/logs"

# Extract run names from runs.toml
RUNS=$(grep '^\[runs\.' "$SCRIPT_DIR/runs.toml" | sed 's/\[runs\.\(.*\)\]/\1/')

echo "Runs to execute: $RUNS"
echo ""

all_ok=true
for run in $RUNS; do
    echo "Starting: $run"
    python "$SCRIPT_DIR/run.py" "$run" $EXTRA_ARGS &> "$SCRIPT_DIR/logs/${run}.log"
    if [ $? -eq 0 ]; then
        echo "Done:   $run"
    else
        echo "FAILED: $run (see logs/${run}.log)"
        all_ok=false
    fi
done

$all_ok && echo "" && echo "All runs completed successfully."
