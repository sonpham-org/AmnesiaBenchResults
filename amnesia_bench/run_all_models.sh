#!/bin/bash
# Run all models through scott problems on a single Spark.
# Usage:
#   ./amnesia_bench/run_all_models.sh local   # Spark A (localhost)
#   ./amnesia_bench/run_all_models.sh remote  # Spark B (192.168.100.11)

set -e
cd "$(dirname "$0")/.."

SPARK="${1:-local}"
if [ "$SPARK" = "remote" ] || [ "$SPARK" = "b" ]; then
    export OLLAMA_HOST="http://192.168.100.11:11434"
    echo "Using Spark B ($OLLAMA_HOST)"
else
    export OLLAMA_HOST="http://localhost:11434"
    echo "Using Spark A ($OLLAMA_HOST)"
fi

# Models to run (in order) — edit as needed
MODELS=(
    "$2"  # first arg = current model (may be empty)
)

run_model() {
    local model="$1"
    echo ""
    echo "================================================================"
    echo "  Starting: $model"
    echo "  $(date)"
    echo "================================================================"

    python3 amnesia_bench/queue_runner.py \
        --model "$model" \
        --problem-type scott \
        --full --sweep \
        --prompt-variant structured \
        --workers 1 \
        --unbounded-runs 3 \
        --trials 3

    echo ""
    echo "  $model DONE — uploading to Railway..."
    local safe=$(echo "$model" | tr '/:' '__')
    python3 amnesia_bench/upload_results.py --model "$safe" || echo "  Upload failed (non-fatal)"
    echo "  Upload complete."
}

for m in "${MODELS[@]}"; do
    [ -z "$m" ] && continue
    run_model "$m"
done

echo ""
echo "All models done!"
