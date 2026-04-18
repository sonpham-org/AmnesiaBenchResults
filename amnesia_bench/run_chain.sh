#!/bin/bash
# Chain runner: runs models sequentially on a Spark.
# Usage: OLLAMA_HOST=http://192.168.100.11:11434 ./run_chain.sh model1 model2 ...

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKERS=${WORKERS:-3}

for model in "$@"; do
    echo ""
    echo "=============================="
    echo "  Starting: $model ($(date '+%H:%M:%S'))"
    echo "  Workers: $WORKERS"
    echo "=============================="
    python3 "$SCRIPT_DIR/queue_runner.py" \
        --model "$model" \
        --problem-type aimo3 \
        --workers "$WORKERS" \
        --full
    echo "  Finished: $model ($(date '+%H:%M:%S'))"
done
echo ""
echo "All models done! ($(date '+%H:%M:%S'))"
