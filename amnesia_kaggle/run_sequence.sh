#!/bin/bash
# Run a list of models sequentially against local Ollama.
# Usage: run_sequence.sh "model1 model2 model3"
set -u
cd /home/son/GitHub/autoresearch-arena
mkdir -p amnesia_kaggle/logs amnesia_kaggle/data/runs
PY="${PYTHON_BIN:-python3}"
for m in $1; do
    slug=$(echo "$m" | tr '/:' '__')
    log="amnesia_kaggle/logs/${slug}.log"
    out="amnesia_kaggle/data/runs/${slug}.json"
    if [ -f "$out" ]; then
        echo "SKIP $m (already has $out)" >> amnesia_kaggle/logs/sequence.log
        continue
    fi
    echo "=== START $m $(date) ===" >> amnesia_kaggle/logs/sequence.log
    PYTHONUNBUFFERED=1 $PY amnesia_kaggle/run_ollama_bench.py --model "$m" --base-url http://localhost:11434 --ctx-window 8192 > "$log" 2>&1
    echo "=== END $m $(date) rc=$? ===" >> amnesia_kaggle/logs/sequence.log
done
