#!/bin/bash
# Run gemma4:26b and gemma4:31b on Scott's 25 problems, sequentially.
# Waits for any existing gemma4:latest run to finish first.
set -u
cd /home/son/GitHub/autoresearch-arena

SCOTT_PROBLEMS="op_prime_2149519 op_prime_7547 op_labeled_trees op_closed_loops op_prime_sum_negative op_walking_neighbor_100 op_walking_neighbor_10000 op_pressure_vessel op_char_count op_big_multiply op_linear_equation op_cubic_equation op_prime_sequence_300 op_stock_cut_3 op_stock_cut_4 op_stock_cut_5 op_letter_distance_q op_python_script_1 op_python_script_2 op_prisoners op_simulation_5x5x5 op_simulation_10x10x10 op_color_blocks_10 op_color_blocks_12 op_chess_board"

mkdir -p amnesia_kaggle/logs amnesia_kaggle/data/runs

# Wait for any running gemma4:latest bench to finish
echo "$(date) Waiting for gemma4:latest run to finish..." >> amnesia_kaggle/logs/scott25_sequence.log
while pgrep -f "run_ollama_bench.*gemma4:latest" > /dev/null 2>&1; do
    sleep 30
done
echo "$(date) gemma4:latest run done" >> amnesia_kaggle/logs/scott25_sequence.log

for MODEL in "gemma4:26b" "gemma4:31b"; do
    SLUG=$(echo "$MODEL" | tr '/:' '__')
    OUT="amnesia_kaggle/data/runs/${SLUG}_scott25.json"
    LOG="amnesia_kaggle/logs/${SLUG}_scott25.log"

    if [ -f "$OUT" ]; then
        echo "$(date) SKIP $MODEL (already has $OUT)" >> amnesia_kaggle/logs/scott25_sequence.log
        continue
    fi

    echo "$(date) === START $MODEL ===" >> amnesia_kaggle/logs/scott25_sequence.log
    PYTHONUNBUFFERED=1 python3 amnesia_kaggle/run_ollama_bench.py \
        --model "$MODEL" \
        --base-url http://localhost:11434 \
        --ctx-window 8192 \
        --problems $SCOTT_PROBLEMS \
        --out "$OUT" \
        > "$LOG" 2>&1
    RC=$?
    echo "$(date) === END $MODEL rc=$RC ===" >> amnesia_kaggle/logs/scott25_sequence.log
done

echo "$(date) All runs complete" >> amnesia_kaggle/logs/scott25_sequence.log
