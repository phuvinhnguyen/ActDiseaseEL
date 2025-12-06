#!/bin/bash
mkdir -p logs

LLM_METHODS=("general-graph-llm")
BENCHMARKS=("aida-conll-test")
# "aida-conll-test" "kore50" "msnbc""msnbc-updated" "derczynski" "news-fair-no-coref" "news-fair-v2-no-coref"

for m in "${LLM_METHODS[@]}"; do mkdir -p "evaluation-results/$m"; done

for method in "${LLM_METHODS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        (( $(jobs -r | wc -l) >= 7 )) && wait -n
        python link_benchmark.py test -l "$method" -b "$benchmark" > "logs/link_${method}_${benchmark}.log" 2>&1 &
    done
done; wait

# Evaluation phase (4 parallel)
for f in evaluation-results/*/*.linked_articles.jsonl; do
    [[ -e $f ]] || continue
    (( $(jobs -r | wc -l) >= 14 )) && wait -n
    python evaluate.py "$f" > "logs/eval_${f##*/}.log" 2>&1 &
done; wait