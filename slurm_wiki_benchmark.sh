#!/bin/bash

#SBATCH --job-name=wiki_benchmark
#SBATCH --partition=amperenodes-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/wiki_benchmark_%j.log
#SBATCH --error=logs/wiki_benchmark_%j.err

# ============================================================================
# Wikipedia/Wikidata Benchmark Evaluation
# Methods: graph-llm, onenet-llm, refined, rel, random, spacy, baseline
# ============================================================================

mkdir -p logs

# Define methods to evaluate
# Our LLM methods
LLM_METHODS=("general-graph-llm" "general-onenet-llm")

# All methods combined
ALL_METHODS=("${LLM_METHODS[@]}")

echo ""
echo "Methods to evaluate: ${ALL_METHODS[@]}"
echo ""

# Create result directories
for method in "${ALL_METHODS[@]}"; do
    mkdir -p "evaluation-results/$method"
done

# ============================================================================
# PART 1: Run LLM Methods (Graph-LLM and OneNet-LLM)
# ============================================================================

echo ""
echo "========================================================================"
echo "PART 1: Running LLM Methods (Graph-LLM, OneNet-LLM)"
echo "========================================================================"

for method in "${LLM_METHODS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running $method on benchmarks"
    echo "=========================================="
    
    python link_benchmark.py test -l "$method" -b ALL
    
    echo ""
    echo "Evaluating $method results..."
    for f in evaluation-results/$method/*.linked_articles.jsonl; do
        [[ -e $f ]] || continue
        echo ">>> Evaluating $f"
        python evaluate.py "$f"
    done
done
