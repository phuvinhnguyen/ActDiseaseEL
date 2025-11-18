#!/bin/bash

# Script to evaluate general-graph-llm linker on kore50 benchmark
# This uses spaCy for NER and LLM for entity linking

set -e

cd /media/volume/LLMRag2/.local/ActDiseaseEL

# Set environment variables
export PYTHONPATH="src:$PYTHONPATH"
export LLM_QUANT="4bit"

# Configuration
LINKER_NAME="baseline" #"general-graph-llm"
BENCHMARK="kore50"
EXPERIMENT_NAME="spacy-ner-llm-linking"
CONFIG_FILE="configs/baseline.config.json" #"configs/general-graph-llm.config.json"

echo "=========================================="
echo "Evaluating $LINKER_NAME on $BENCHMARK"
echo "=========================================="
echo ""

# Step 1: Link entities
echo "Step 1: Linking entities..."
python link_benchmark.py "$EXPERIMENT_NAME" \
    --linker_name "$LINKER_NAME" \
    --linker_config "$CONFIG_FILE" \
    --benchmark "$BENCHMARK" \
    --evaluation_dir "evaluation-results"

echo ""
echo "Step 1 completed!"
echo ""

# Step 2: Evaluate results
echo "Step 2: Evaluating results..."
LINKED_FILE="evaluation-results/${LINKER_NAME}/${EXPERIMENT_NAME}.${BENCHMARK}.linked_articles.jsonl"

if [ ! -f "$LINKED_FILE" ]; then
    echo "Error: Linked file not found: $LINKED_FILE"
    exit 1
fi

python evaluate.py "$LINKED_FILE" \
    --benchmark "$BENCHMARK" \
    --output_file "evaluation-results/${LINKER_NAME}/${EXPERIMENT_NAME}.${BENCHMARK}.eval_cases.jsonl"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Linked articles: $LINKED_FILE"
echo "  - Evaluation cases: evaluation-results/${LINKER_NAME}/${EXPERIMENT_NAME}.${BENCHMARK}.eval_cases.jsonl"
echo "  - Evaluation results: evaluation-results/${LINKER_NAME}/${EXPERIMENT_NAME}.${BENCHMARK}.eval_results.json"
echo ""

