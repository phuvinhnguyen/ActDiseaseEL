#!/bin/bash
# Optimized DOID Entity Linking Script
# Uses batch processing for 5-10x faster inference

echo "=========================================="
echo "DOID Entity Linking with Optimized LLM"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs
mkdir -p ../doid-results

# Input corpus (modify as needed)
INPUT_CORPUS="english_healthcare_history.jsonl"
if [ ! -f "$INPUT_CORPUS" ]; then
    echo "ERROR: Input corpus not found: $INPUT_CORPUS"
    exit 1
fi

# Count input documents
NUM_DOCS=$(wc -l < "$INPUT_CORPUS")
echo ""
echo "Input corpus: $INPUT_CORPUS ($NUM_DOCS documents)"
echo ""

# Set environment for optimal performance
export LLM_QUANT="4bit"  # Use 4-bit quantization for speed
export PYTHONPATH="src:$PYTHONPATH"

# Run Graph-LLM (optimized with batch processing)
echo "----------------------------------------"
echo "Running Graph-LLM (DOID) - Optimized with Batch Processing"
echo "Expected time: ~5-10s per document"
echo "----------------------------------------"
START_TIME=$(date +%s)

python link_text.py \
    "$INPUT_CORPUS" \
    ../doid-results/graph_llm_doid.jsonl \
    -l graph-llm \
    --article_format \
    --custom_kb

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
AVG_TIME=$(echo "scale=2; $ELAPSED / $NUM_DOCS" | bc -l)

echo ""
echo "✓ Graph-LLM (DOID) complete"
echo "  Output: ../doid-results/graph_llm_doid.jsonl"
echo "  Total time: ${ELAPSED}s"
echo "  Average per document: ${AVG_TIME}s"

# # Run OneNet-LLM (optimized with batch processing)
# echo ""
# echo "----------------------------------------"
# echo "Running OneNet-LLM (DOID) - Optimized with Batch Processing"
# echo "Expected time: ~5-8s per document"
# echo "----------------------------------------"
# START_TIME=$(date +%s)

# python link_text.py \
#     "$INPUT_CORPUS" \
#     ../doid-results/onenet_llm_doid.jsonl \
#     -l onenet-llm \
#     --article_format \
#     --custom_kb

# END_TIME=$(date +%s)
# ELAPSED=$((END_TIME - START_TIME))
# AVG_TIME=$(echo "scale=2; $ELAPSED / $NUM_DOCS" | bc -l)

# echo ""
# echo "✓ OneNet-LLM (DOID) complete"
# echo "  Output: ../doid-results/onenet_llm_doid.jsonl"
# echo "  Total time: ${ELAPSED}s"
# echo "  Average per document: ${AVG_TIME}s"

# echo ""
# echo "=========================================="
# echo "DOID inference complete"
# echo "End time: $(date)"
# echo "=========================================="

# Summary statistics
echo ""
echo "Results Summary:"
echo "----------------"
TOTAL_START=$(date +%s)
for result in ../doid-results/graph_llm_doid.jsonl ../doid-results/onenet_llm_doid.jsonl; do
    [[ -e $result ]] || continue
    method=$(basename "$result" .jsonl)
    num_articles=$(wc -l < "$result" 2>/dev/null || echo "0")
    num_entities=$(python3 -c "
import json
total = 0
try:
    with open('$result') as f:
        for line in f:
            data = json.loads(line)
            total += len(data.get('entity_mentions', []))
except:
    pass
print(total)
" 2>/dev/null || echo "0")
    echo "  ✓ $method:"
    echo "      - $num_articles documents processed"
    echo "      - $num_entities entities linked"
done
