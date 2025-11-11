#!/bin/bash

#SBATCH --job-name=doid_local
#SBATCH --partition=amperenodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/doid_local_%j.log
#SBATCH --error=logs/doid_local_%j.err

# ============================================================================
# DOID Custom KB Inference with Local Models
# Tests: graph-llm and onenet-llm with local Qwen models
# ============================================================================

echo "=========================================="
echo "DOID Inference with Local Models"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Set environment
export PYTHONPATH=src
cd /home/kat/Desktop/UppsalaUniversity/Project/EntityLinking/elevant

# ============================================================================
# Switch to DOID Custom Knowledge Base data
# ============================================================================
echo "Switching to DOID custom KB data..."
if [ -L elevant_data ]; then
    rm elevant_data
    echo "Removed existing symlink"
fi

if [ -d elevant_data_doid ]; then
    ln -s elevant_data_doid elevant_data
    echo "✓ Linked to elevant_data_doid"
else
    echo "ERROR: elevant_data_doid folder not found!"
    echo "Please set up DOID data according to DATA_SETUP_GUIDE.md"
    echo "Run: python scripts/convert_do_to_elevant.py <doid-file> elevant_data_doid"
    exit 1
fi

# Verify the symlink
echo "Current data folder: $(readlink elevant_data)"

# Create logs directory
mkdir -p logs
mkdir -p ../doid-results

# Input corpus (modify as needed)
INPUT_CORPUS="english_healthcare_history.jsonl"
if [ ! -f "$INPUT_CORPUS" ]; then
    echo "ERROR: Input corpus not found: $INPUT_CORPUS"
    exit 1
fi

# Install flash-attn if using models that require it
# Note: This may take 5-10 minutes
echo "Installing flash-attn (may take several minutes)..."
pip install flash-attn --no-build-isolation 2>&1 | tee logs/flash_attn_install.log
if [ $? -ne 0 ]; then
    echo "WARNING: flash-attn installation failed. Continuing without it..."
    echo "Some models may not work, but Qwen2.5-3B should be fine."
fi

echo ""
echo "=========================================="
echo "Running graph-llm (DOID) with Qwen2.5-3B"
echo "=========================================="

python link_text.py \
    "$INPUT_CORPUS" \
    ../doid-results/graph_llm_doid_qwen3b.jsonl \
    -l graph-llm \
    --linker_config configs/graph-llm-local-qwen3b.config.json \
    --article_format \
    --custom_kb

echo ""
echo "Graph-LLM (DOID, Qwen3B) complete"

echo ""
echo "=========================================="
echo "Running onenet-llm (DOID) with Qwen2.5-3B"
echo "=========================================="

python link_text.py \
    "$INPUT_CORPUS" \
    ../doid-results/onenet_llm_doid_qwen3b.jsonl \
    -l onenet-llm \
    --linker_config configs/onenet-llm-local.config.json \
    --article_format \
    --custom_kb

echo ""
echo "OneNet-LLM (DOID, Qwen3B) complete"

echo ""
echo "=========================================="
echo "DOID inference with local models complete"
echo "End time: $(date)"
echo "=========================================="

# Summary statistics
echo ""
echo "Results Summary:"
echo "----------------"
for result in ../doid-results/*qwen3b.jsonl; do
    [[ -e $result ]] || continue
    method=$(basename "$result" .jsonl)
    num_articles=$(wc -l < "$result")
    num_entities=$(python3 -c "
import json
total = 0
with open('$result') as f:
    for line in f:
        data = json.loads(line)
        total += len(data.get('entity_mentions', []))
print(total)
" 2>/dev/null || echo "N/A")
    echo "  $method: $num_articles articles, $num_entities entities"
done

