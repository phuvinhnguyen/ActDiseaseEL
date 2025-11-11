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

echo "=========================================="
echo "Wikipedia Benchmark Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Set environment
export PYTHONPATH=src
cd /home/kat/Desktop/UppsalaUniversity/Project/EntityLinking/elevant

# ============================================================================
# Switch to Wikipedia/Wikidata data
# ============================================================================
echo "Switching to Wikipedia/Wikidata data..."
if [ -L elevant_data ]; then
    rm elevant_data
    echo "Removed existing symlink"
fi

if [ -d elevant_data_wiki ]; then
    ln -s elevant_data_wiki elevant_data
    echo "✓ Linked to elevant_data_wiki"
else
    echo "ERROR: elevant_data_wiki folder not found!"
    echo "Please set up data folders according to DATA_SETUP_GUIDE.md"
    exit 1
fi

# Verify the symlink
echo "Current data folder: $(readlink elevant_data)"

# Create logs directory
mkdir -p logs

# Define methods to evaluate
# Our LLM methods
LLM_METHODS=("general-graph-llm" "general-onenet-llm")

# Baseline and comparison methods
BASELINE_METHODS=("refined" "rel" "random" "spacy" "baseline")

# All methods combined
ALL_METHODS=("${LLM_METHODS[@]}" "${BASELINE_METHODS[@]}")

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

# ============================================================================
# PART 2: Run Comparison Methods (Refined, REL, Random, spaCy, Baseline)
# ============================================================================

echo ""
echo "========================================================================"
echo "PART 2: Running Comparison Methods (Refined, REL, Random, spaCy, Baseline)"
echo "========================================================================"

for method in "${BASELINE_METHODS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running $method on benchmarks"
    echo "=========================================="
    
    # Use appropriate prefix for link_benchmark.py
    case "$method" in
        "baseline")
            prefix="baseline"
            ;;
        "spacy")
            prefix="spacy"
            ;;
        "random")
            prefix="random"
            ;;
        "refined")
            prefix="refined"
            ;;
        "rel")
            prefix="rel"
            ;;
        *)
            prefix="test"
            ;;
    esac
    
    python link_benchmark.py "$prefix" -l "$method" -b ALL
    
    echo ""
    echo "Evaluating $method results..."
    for f in evaluation-results/$method/*.linked_articles.jsonl; do
        [[ -e $f ]] || continue
        echo ">>> Evaluating $f"
        python evaluate.py "$f"
    done
done

echo ""
echo "========================================================================"
echo "All Benchmark Evaluations Complete"
echo "End time: $(date)"
echo "========================================================================"

# ============================================================================
# Detailed Results Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "COMPREHENSIVE RESULTS SUMMARY"
echo "========================================================================"

# Get list of all benchmarks evaluated
BENCHMARKS=($(ls evaluation-results/general-graph-llm/*.eval_results.json 2>/dev/null | xargs -n1 basename | sed 's/.*\.\(.*\)\.eval_results\.json/\1/' | sort -u))

echo ""
echo "Benchmarks evaluated: ${BENCHMARKS[@]}"
echo ""

# Print results in a table format for each benchmark
for benchmark in "${BENCHMARKS[@]}"; do
    echo "=========================================="
    echo "Benchmark: $benchmark"
    echo "=========================================="
    printf "%-25s %10s %10s %10s\n" "Method" "Precision" "Recall" "F1"
    printf "%-25s %10s %10s %10s\n" "------" "---------" "------" "--"
    
    # LLM Methods
    echo "--- LLM Methods ---"
    for method in "${LLM_METHODS[@]}"; do
        result_file="evaluation-results/$method/*.$benchmark.eval_results.json"
        result=$(ls $result_file 2>/dev/null | head -1)
        if [[ -e "$result" ]]; then
            precision=$(python3 -c "import json; print(f\"{json.load(open('$result')).get('precision', 0):.4f}\")" 2>/dev/null || echo "N/A")
            recall=$(python3 -c "import json; print(f\"{json.load(open('$result')).get('recall', 0):.4f}\")" 2>/dev/null || echo "N/A")
            f1=$(python3 -c "import json; print(f\"{json.load(open('$result')).get('f1', 0):.4f}\")" 2>/dev/null || echo "N/A")
            printf "%-25s %10s %10s %10s\n" "$method" "$precision" "$recall" "$f1"
        else
            printf "%-25s %10s %10s %10s\n" "$method" "N/A" "N/A" "N/A"
        fi
    done
    
    # Comparison Methods
    echo "--- Comparison Methods ---"
    for method in "${BASELINE_METHODS[@]}"; do
        result_file="evaluation-results/$method/*.$benchmark.eval_results.json"
        result=$(ls $result_file 2>/dev/null | head -1)
        if [[ -e "$result" ]]; then
            precision=$(python3 -c "import json; print(f\"{json.load(open('$result')).get('precision', 0):.4f}\")" 2>/dev/null || echo "N/A")
            recall=$(python3 -c "import json; print(f\"{json.load(open('$result')).get('recall', 0):.4f}\")" 2>/dev/null || echo "N/A")
            f1=$(python3 -c "import json; print(f\"{json.load(open('$result')).get('f1', 0):.4f}\")" 2>/dev/null || echo "N/A")
            printf "%-25s %10s %10s %10s\n" "$method" "$precision" "$recall" "$f1"
        else
            printf "%-25s %10s %10s %10s\n" "$method" "N/A" "N/A" "N/A"
        fi
    done
    echo ""
done

# ============================================================================
# Average Performance Across All Benchmarks
# ============================================================================

echo ""
echo "=========================================="
echo "Average Performance Across All Benchmarks"
echo "=========================================="
printf "%-25s %10s %10s %10s\n" "Method" "Avg P" "Avg R" "Avg F1"
printf "%-25s %10s %10s %10s\n" "------" "------" "------" "------"

for method in "${ALL_METHODS[@]}"; do
    avg_stats=$(python3 << EOF
import json
import glob

results = glob.glob('evaluation-results/$method/*.eval_results.json')
if not results:
    print("N/A N/A N/A")
else:
    precisions, recalls, f1s = [], [], []
    for r in results:
        with open(r) as f:
            data = json.load(f)
            precisions.append(data.get('precision', 0))
            recalls.append(data.get('recall', 0))
            f1s.append(data.get('f1', 0))
    
    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0
    
    print(f"{avg_p:.4f} {avg_r:.4f} {avg_f1:.4f}")
EOF
)
    read avg_p avg_r avg_f1 <<< "$avg_stats"
    printf "%-25s %10s %10s %10s\n" "$method" "$avg_p" "$avg_r" "$avg_f1"
done

echo ""
echo "========================================================================"
echo "Evaluation Complete!"
echo "========================================================================"
echo ""
echo "Result files saved in: evaluation-results/<method>/"
echo "Log file: logs/wiki_benchmark_$SLURM_JOB_ID.log"
echo ""

