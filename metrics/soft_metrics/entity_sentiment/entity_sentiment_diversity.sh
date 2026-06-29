#!/bin/bash
#SBATCH --job-name=eval_entity_sentiment
#SBATCH --output=logs/eval_entity_sentiment_%j.out
#SBATCH --error=logs/eval_entity_sentiment_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=qcpu

# =============================================================================
# Entity sentiment diversity evaluation across models and data types.
# Runs entity_sentiment_diversity.py for each (data_type, model) pair.
# Usage: sbatch eval_entity_sentiment.sh
# =============================================================================

set -euo pipefail

# path and param
METHOD="all_input"  # all_input | balanced_input | debias_instruction | debias_persona | structured_prompt | oracle
TOP_K=2          # Number of top entities to analyse per summary

BASE_SUMMARY_DIR="generated_summary"
BASE_OUTPUT_DIR="evaluation/output"
EVAL_SUBPATH="soft/entity_sentiment_diversity"

SUMMARY_BASE_DIR="${BASE_SUMMARY_DIR}/${METHOD}"
OUTPUT_BASE_DIR="${BASE_OUTPUT_DIR}/${METHOD}/${EVAL_SUBPATH}"
OUTPUT_TOP_K_DIR="${OUTPUT_BASE_DIR}/top_${TOP_K}"

# config
DATA_TYPES=("random" "lead_left" "lead_right" "lead_center")

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "google/gemma-3-1b-it"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
)

# setup
export PYTHONUNBUFFERED=1

mkdir -p logs
for data_type in "${DATA_TYPES[@]}"; do
  mkdir -p "${OUTPUT_TOP_K_DIR}/${data_type}"
done

LOG_FILE="logs/eval_entity_sentiment_${SLURM_JOB_ID}.log"

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_JOB_NODELIST"
echo "TOP_K:   $TOP_K"
echo "Output:  $OUTPUT_TOP_K_DIR"
echo "Start:   $(date)"

# Dependencies (no-op if already installed/downloaded)
pip install --quiet NewsSentiment ot spacy nltk tqdm
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
python -c "from NewsSentiment import TargetSentimentClassifier; TargetSentimentClassifier(); print('Sentiment model loaded successfully')"

# main
TOTAL=$(( ${#DATA_TYPES[@]} * ${#MODELS[@]} ))
COUNT=0

echo "Starting evaluation at $(date)" > "$LOG_FILE"

for data_type in "${DATA_TYPES[@]}"; do
  for model_path in "${MODELS[@]}"; do
    model_name="$(basename "$model_path")"
    COUNT=$(( COUNT + 1 ))
    TAG="[$COUNT/$TOTAL] $model_name @ $data_type"

    SUMMARY_FILE="${SUMMARY_BASE_DIR}/${data_type}/${model_name}/summaries.json"
    OUTPUT_DIR="${OUTPUT_TOP_K_DIR}/${data_type}/${model_name}"
    OUTPUT_FILE="${OUTPUT_DIR}/results.json"

    if [[ ! -f "$SUMMARY_FILE" ]]; then
      echo "SKIPPED  — file not found: $SUMMARY_FILE"
      echo "SKIPPED  $TAG" >> "$LOG_FILE"
      continue
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "Running  $TAG"

    start=$(date +%s)
    if python entity_sentiment_diversity.py \
        --input  "$SUMMARY_FILE" \
        --output "$OUTPUT_FILE" \
        --top_k  "$TOP_K"; then
      duration=$(( $(date +%s) - start ))
      echo "SUCCESS  $TAG (${duration}s)" >> "$LOG_FILE"
      echo "✓ done in ${duration}s"
    else
      duration=$(( $(date +%s) - start ))
      echo "FAILED   $TAG (${duration}s)" >> "$LOG_FILE"
      echo "✗ failed after ${duration}s"
    fi

    echo "----------------------------------------"
  done
done

# summ
SUCCESS=$(grep -c "^SUCCESS" "$LOG_FILE" || true)
FAILED=$( grep -c "^FAILED"  "$LOG_FILE" || true)
SKIPPED=$(grep -c "^SKIPPED" "$LOG_FILE" || true)

printf '\nSUMMARY\n-------\nSuccess: %s\nFailed:  %s\nSkipped: %s\n' \
  "$SUCCESS" "$FAILED" "$SKIPPED" | tee -a "$LOG_FILE"

echo "End: $(date)"