#!/bin/bash
#SBATCH --job-name=eval_fairness_neutralisation
#SBATCH --output=logs/eval_fairness_neutralisation_%j.out
#SBATCH --error=logs/eval_fairness_neutralisation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=qcpu

# =============================================================================
# Sentence-level neutralisation fairness evaluation across models and data types.
# Runs fairness_sentence_level_neutralisation.py for each (data_type, model) pair.
# Usage: sbatch eval_fairness_neutralisation.sh
# =============================================================================

set -euo pipefail

# path
METHOD="all_input"  # all_input | balanced_input | debias_instruction | debias_persona | structured_prompt | oracle


BASE_SUMMARY_DIR="generated_summary"
BASE_OUTPUT_DIR="evaluation/output"
EVAL_SUBPATH="hard/neutralisation"

SUMMARY_BASE_DIR="${BASE_SUMMARY_DIR}/${METHOD}"
OUTPUT_BASE_DIR="${BASE_OUTPUT_DIR}/${METHOD}/${EVAL_SUBPATH}"

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
  mkdir -p "${OUTPUT_BASE_DIR}/${data_type}"
done

LOG_FILE="logs/eval_fairness_neutralisation_${SLURM_JOB_ID}.log"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Start:  $(date)"
lscpu | grep "Model name"
free -h | grep "Mem"

# Background system monitor (logs every 10 minutes)
SYS_LOG="logs/sys_monitor_${SLURM_JOB_ID}.log"
( while true; do
    printf '\n[%s] System Status:\n' "$(date)" >> "$SYS_LOG"
    top -b -n 1 | head -n 20 >> "$SYS_LOG"
    free -h >> "$SYS_LOG"
    sleep 600
  done ) &
MONITOR_PID=$!

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
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${data_type}/${model_name}"
    OUTPUT_FILE="${OUTPUT_DIR}/results.json"

    if [[ ! -f "$SUMMARY_FILE" ]]; then
      echo "SKIPPED  — file not found: $SUMMARY_FILE"
      echo "SKIPPED  $TAG" >> "$LOG_FILE"
      continue
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "Running  $TAG"

    start=$(date +%s)
    if python fairness_sentence_level_neutralisation.py --input "$SUMMARY_FILE" --output "$OUTPUT_FILE"; then
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

kill "$MONITOR_PID" 2>/dev/null && echo "System monitor stopped. Log: $SYS_LOG"

SUCCESS=$(grep -c "^SUCCESS" "$LOG_FILE" || true)
FAILED=$( grep -c "^FAILED"  "$LOG_FILE" || true)
SKIPPED=$(grep -c "^SKIPPED" "$LOG_FILE" || true)

printf '\nSUMMARY\n-------\nSuccess: %s\nFailed:  %s\nSkipped: %s\n' \
  "$SUCCESS" "$FAILED" "$SKIPPED" | tee -a "$LOG_FILE"

free -h
echo "End: $(date)"