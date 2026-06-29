#!/bin/bash
#SBATCH --job-name=eval_fairness_equal
#SBATCH --output=logs/eval_fairness_equal_%j.out
#SBATCH --error=logs/eval_fairness_equal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=qgpu

# =============================================================================
# Sentence-level fairness evaluation across models and data types.
# Runs fairness_sentence_level_equal.py for each (data_type, model) pair.
# Usage: sbatch eval_fairness_equal.sh
# =============================================================================

set -euo pipefail

# path
METHOD="all_input"  # all_input | balanced_input | debias_instruction | debias_persona | structured_prompt | oracle

BASE_SUMMARY_DIR="generated_summary"
BASE_OUTPUT_DIR="evaluation/output"
EVAL_SUBPATH="hard/equal"

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

LOG_FILE="logs/eval_fairness_equal_${SLURM_JOB_ID}.log"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_JOB_NODELIST"
echo "Start:     $(date)"

# Resolve CUDA device
if   [[ -n "${SLURM_JOB_GPUS:-}"  ]]; then export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
elif [[ -n "${SLURM_STEP_GPUS:-}" ]]; then export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
else                                         export CUDA_VISIBLE_DEVICES=0
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi
python -c "import torch; print('PyTorch CUDA —', 'available' if torch.cuda.is_available() else 'NOT available', '| devices:', torch.cuda.device_count())"

# Background GPU monitor (logs every 5 minutes)
GPU_LOG="logs/gpu_monitor_${SLURM_JOB_ID}.log"
( while true; do
    printf '\n[%s] GPU Status:\n' "$(date)" >> "$GPU_LOG"
    nvidia-smi >> "$GPU_LOG"
    sleep 300
  done ) &
GPU_MONITOR_PID=$!

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
    if python fairness_sentence_level_equal.py --input "$SUMMARY_FILE" --output "$OUTPUT_FILE"; then
      duration=$(( $(date +%s) - start ))
      echo "SUCCESS  $TAG (${duration}s)" >> "$LOG_FILE"
      echo "✓ done in ${duration}s"
    else
      duration=$(( $(date +%s) - start ))
      echo "FAILED   $TAG (${duration}s)" >> "$LOG_FILE"
      echo "✗ failed after ${duration}s"
    fi

    python -c "import torch; torch.cuda.empty_cache()"
    echo "----------------------------------------"
  done
done

kill "$GPU_MONITOR_PID" 2>/dev/null && echo "GPU monitor stopped. Log: $GPU_LOG"

SUCCESS=$(grep -c "^SUCCESS" "$LOG_FILE" || true)
FAILED=$( grep -c "^FAILED"  "$LOG_FILE" || true)
SKIPPED=$(grep -c "^SKIPPED" "$LOG_FILE" || true)

printf '\nSUMMARY\n-------\nSuccess: %s\nFailed:  %s\nSkipped: %s\n' \
  "$SUCCESS" "$FAILED" "$SKIPPED" | tee -a "$LOG_FILE"

nvidia-smi
echo "End: $(date)"