#!/usr/bin/env bash
# run_pipeline.sh — Full media bias event pipeline
# Usage: ./run_pipeline.sh
#
# NOTE: batch_processing_with_checkpoints.py reads its input path from inside
# the script. Update that path directly before running this pipeline.
#
# Final outputs (all optional steps enabled):
#   ./filtered_output/wo_entertainment_balanced/random_position_5000_{random,left,center,right}_wo_entertainment.json

set -euo pipefail

MAX_WORDS=5000 # Context window filter threshold

# helpers
log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_script() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: Script not found: $1"
    exit 1
  fi
}

log "Checking required scripts..."
REQUIRED_SCRIPTS=(
  "batch_processing_with_checkpoints.py"
  "check_publication_tag.py"
  "change_all_the_news_label.py"
  "add_bias_tag.py"
  "json_cleaner.py"
  "political_balance_filter.py"
  "merge_overlap_events.py"
  "filter_and_randomise_with_bias.py"
  "extra/check_section_tags.py"
  "extra/filter_balanced_events.py"
)
for script in "${REQUIRED_SCRIPTS[@]}"; do
  check_script "$script"
done

if [[ ! -f "get_allside_rating/media_bias_ratings.csv" ]]; then
  echo "ERROR: get_allside_rating/media_bias_ratings.csv not found"
  exit 1
fi

# output dir
mkdir -p ./multi_news_dataset_output/json
mkdir -p ./multi_news_dataset_output/clean_json
mkdir -p ./multi_news_dataset_output/balanced_json
mkdir -p ./multi_news_dataset_output/merged_events_json
mkdir -p ./filtered_output
mkdir -p ./filtered_output/wo_entertainment
mkdir -p ./filtered_output/wo_entertainment_balanced

# Step 1 — Clean & group articles (long-running, run via nohup)
log "Step 1: Cleaning and grouping articles (this may take a while)..."
nohup python batch_processing_with_checkpoints.py > processing_multi_news.log 2>&1 &
BATCH_PID=$!
log "  Running in background (PID $BATCH_PID). Tailing log..."
tail -f processing_multi_news.log &
TAIL_PID=$!
wait $BATCH_PID
BATCH_EXIT=$?
kill $TAIL_PID 2>/dev/null || true
if [[ $BATCH_EXIT -ne 0 ]]; then
  echo "ERROR: batch_processing_with_checkpoints.py failed. Check processing_multi_news.log"
  exit 1
fi
# Output: ./multi_news_dataset_output/json/

# Step 2 (Optional) — Check publisher name mismatches
log "Step 2 (optional): Checking publisher name mismatches..."
python check_publication_tag.py \
  ./multi_news_dataset_output/json \
  --csv_file ./get_allside_rating/media_bias_ratings.csv \
  --output publications_report.json
# Output: publications_report.json (diagnostic only)

# Step 3 — Fix publisher labels to align with AllSides tags
log "Step 3: Fixing publisher labels..."
python change_all_the_news_label.py ./multi_news_dataset_output/json
# Output: ./multi_news_dataset_output/json/ (modified in place)

# Step 4 — Add bias tags from AllSides
log "Step 4: Adding bias tags..."
python add_bias_tag.py \
  ./multi_news_dataset_output/json \
  ./multi_news_dataset_output/json \
  ./get_allside_rating/media_bias_ratings.csv
# Output: ./multi_news_dataset_output/json/ (modified in place)

# Step 5 — Second cleaning pass
log "Step 5: Running second cleaning pass..."
python json_cleaner.py \
  ./multi_news_dataset_output/json \
  ./multi_news_dataset_output/clean_json
# Output: ./multi_news_dataset_output/clean_json/

# Step 6 (Optional) — Filter for politically balanced events
log "Step 6 (optional): Filtering for politically balanced events..."
python political_balance_filter.py \
  ./multi_news_dataset_output/clean_json \
  ./multi_news_dataset_output/balanced_json \
  --require_center
# Output: ./multi_news_dataset_output/balanced_json/

# Step 7 — Merge overlapping events
log "Step 7: Merging overlapping events..."
python merge_overlap_events.py \
  ./multi_news_dataset_output/balanced_json \
  ./multi_news_dataset_output/merged_events_json \
  --simplify_labels
# Output: ./multi_news_dataset_output/merged_events_json/

# Step 8 (Optional) — Filter by length & randomise article order
# Runs 4 variants: random, left-first, center-first, right-first
log "Step 8 (optional): Filtering by length and randomising article order..."
for BIAS in random left center right; do
  if [[ "$BIAS" == "random" ]]; then
    python filter_and_randomise_with_bias.py \
      ./multi_news_dataset_output/merged_events_json \
      ./filtered_output/random_position_${MAX_WORDS}.json \
      --max_words "$MAX_WORDS"
  else
    python filter_and_randomise_with_bias.py \
      ./multi_news_dataset_output/merged_events_json \
      ./filtered_output/random_position_${MAX_WORDS}_${BIAS}.json \
      --max_words "$MAX_WORDS" \
      --first_bias "$BIAS"
  fi
done
# Output: ./filtered_output/random_position_5000{,_left,_center,_right}.json

# Step 9 (Optional) — Filter out entertainment events (one per bias variant)
log "Step 9 (optional): Filtering out entertainment events..."
for BIAS in random left center right; do
  if [[ "$BIAS" == "random" ]]; then
    IN="./filtered_output/random_position_${MAX_WORDS}.json"
    OUT="./filtered_output/wo_entertainment/random_position_${MAX_WORDS}_wo_entertainment.json"
  else
    IN="./filtered_output/random_position_${MAX_WORDS}_${BIAS}.json"
    OUT="./filtered_output/wo_entertainment/random_position_${MAX_WORDS}_${BIAS}_wo_entertainment.json"
  fi
  python extra/check_section_tags.py "$IN" --output "$OUT"
done
# Output: ./filtered_output/wo_entertainment/

# Step 10 (Optional) — Keep only fully balanced events (one per bias variant)
log "Step 10 (optional): Filtering for fully balanced events..."
for BIAS in random left center right; do
  if [[ "$BIAS" == "random" ]]; then
    IN="./filtered_output/wo_entertainment/random_position_${MAX_WORDS}_wo_entertainment.json"
    OUT="./filtered_output/wo_entertainment_balanced/random_position_${MAX_WORDS}_wo_entertainment.json"
  else
    IN="./filtered_output/wo_entertainment/random_position_${MAX_WORDS}_${BIAS}_wo_entertainment.json"
    OUT="./filtered_output/wo_entertainment_balanced/random_position_${MAX_WORDS}_${BIAS}_wo_entertainment.json"
  fi
  python extra/filter_balanced_events.py "$IN" --output "$OUT"
done
# Output: ./filtered_output/wo_entertainment_balanced/

log "Pipeline complete."
log "Final outputs in: ./filtered_output/wo_entertainment_balanced/"
ls ./filtered_output/wo_entertainment_balanced/