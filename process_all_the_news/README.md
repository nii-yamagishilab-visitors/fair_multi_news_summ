# Media Bias Dataset Pipeline

A data processing pipeline that combines news articles from the All the News dataset with political bias ratings from AllSides to create a bias-tagged, event-grouped corpus.

---

## Data Sources

### Bias Ratings
Bias ratings were manually extracted from [AllSides Media Bias Ratings](https://www.allsides.com/media-bias/ratings) and collected using [`get_allside_rating`](./get_allside_rating) and saved to `get_allside_rating/media_bias_ratings.csv` *(accessed 3 Apr 2025)*.

### News Articles
Download the All the News dataset from the official source *(accessed 15 Apr 2025)*:
- Primary: [components.one/datasets/all-the-news-articles-dataset](https://components.one/datasets/all-the-news-articles-dataset) *(may no longer be active)*
- Alternative: [HuggingFace](https://huggingface.co) or others
- Save as `all-the-news-2-1.csv` in the current directory

---

## Pipeline

Run the following scripts in order, or use `run_pipeline.sh` to execute all steps automatically.

### 1. Clean & Group Articles — `batch_processing_with_checkpoints.py`
Processes the large dataset in batches with checkpoints to avoid memory issues.
- Removes articles with fewer than 20 words
- Groups articles by time proximity and vector similarity
- **Output:** `./multi_news_dataset_output/json/`

### 2. *(Optional)* Check Publisher Name Mismatches — `check_publication_tag.py`
Identifies publisher names in All the News that don't match AllSides entries.
- **Output:** `publications_report.json` (diagnostic only)

### 3. Fix Publisher Labels — `change_all_the_news_label.py`
Renames mismatched publisher names in the All the News dataset to align with AllSides tags.
- **Output:** `./multi_news_dataset_output/json/` (modified in place)

### 4. Add Bias Tags — `add_bias_tag.py`
Joins AllSides bias ratings into each article using the matched publisher tag.
- **Output:** `./multi_news_dataset_output/json/` (modified in place)

### 5. Second Cleaning Pass — `json_cleaner.py`
- Removes articles whose publisher wasn't found in AllSides
- Removes events with fewer than 2 articles
- **Output:** `./multi_news_dataset_output/clean_json/`

### 6. *(Optional)* Filter for Politically Balanced Events — `political_balance_filter.py`
Keeps only events that have coverage from all sides of the political spectrum. Use `--require_center` to enforce center coverage.
- **Output:** `./multi_news_dataset_output/balanced_json/`

### 7. Merge Overlapping Events — `merge_overlap_events.py`
Merges events with overlapping articles into a single record using a lightweight merging algorithm. Pass `--simplify_labels` to collapse `"lean left"` → `"left"` etc., for compatibility with 3-class models (left / center / right).
- **Output:** `./multi_news_dataset_output/merged_events_json/`

### 8. *(Optional)* Filter by Length & Randomise — `filter_and_randomise_with_bias.py`
- Removes events exceeding 5,000 words (for model context window constraints)
- Randomises article order within each event to mitigate position bias
- Optionally pins a specific bias (`left`, `center`, `right`) to the first position
- **Output:** `./filtered_output/random_position_5000{,_left,_center,_right}.json`

### 9. *(Optional)* Filter Out Entertainment Events — `./extra/check_section_tags.py`
- **Output:** `./filtered_output/wo_entertainment/`

### 10. *(Optional)* Filter for Fully Balanced Events — `./extra/filter_balanced_events.py`
Keeps only events with exactly 3 articles — one from each political side (left, center, right).
- **Output:** `./filtered_output/wo_entertainment_balanced/` ← **final output**

---

## Helper Scripts

| Script | Description |
|---|---|
| `helper.py` | Shared utility functions for cleaning and grouping |