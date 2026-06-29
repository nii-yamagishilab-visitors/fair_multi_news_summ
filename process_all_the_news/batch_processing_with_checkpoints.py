import os
import pandas as pd
import numpy as np
import json
import gc
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from helper import MultiNewsDatasetCreator, load_and_preprocess_data_with_date_filter

MultiNewsDatasetCreator.load_and_preprocess_data_with_date_filter = load_and_preprocess_data_with_date_filter


def process_full_dataset(input_csv, output_dir, overlap_days=10, time_window_days=3,
                         similarity_threshold=0.6, resume=False):
    """
    Process the entire dataset in batches with overlapping time windows to prevent
    splitting events across chunks. Outputs in JSON format only.

    Parameters:
    -----------
    input_csv : str
        Path to the All the News 2.0 dataset CSV file
    output_dir : str
        Directory to save the output files
    overlap_days : int
        Number of days to overlap between adjacent batches
    time_window_days : int
        Time window for grouping related articles
    similarity_threshold : float
        Threshold for considering articles as related
    resume : bool
        Whether to resume processing from the last checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)

    json_output_dir = os.path.join(output_dir, "json")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(checkpoint_dir, "processing_state.pkl")
    processed_batches_file = os.path.join(checkpoint_dir, "processed_batches.json")

    processed_titles = set()
    all_event_dfs = []
    all_article_dfs = []
    date_ranges = []
    current_batch_index = 0
    processed_batches = {}

    if resume and os.path.exists(checkpoint_file) and os.path.exists(processed_batches_file):
        print("Resuming from checkpoint...")
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                processed_titles = checkpoint_data.get('processed_titles', set())
                all_event_dfs = checkpoint_data.get('all_event_dfs', [])
                all_article_dfs = checkpoint_data.get('all_article_dfs', [])
                date_ranges = checkpoint_data.get('date_ranges', [])
                current_batch_index = checkpoint_data.get('current_batch_index', 0)

            with open(processed_batches_file, 'r') as f:
                processed_batches = json.load(f)

            print(f"Resuming from batch {current_batch_index + 1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            resume = False

    if not resume or not date_ranges:
        print(f"Counting total rows in {input_csv}...")
        total_rows = 0
        for chunk in pd.read_csv(input_csv, chunksize=1000000):
            total_rows += len(chunk)
        print(f"Total articles: {total_rows:,}")

        print("Getting date range information...")
        date_info = []
        for chunk in pd.read_csv(input_csv, usecols=['date'], chunksize=1000000):
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk = chunk.dropna(subset=['date'])
            if len(chunk) > 0:
                date_info.append((chunk['date'].min(), chunk['date'].max()))

        min_dataset_date = min(d[0] for d in date_info)
        max_dataset_date = max(d[1] for d in date_info)
        print(f"Dataset date range: {min_dataset_date.date()} to {max_dataset_date.date()}")

        date_ranges = []
        current_date = min_dataset_date
        batch_size_days = 60

        while current_date <= max_dataset_date:
            end_date = current_date + timedelta(days=batch_size_days)
            extended_end_date = min(end_date + timedelta(days=overlap_days), max_dataset_date)
            date_ranges.append((current_date, extended_end_date))
            current_date = current_date + timedelta(days=batch_size_days)

    for i in range(current_batch_index, len(date_ranges)):
        start_date, end_date = date_ranges[i]
        print(f"\nProcessing batch {i+1}/{len(date_ranges)}: {start_date.date()} to {end_date.date()}")

        batch_key = f"batch_{i+1:03d}"
        if resume and batch_key in processed_batches:
            print(f"Batch {i+1} already processed. Skipping.")
            continue

        creator = MultiNewsDatasetCreator(
            news_dataset_path=input_csv,
            similarity_threshold=similarity_threshold,
            time_window_days=time_window_days
        )

        df = creator.load_and_preprocess_data_with_date_filter(start_date, end_date)

        if len(df) == 0:
            print("No articles found in this date range. Skipping.")
            processed_batches[batch_key] = {
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "status": "empty",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            save_checkpoint(checkpoint_file, processed_batches_file, processed_titles,
                            all_event_dfs, all_article_dfs, date_ranges, i + 1, processed_batches)
            continue

        print(f"Found {len(df):,} articles in this date range")
        print("Available columns:", df.columns.tolist())

        text_column = next((col for col in ['content', 'article', 'text', 'body'] if col in df.columns), None)

        filtered_count = 0
        if text_column is None:
            print("WARNING: Could not find a column containing article text. Skipping word count filter.")
        else:
            print(f"Found text column: {text_column}")
            original_count = len(df)
            df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
            df = df[df['word_count'] >= 20].drop(columns=['word_count'])
            filtered_count = original_count - len(df)
            print(f"Filtered out {filtered_count} articles with fewer than 20 words")
            print(f"Remaining articles: {len(df):,}")

        if len(processed_titles) > 0:
            original_len = len(df)
            df['title_hash'] = df['title'].apply(creator.compute_text_hash)
            df = df[~df['title_hash'].isin(processed_titles)].drop(columns=['title_hash'])
            print(f"Removed {original_len - len(df)} already processed articles from overlap")

        event_clusters = creator.cluster_articles_by_date_and_similarity(df)
        event_df, article_df = creator.create_multi_news_dataset(event_clusters)

        batch_json = create_json_output(event_df, article_df)
        batch_json_path = os.path.join(json_output_dir, f"batch_{i+1:03d}.json")
        with open(batch_json_path, 'w') as f:
            json.dump(batch_json, f)

        all_event_dfs.append(event_df)
        all_article_dfs.append(article_df)

        processed_titles.update(df['title'].apply(creator.compute_text_hash).tolist())

        processed_batches[batch_key] = {
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "status": "completed",
            "events": len(event_df),
            "articles": len(article_df),
            "filtered_articles": filtered_count,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        save_checkpoint(checkpoint_file, processed_batches_file, processed_titles,
                        all_event_dfs, all_article_dfs, date_ranges, i + 1, processed_batches)

        del df, event_clusters
        gc.collect()

        print(f"Completed batch {i+1}/{len(date_ranges)}")

    print("\nMerging all batches into final dataset...")


def save_checkpoint(checkpoint_file, processed_batches_file, processed_titles,
                    all_event_dfs, all_article_dfs, date_ranges, current_batch_index, processed_batches):
    """Save checkpoint information to resume processing later."""
    print("Saving checkpoint...")
    checkpoint_data = {
        'processed_titles': processed_titles,
        'all_event_dfs': all_event_dfs,
        'all_article_dfs': all_article_dfs,
        'date_ranges': date_ranges,
        'current_batch_index': current_batch_index
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    with open(processed_batches_file, 'w') as f:
        json.dump(processed_batches, f, indent=2)
    print(f"Checkpoint saved at batch {current_batch_index}")


def create_json_output(event_df, article_df):
    """
    Create a structured JSON output from the events and articles dataframes.

    Structure:
    {
        "EVENT_000001": {
            "event_metadata": { ... },
            "articles": [ { ... }, ... ]
        },
        ...
    }
    """
    print("Creating structured JSON output...")
    structured_data = {}

    for event_id in event_df['event_id'].unique():
        event_metadata = event_df[event_df['event_id'] == event_id].iloc[0].to_dict()
        if isinstance(event_metadata.get('event_date'), pd.Timestamp):
            event_metadata['event_date'] = event_metadata['event_date'].strftime('%Y-%m-%d')

        structured_data[event_id] = {
            "event_metadata": event_metadata,
            "articles": []
        }

        for _, article in article_df[article_df['event_id'] == event_id].iterrows():
            article_data = article.to_dict()
            if isinstance(article_data.get('date'), pd.Timestamp):
                article_data['date'] = article_data['date'].strftime('%Y-%m-%d')
            structured_data[event_id]["articles"].append(article_data)

    return structured_data


if __name__ == "__main__":
    input_file = "all-the-news-2-1.csv"
    output_directory = "multi_news_dataset_output"

    process_full_dataset(
        input_csv=input_file,
        output_dir=output_directory,
        overlap_days=10,
        time_window_days=3,
        similarity_threshold=0.6,
        resume=True
    )