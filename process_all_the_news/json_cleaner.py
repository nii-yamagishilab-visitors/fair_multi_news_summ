"""
Removes articles from unfound publishers (Hyperallergic and TMZ by default)
Removes events that have fewer than 2 articles after filtering
Updates the num_articles count in event metadata
Saves the cleaned files to a new directory
"""

import json
import os
import glob
import shutil
from pathlib import Path

def clean_json_files(input_directory, output_directory, unfound_publishers=None):
    """
    Process JSON files by:
    1. Removing articles from unfound publishers
    2. Removing events with less than 2 articles after filtering
    3. Saving filtered files to a new directory
    
    Args:
        input_directory (str): Path to directory containing JSON files
        output_directory (str): Path to directory where cleaned files will be saved
        unfound_publishers (list): List of publisher names to remove
    
    Returns:
        dict: Statistics about the cleaning process
    """
    if unfound_publishers is None:
        unfound_publishers = ["Hyperallergic", "TMZ"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_directory, "*.json"))
    
    stats = {
        "total_files": len(json_files),
        "processed_files": 0,
        "removed_articles": 0,
        "removed_events": 0,
        "errors": 0
    }
    
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        output_path = os.path.join(output_directory, file_name)
        
        try:
            # Load the JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Make a copy to avoid modification during iteration
            events_to_remove = []
            articles_removed = 0
            
            # Process each event
            for event_id, event_data in data.items():
                if "articles" not in event_data:
                    continue
                
                # Filter out articles from unfound publishers
                original_article_count = len(event_data["articles"])
                event_data["articles"] = [
                    article for article in event_data["articles"]
                    if article.get("publication") not in unfound_publishers
                ]
                
                # Update the number of removed articles
                articles_removed += original_article_count - len(event_data["articles"])
                
                # Mark events for removal if they have less than 2 articles after filtering
                if len(event_data["articles"]) < 2:
                    events_to_remove.append(event_id)
                else:
                    # Update the num_articles field in event_metadata
                    if "event_metadata" in event_data:
                        event_data["event_metadata"]["num_articles"] = len(event_data["articles"])
            
            # Remove marked events
            for event_id in events_to_remove:
                del data[event_id]
            
            # Save the modified data
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Update statistics
            stats["processed_files"] += 1
            stats["removed_articles"] += articles_removed
            stats["removed_events"] += len(events_to_remove)
            
            print(f"Processed {file_name}: Removed {articles_removed} article(s) and {len(events_to_remove)} event(s)")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            stats["errors"] += 1
    
    return stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean JSON files by removing unwanted publishers and events")
    parser.add_argument("input_dir", help="Directory containing the input JSON files")
    parser.add_argument("output_dir", help="Directory where cleaned JSON files will be saved")
    parser.add_argument("--unfound_file", help="Path to JSON file containing unfound publishers list")
    
    args = parser.parse_args()
    
    # Load unfound publishers from file if provided
    unfound_publishers = ["Hyperallergic", "TMZ"]  # Default values
    if args.unfound_file:
        try:
            with open(args.unfound_file, 'r', encoding='utf-8') as f:
                unfound_data = json.load(f)
                if "unfound_publishers" in unfound_data:
                    unfound_publishers = unfound_data["unfound_publishers"]
        except Exception as e:
            print(f"Warning: Could not load unfound publishers from {args.unfound_file}: {e}")
            print("Using default unfound publishers list.")
    
    print(f"Removing articles from the following publishers: {', '.join(unfound_publishers)}")
    print(f"Processing JSON files from {args.input_dir}...")
    
    stats = clean_json_files(args.input_dir, args.output_dir, unfound_publishers)
    
    print("\nSummary:")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files successfully processed: {stats['processed_files']}")
    print(f"Articles removed: {stats['removed_articles']}")
    print(f"Events removed: {stats['removed_events']}")
    print(f"Files with errors: {stats['errors']}")
    print(f"\nCleaned files saved to {args.output_dir}")

if __name__ == "__main__":
    main()