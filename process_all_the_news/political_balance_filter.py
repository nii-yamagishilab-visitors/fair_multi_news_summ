import json
import os
import glob
import shutil
from pathlib import Path
from collections import Counter

def categorize_political_leaning(article):
    """
    Extract the political leaning from an article's bias_rating field.
    If bias_rating is not available, tries to infer from publication name.
    
    Args:
        article (dict): The article data dictionary
        
    Returns:
        str: The political leaning category 
             ('left', 'lean_left', 'center', 'lean_right', 'right', 'unknown')
    """
    # First check if the article has a bias_rating field
    if "bias_rating" in article and article["bias_rating"]:
        bias = article["bias_rating"]
        
        # Map the bias_rating to our categories
        if bias == "Left":
            return "left"
        elif bias == "Lean Left":
            return "lean_left"
        elif bias == "Center":
            return "center"
        elif bias == "Lean Right":
            return "lean_right"
        elif bias == "Right":
            return "right"
    
    # If no bias_rating or it's not one of our categories, try to infer from publication
    publication = article.get("publication", "")
    
    # Fallback publication mapping for articles without bias_rating
    leaning_map = {
        # Left
        "The Guardian": "left",
        "Mother Jones": "left",
        "Huffington Post": "left",
        "Vox": "left",
        "Slate": "left",
        
        # Lean Left
        "The New York Times": "lean_left",
        "Washington Post": "lean_left",
        "CNN": "lean_left",
        "NBC News": "lean_left",
        "MSNBC": "lean_left",
        "The Atlantic": "lean_left",
        
        # Center
        "Reuters": "center",
        "Associated Press": "center",
        "BBC": "center",
        "Bloomberg": "center",
        "USA Today": "center",
        
        # Lean Right
        "The Wall Street Journal": "lean_right",
        "The Economist": "lean_right",
        "The Hill": "lean_right",
        "New York Post": "lean_right",
        
        # Right
        "Fox News": "right",
        "National Review": "right",
        "Breitbart": "right",
        "The Daily Caller": "right",
        "Washington Examiner": "right"
    }
    
    return leaning_map.get(publication, "unknown")

def simplify_leaning(leaning):
    """
    Group political leanings into broader categories for checking diversity.
    
    Args:
        leaning (str): Detailed political leaning
        
    Returns:
        str: Simplified category ('left_side', 'center', 'right_side')
    """
    if leaning in ["left", "lean_left"]:
        return "left_side"
    elif leaning == "center":
        return "center"
    elif leaning in ["right", "lean_right"]:
        return "right_side"
    else:
        return "unknown"

def has_political_diversity(articles, require_center=False):
    """
    Check if the articles represent a diverse political spectrum.
    
    Args:
        articles (list): List of article dictionaries
        require_center (bool): If True, also require center-leaning articles
        
    Returns:
        bool: True if there is political diversity, False otherwise
    """
    # Count the number of articles from each political leaning
    leanings = [simplify_leaning(categorize_political_leaning(article)) 
                for article in articles]
    
    # Filter out unknown leanings
    known_leanings = [l for l in leanings if l != "unknown"]
    
    # Check if we have articles from each side
    has_left = "left_side" in known_leanings
    has_center = "center" in known_leanings
    has_right = "right_side" in known_leanings
    
    # Determine if the collection meets our diversity requirements
    if require_center:
        # Require all three perspectives (left, center, and right)
        return has_left and has_center and has_right
    else:
        # Only require left and right perspectives
        return has_left and has_right

def filter_events_by_political_balance(input_directory, output_directory, min_articles=3, require_center=False):
    """
    Filter JSON files to keep only events that:
    1. Have more than min_articles articles
    2. Include articles from multiple political leanings
    
    Args:
        input_directory (str): Path to directory containing JSON files
        output_directory (str): Path to directory where filtered files will be saved
        min_articles (int): Minimum number of articles required (default: 3)
        require_center (bool): If True, requires center-leaning articles in addition to left and right
        
    Returns:
        dict: Statistics about the filtering process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_directory, "*.json"))
    
    stats = {
        "total_files": len(json_files),
        "processed_files": 0,
        "kept_events": 0,
        "removed_events": 0,
        "files_with_kept_events": 0,
        "files_without_kept_events": 0,
        "errors": 0,
        "political_distribution": Counter()
    }
    
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        output_path = os.path.join(output_directory, file_name)
        
        try:
            # Load the JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Make a copy for filtered events
            filtered_data = {}
            events_removed = 0
            events_kept = 0
            
            # Process each event
            for event_id, event_data in data.items():
                if "articles" not in event_data:
                    events_removed += 1
                    continue
                
                articles = event_data["articles"]
                
                # Check if event meets our criteria
                if (len(articles) >= min_articles) and has_political_diversity(articles, require_center):
                    filtered_data[event_id] = event_data
                    events_kept += 1
                    
                    # Track political distribution for statistics
                    for article in articles:
                        leaning = categorize_political_leaning(article)
                        stats["political_distribution"][leaning] += 1
                else:
                    events_removed += 1
            
            # Only save the file if there are any events left
            if filtered_data:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                stats["files_with_kept_events"] += 1
            else:
                stats["files_without_kept_events"] += 1
            
            # Update statistics
            stats["processed_files"] += 1
            stats["kept_events"] += events_kept
            stats["removed_events"] += events_removed
            
            print(f"Processed {file_name}: Kept {events_kept} event(s), removed {events_removed} event(s)")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            stats["errors"] += 1
    
    return stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter events by political balance and article count")
    parser.add_argument("input_dir", help="Directory containing the input JSON files")
    parser.add_argument("output_dir", help="Directory where filtered JSON files will be saved")
    parser.add_argument("--min_articles", type=int, default=3, 
                        help="Minimum number of articles per event (default: 3)")
    parser.add_argument("--require_center", action="store_true",
                        help="If set, requires center-leaning articles in addition to left and right")
    
    args = parser.parse_args()
    
    if args.require_center:
        print(f"Filtering events with >= {args.min_articles} articles and ALL political perspectives (left, center, right)")
    else:
        print(f"Filtering events with >= {args.min_articles} articles and diverse political representation (left and right)")
    
    print(f"Processing JSON files from {args.input_dir}...")
    
    stats = filter_events_by_political_balance(
        args.input_dir, 
        args.output_dir, 
        args.min_articles, 
        args.require_center
    )
    
    print("\nSummary:")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files successfully processed: {stats['processed_files']}")
    print(f"Events kept: {stats['kept_events']}")
    print(f"Events removed: {stats['removed_events']}")
    print(f"Files with kept events: {stats['files_with_kept_events']}")
    print(f"Files without kept events: {stats['files_without_kept_events']}")
    print(f"Files with errors: {stats['errors']}")
    
    print("\nPolitical distribution in kept articles:")
    for leaning, count in stats["political_distribution"].most_common():
        print(f"  {leaning}: {count} articles")
    
    print(f"\nFiltered files saved to {args.output_dir}")

if __name__ == "__main__":
    main()