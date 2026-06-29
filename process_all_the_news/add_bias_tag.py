"""
Publication Matching:

Handles common variations in publication names (e.g., "CNN" vs "CNN Digital")
Performs partial matching for publications that don't have exact matches
Includes a comprehensive mapping of common name variations


Adds Bias Ratings:

Adds a new bias_rating field to each article in your JSON files
Preserves the original JSON structure
"""
import json
import csv
import os
import glob
import argparse
from pathlib import Path

def load_bias_ratings(csv_file_path):
    """
    Load media bias ratings from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing bias ratings
        
    Returns:
        dict: Dictionary mapping news sources to their bias ratings
    """
    bias_ratings = {}
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                news_source = row.get('News Source')
                bias_rating = row.get('AllSides Bias Rating')
                if news_source and bias_rating:
                    bias_ratings[news_source] = bias_rating
        
        print(f"Loaded {len(bias_ratings)} media bias ratings from {csv_file_path}")
        return bias_ratings
    
    except Exception as e:
        print(f"Error loading bias ratings from {csv_file_path}: {e}")
        return {}

def map_similar_publications(publication_name, bias_ratings):
    """
    Find the closest match for a publication name in the bias ratings dictionary.
    This handles slight variations in publication names.
    
    Args:
        publication_name (str): The publication name from the article
        bias_ratings (dict): Dictionary of media bias ratings
        
    Returns:
        str or None: The bias rating if found, None otherwise
    """
    # Direct match
    if publication_name in bias_ratings:
        return bias_ratings[publication_name]
    
    # Common variations and standardizations
    name_variations = {
        "CNN": "CNN Digital",
        "Washington Post": "The Washington Post",
        "New York Times (News)": "The New York Times",
        "New York Times": "The New York Times",
        "Wall Street Journal": "The Wall Street Journal",
        "Fox News": "Fox News Digital",
        "Guardian": "The Guardian",
        "AP": "Associated Press",
        "NPR": "NPR (Web)",
        "USA Today": "USA TODAY",
        "Politico": "POLITICO",
        "NBC News": "NBC News (Online)",
        "MSNBC": "MSNBC (Web)",
        "ABC News": "ABC News (Online)",
        "CBS News": "CBS News (Online)",
        "BuzzFeed News": "Buzzfeed News",
        "WIRED": "Wired",
        "Bloomberg": "Bloomberg News",
        "Atlantic": "The Atlantic",
        "Daily Beast": "The Daily Beast",
        "Vox": "Vox News",
        "Vice": "VICE",
        "BBC": "BBC News",
        "Reuters": "Reuters News",
        "Economist": "The Economist",
        "CNBC": "CNBC Digital"
    }
    
    # Check if there's a standardized version of the publication name
    if publication_name in name_variations:
        standardized_name = name_variations[publication_name]
        if standardized_name in bias_ratings:
            return bias_ratings[standardized_name]
    
    # Try partial matching for non-exact matches
    # This covers cases like "The New York Times" vs "New York Times"
    for source, rating in bias_ratings.items():
        if publication_name in source or source in publication_name:
            return rating
    
    # No match found
    return None

def add_bias_ratings_to_json(json_file_path, output_file_path, bias_ratings):
    """
    Add bias ratings to publications in a JSON file.
    
    Args:
        json_file_path (str): Path to the input JSON file
        output_file_path (str): Path where the modified JSON will be saved
        bias_ratings (dict): Dictionary mapping news sources to their bias ratings
        
    Returns:
        dict: Statistics about the process
    """
    stats = {
        "total_articles": 0,
        "articles_with_ratings": 0,
        "unmatched_publications": set()
    }
    
    try:
        # Load the JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each event and its articles
        for event_id, event_data in data.items():
            if "articles" not in event_data:
                continue
            
            for article in event_data["articles"]:
                stats["total_articles"] += 1
                
                if "publication" in article:
                    publication = article["publication"]
                    bias_rating = map_similar_publications(publication, bias_ratings)
                    
                    if bias_rating:
                        # Add the bias rating to the article
                        article["bias_rating"] = bias_rating
                        stats["articles_with_ratings"] += 1
                    else:
                        stats["unmatched_publications"].add(publication)
        
        # Save the modified JSON
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return stats
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return stats

def process_directory(input_dir, output_dir, bias_ratings_file):
    """
    Process all JSON files in a directory, adding bias ratings to each.
    
    Args:
        input_dir (str): Directory containing JSON files
        output_dir (str): Directory where modified files will be saved
        bias_ratings_file (str): Path to CSV file with bias ratings
        
    Returns:
        dict: Overall statistics about the process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load bias ratings
    bias_ratings = load_bias_ratings(bias_ratings_file)
    if not bias_ratings:
        print("No bias ratings loaded. Exiting.")
        return
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    # Overall statistics
    overall_stats = {
        "files_processed": 0,
        "total_articles": 0,
        "articles_with_ratings": 0,
        "unmatched_publications": set()
    }
    
    # Process each file
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"Processing {file_name}...")
        
        stats = add_bias_ratings_to_json(json_file, output_path, bias_ratings)
        
        # Update overall statistics
        overall_stats["files_processed"] += 1
        overall_stats["total_articles"] += stats["total_articles"]
        overall_stats["articles_with_ratings"] += stats["articles_with_ratings"]
        overall_stats["unmatched_publications"].update(stats["unmatched_publications"])
    
    return overall_stats

def main():
    parser = argparse.ArgumentParser(description="Add media bias ratings to publications in JSON files")
    parser.add_argument("input_dir", help="Directory containing input JSON files")
    parser.add_argument("output_dir", help="Directory where output JSON files will be saved")
    parser.add_argument("ratings_csv", help="CSV file containing media bias ratings")
    
    args = parser.parse_args()
    
    print(f"Adding media bias ratings to JSON files in {args.input_dir}")
    print(f"Using bias ratings from {args.ratings_csv}")
    print(f"Output will be saved to {args.output_dir}")
    
    overall_stats = process_directory(args.input_dir, args.output_dir, args.ratings_csv)
    
    if overall_stats:
        print("\nProcessing complete!")
        print(f"Files processed: {overall_stats['files_processed']}")
        print(f"Total articles: {overall_stats['total_articles']}")
        print(f"Articles with ratings added: {overall_stats['articles_with_ratings']} " +
              f"({overall_stats['articles_with_ratings'] / overall_stats['total_articles'] * 100:.1f}%)")
        
        if overall_stats["unmatched_publications"]:
            print("\nUnmatched publications:")
            for pub in sorted(overall_stats["unmatched_publications"]):
                print(f"  - {pub}")

if __name__ == "__main__":
    main()