"""
Cross check with Allsides label see which publisher does not have a tag
"""

import json
import csv
import re
import os
from glob import glob

def extract_publications_from_json(json_file_path):
    """
    Extract unique publications from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        set: Set of publication names
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Set to store publications
        publications = set()
        
        # Extract publications from each event's articles
        for event_id, event_data in data.items():
            if 'articles' in event_data and isinstance(event_data['articles'], list):
                for article in event_data['articles']:
                    if 'publication' in article and article['publication']:
                        publications.add(article['publication'])
        
        return publications
    
    except json.JSONDecodeError:
        # If JSON parsing fails, try regex approach
        print(f"JSON parsing failed for {json_file_path}. Trying alternative approach...")
        return extract_publications_with_regex(json_file_path)

def extract_publications_with_regex(file_path):
    """
    Extract publications using regex when JSON parsing fails.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        set: Set of publication names
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use regex to find publications
        publications = set()
        publication_pattern = r'"publication"\s*:\s*"([^"]*)"'
        
        for match in re.finditer(publication_pattern, content):
            pub_name = match.group(1)
            if pub_name:
                publications.add(pub_name)
        
        return publications
    
    except Exception as e:
        print(f"Error extracting publications from {file_path}: {e}")
        return set()

def read_media_bias_csv(csv_file_path):
    """
    Read the media bias ratings CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        set: Set of news source names
    """
    media_sources = set()
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'News Source' in row and row['News Source']:
                    media_sources.add(row['News Source'])
        
        return media_sources
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return set()

def extract_unfound_publishers(json_dir_or_pattern, csv_file_path, output_file):
    """
    Process all JSON files and extract publishers not found in the media bias CSV.
    
    Args:
        json_dir_or_pattern (str): Directory containing JSON files or a file pattern
        csv_file_path (str): Path to the media bias CSV file
        output_file (str): Path to the output file
    """
    # Determine JSON files to process
    json_file_paths = []
    
    if os.path.isdir(json_dir_or_pattern):
        # It's a directory, get all JSON files
        for file in os.listdir(json_dir_or_pattern):
            if file.endswith('.json') or file.endswith('.txt'):
                json_file_paths.append(os.path.join(json_dir_or_pattern, file))
    else:
        # Treat as a pattern
        json_file_paths = glob(json_dir_or_pattern)
    
    if not json_file_paths:
        print(f"No JSON files found at {json_dir_or_pattern}")
        return
    
    print(f"Processing {len(json_file_paths)} JSON file(s)")
    
    # Read media sources from CSV
    csv_media_sources = read_media_bias_csv(csv_file_path)
    print(f"Read {len(csv_media_sources)} media sources from CSV")
    
    # Process all JSON files and collect publications
    all_publications = set()
    
    for i, json_path in enumerate(json_file_paths):
        print(f"Processing file {i+1}/{len(json_file_paths)}: {json_path}")
        file_publications = extract_publications_from_json(json_path)
        all_publications.update(file_publications)
    
    # Find unfound publishers (those not in the CSV)
    unfound_publishers = sorted(all_publications - csv_media_sources)
    
    # Save the list of unfound publishers
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write as simple text file, one publisher per line
            f.write('\n'.join(unfound_publishers))
        
        print(f"\nFound {len(unfound_publishers)} publishers not in the CSV")
        print(f"Unfound publishers saved to {output_file}")
        
        # Also save as JSON for easier processing
        json_output = f"{os.path.splitext(output_file)[0]}.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump({
                "total_publishers": len(all_publications),
                "total_csv_sources": len(csv_media_sources),
                "unfound_count": len(unfound_publishers),
                "unfound_publishers": unfound_publishers
            }, f, indent=2)
        
        print(f"Detailed report saved to {json_output}")
        
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Extract publishers not found in media bias CSV.')
    parser.add_argument('json_input', help='Directory containing JSON files or a file pattern')
    parser.add_argument('--csv_file', default='media_bias_ratings.csv', help='Path to media bias ratings CSV file')
    parser.add_argument('--output', default='unfound_publishers.txt', help='Path to output file')
    
    args = parser.parse_args()
    
    # Extract unfound publishers
    extract_unfound_publishers(args.json_input, args.csv_file, args.output)