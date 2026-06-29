"""
Several publishers' name mismatch with Allsides, we modify the names in the all the news dataset so that we can match back to the rating using the Allsides tag

The following changes were made:
- The New York Times to New York Times (News)
- Vice News to Vice
- Wired to WIRED
- New Yorker to The New Yorker
- Economist to The Economist
- Buzzfeed News to BuzzFeed News
- Refinery 29 to Refinery29
"""

import json
import os
import glob

def modify_publication_names(json_file_path):
    """
    Modifies publication names in a JSON file according to specified rules.
    
    Args:
        json_file_path (str): Path to the JSON file to be processed
    
    Returns:
        bool: True if file was modified, False otherwise
    """
    # Publication name mapping
    name_mapping = {
        "The New York Times": "New York Times (News)",
        "Vice News": "Vice",
        "Wired": "WIRED",
        "New Yorker": "The New Yorker",
        "Economist": "The Economist",
        "Buzzfeed News": "BuzzFeed News",
        "Refinery 29": "Refinery29"
    }
    
    # Load the JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {json_file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading file {json_file_path}: {e}")
        return False
    
    # Track if any modifications were made
    modified = False
    
    # Process each event in the data
    for event_id, event_data in data.items():
        if "articles" in event_data:
            for article in event_data["articles"]:
                if "publication" in article:
                    current_pub = article["publication"]
                    # Check if publication name needs to be modified
                    if current_pub in name_mapping:
                        article["publication"] = name_mapping[current_pub]
                        modified = True
    
    # Save the modified data if changes were made
    if modified:
        try:
            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            print(f"Modified publication names in {json_file_path}")
        except Exception as e:
            print(f"Error writing to file {json_file_path}: {e}")
            return False
    else:
        print(f"No publication names to modify in {json_file_path}")
    
    return modified

def process_json_files(directory_path, file_pattern="*.json"):
    """
    Processes all JSON files in a directory that match the given pattern.
    
    Args:
        directory_path (str): Path to the directory containing JSON files
        file_pattern (str): Pattern to match JSON files (default: "*.json")
    
    Returns:
        dict: Summary of processing results
    """
    json_files = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not json_files:
        print(f"No JSON files found in {directory_path} matching pattern {file_pattern}")
        return {"total": 0, "modified": 0, "failed": 0}
    
    results = {
        "total": len(json_files),
        "modified": 0,
        "failed": 0
    }
    
    for json_file in json_files:
        try:
            if modify_publication_names(json_file):
                results["modified"] += 1
        except Exception as e:
            print(f"Failed to process {json_file}: {e}")
            results["failed"] += 1
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Modify publication names in JSON files')
    parser.add_argument('directory', help='Directory containing JSON files')
    parser.add_argument('--pattern', default='*.json', help='File pattern to match (default: *.json)')
    
    args = parser.parse_args()
    
    print(f"Processing JSON files in {args.directory}...")
    results = process_json_files(args.directory, args.pattern)
    
    print("\nSummary:")
    print(f"Total files processed: {results['total']}")
    print(f"Files modified: {results['modified']}")
    print(f"Files failed: {results['failed']}")