import json
import re
import argparse
import os
import glob
from collections import Counter, defaultdict

def analyze_sections_from_text(text_content):
    """
    Analyzes article sections from text content using regex pattern matching
    when JSON parsing might be problematic.
    """
    # Extract all section values using regex
    section_regex = r'"section":\s*(?:"([^"]*)"|([^",\n]*)|\s*(NaN)\s*),?'
    
    sections = []
    for match in re.finditer(section_regex, text_content):
        # Match can be in group 1 (quoted), group 2 (unquoted), or group 3 (NaN)
        section = match.group(1) or match.group(2)
        
        # Skip NaN values and empty strings
        if section and section.lower() != "nan" and section.strip():
            sections.append(section)
    
    # Count the number of articles (by looking for article_id entries)
    article_count = len(re.findall(r'"article_id":', text_content))
    
    # Count sections
    section_counts = Counter(sections)
    
    # Find publications with their sections
    publication_section_regex = r'"publication":\s*"([^"]*)".*?"section":\s*(?:"([^"]*)"|([^",\n]*))'
    publication_sections = []
    
    for match in re.finditer(publication_section_regex, text_content, re.DOTALL):
        publication = match.group(1)
        section = match.group(2) or match.group(3)
        
        # Skip NaN values
        if section and section.lower() != "nan":
            publication_sections.append((publication, section))
    
    # Group by publication
    publication_map = defaultdict(list)
    for publication, section in publication_sections:
        publication_map[publication].append(section)
    
    # Count sections by publication
    publication_section_counts = {}
    for publication, sections in publication_map.items():
        publication_section_counts[publication] = Counter(sections)
    
    # Return all results
    return {
        'total_articles': article_count,
        'articles_with_sections': len(sections),
        'articles_without_sections': article_count - len(sections),
        'unique_sections': len(section_counts),
        'section_counts': dict(section_counts),
        'sections_by_publication': dict(publication_section_counts)
    }

def filter_for_balanced_events(text_content):
    """
    Filters events to keep only those with exactly 3 articles that have a balanced set
    of bias ratings: one Center, one Right, and one Left.
    """
    # Attempt to parse the JSON first
    try:
        data = json.loads(text_content)
        print(f"Successfully parsed JSON with {len(data)} events")
        
        events_to_keep = []
        events_to_filter = []
        
        for event_id, event_data in data.items():
            if not event_id or not isinstance(event_data, dict):
                continue
                
            # Get the articles list
            articles = event_data.get('articles', [])
            
            # Skip if not exactly 3 articles
            if len(articles) != 3:
                events_to_filter.append(event_id)
                continue
            
            # Check if we have one of each bias rating
            bias_ratings = [article.get('bias_rating', '') for article in articles]
            
            # Check if we have exactly one of each required rating
            has_center = bias_ratings.count('Center') == 1
            has_right = bias_ratings.count('Right') == 1
            has_left = bias_ratings.count('Left') == 1
            
            if has_center and has_right and has_left:
                events_to_keep.append(event_id)
            else:
                events_to_filter.append(event_id)
                
        return {
            'total_events': len(data),
            'events_to_keep': len(events_to_keep),
            'events_to_filter': len(events_to_filter),
            'kept_event_ids': events_to_keep,
            'filtered_event_ids': events_to_filter,
            'parsed_data': data  # Return the parsed data for use later
        }
            
    except json.JSONDecodeError as e:
        print(f"Could not parse JSON: {e}")
        print("Falling back to regex pattern matching...")
    
    # If JSON parsing failed, use regex as fallback
    # Find all events in the content
    event_pattern = r'"(MERGED_EVENT_[^"]*|EVENT_[^"]*)":\s*\{[^{]*?"articles":\s*\[(.*?)\]\s*\}'
    events = re.findall(event_pattern, text_content, re.DOTALL | re.IGNORECASE)
    
    events_to_keep = []
    events_to_filter = []
    
    for event_id, articles_section in events:
        # Count the articles in this event
        article_count = len(re.findall(r'"article_id":', articles_section))
        
        # Skip if not exactly 3 articles
        if article_count != 3:
            events_to_filter.append(event_id)
            continue
        
        # Find all bias ratings in this event
        bias_matches = re.findall(r'"bias_rating":\s*"([^"]*)"', articles_section)
        
        # Skip if we don't have exactly 3 bias ratings
        if len(bias_matches) != 3:
            events_to_filter.append(event_id)
            continue
        
        # Check if we have one of each required rating
        has_center = bias_matches.count('Center') == 1
        has_right = bias_matches.count('Right') == 1
        has_left = bias_matches.count('Left') == 1
        
        if has_center and has_right and has_left:
            events_to_keep.append(event_id)
        else:
            events_to_filter.append(event_id)
    
    return {
        'total_events': len(events),
        'events_to_keep': len(events_to_keep),
        'events_to_filter': len(events_to_filter),
        'kept_event_ids': events_to_keep,
        'filtered_event_ids': events_to_filter,
        'parsed_data': None  # No parsed data in regex mode
    }

def save_filtered_events(text_content, event_filter_results, output_file):
    """
    Saves the filtered events to a JSON file.
    
    Args:
        text_content: The original JSON text
        event_filter_results: Results from the filter_balanced_events function
        output_file: Path to save the filtered JSON
    
    Returns:
        bool: True if the save was successful, False otherwise
    """
    if event_filter_results['events_to_keep'] <= 0:
        print("No events to keep after filtering.")
        return False
        
    # If we have parsed data, use that directly
    if event_filter_results.get('parsed_data'):
        kept_events = {}
        for event_id in event_filter_results['kept_event_ids']:
            if event_id in event_filter_results['parsed_data']:
                kept_events[event_id] = event_filter_results['parsed_data'][event_id]
        
        if kept_events:
            with open(output_file, 'w') as f:
                json.dump(kept_events, f, indent=2)
            print(f"\nFiltered events saved to '{output_file}'")
            print(f"The file contains {len(kept_events)} balanced events")
            return True
        else:
            print("\nNo events were kept after filtering, or event IDs didn't match the JSON structure")
            return False
    
    # If we don't have parsed data, use regex to extract the event blocks
    try:
        # More precise event pattern that captures the entire event object
        event_pattern = r'"(MERGED_EVENT_[^"]*|EVENT_[^"]*)":\s*(\{.*?"articles":\s*\[.*?\]\s*\})'
        event_blocks = re.findall(event_pattern, text_content, re.DOTALL)
        
        if event_blocks:
            filtered_json = "{\n"
            kept_count = 0
            
            for event_id, event_block in event_blocks:
                if event_id in event_filter_results['kept_event_ids']:
                    if kept_count > 0:
                        filtered_json += ",\n"
                    filtered_json += f'  "{event_id}": {event_block}'
                    kept_count += 1
            
            filtered_json += "\n}"
            
            if kept_count > 0:
                with open(output_file, 'w') as f:
                    f.write(filtered_json)
                print(f"\nFiltered events saved to '{output_file}' using regex extraction")
                print(f"The file contains {kept_count} balanced events")
                return True
            else:
                print("\nNo events were matched for keeping using the regex approach")
                return False
        else:
            print("\nCould not extract event blocks using regex")
            return False
            
    except Exception as e:
        print(f"\nError saving filtered events: {e}")
        return False

def process_file(input_file, output_file, analysis_only=False):
    """
    Process a single file to filter for balanced events.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output filtered JSON file
        analysis_only: If True, only perform section analysis without filtering
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Read the input file
        with open(input_file, 'r') as file:
            text_content = file.read()
        
        print(f"\nProcessing file: {input_file}")
        
        # Analyze sections
        print("\nANALYSIS OF ALL CONTENT:")
        result = analyze_sections_from_text(text_content)
        
        # Print summary
        print(f"Total articles: {result['total_articles']}")
        print(f"Articles with section information: {result['articles_with_sections']}")
        print(f"Articles without section information: {result['articles_without_sections']}")
        print(f"Number of unique sections: {result['unique_sections']}")
        
        # Print sections by frequency (only top 10 to avoid overwhelming output)
        print("\nTop 10 sections by frequency:")
        sorted_sections = sorted(result['section_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        for section, count in sorted_sections:
            print(f"  {section}: {count}")
        
        # If analysis-only flag is set, exit here
        if analysis_only:
            print("\nAnalysis complete for this file. No filtering performed.")
            return True
        
        # Filter for events with balanced bias ratings
        print("\n\nFILTERING FOR EVENTS WITH BALANCED BIAS RATINGS:")
        event_filter_results = filter_for_balanced_events(text_content)
        print(f"Total events: {event_filter_results['total_events']}")
        print(f"Events without balanced bias ratings (filtered out): {event_filter_results['events_to_filter']}")
        print(f"Events with balanced bias ratings (to keep): {event_filter_results['events_to_keep']}")
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save filtered content
        if event_filter_results['events_to_keep'] > 0:
            print(f"\nSaving filtered content to: {output_file}")
            return save_filtered_events(text_content, event_filter_results, output_file)
        else:
            print("\nNo events to keep after filtering.")
            return False
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def process_directory(input_dir, output_dir, analysis_only=False):
    """
    Process all JSON files in the input directory and save filtered versions to the output directory.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save filtered JSON files
        analysis_only: If True, only perform section analysis without filtering
        
    Returns:
        tuple: (success_count, fail_count)
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0, 0
    
    print(f"Found {len(json_files)} JSON files to process")
    
    success_count = 0
    fail_count = 0
    
    for input_file in json_files:
        # Generate output file path
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_balanced.json")
        
        # Process the file
        if process_file(input_file, output_file, analysis_only):
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count

def main():
    """
    Main function to process command line arguments and filter for balanced events.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Filter for events with balanced bias ratings')
    parser.add_argument('input', help='Path to input JSON file or directory')
    parser.add_argument('--output', '-o', help='Path to output filtered JSON file or directory', default=None)
    parser.add_argument('--analysis-only', '-a', action='store_true', help='Only perform section analysis, no filtering')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all JSON files in the input directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If batch mode is enabled or the input looks like a directory
    if args.batch or os.path.isdir(args.input):
        input_dir = args.input
        
        # Default output directory if not specified
        if args.output is None:
            output_dir = os.path.join(os.path.dirname(input_dir), "balanced")
        else:
            output_dir = args.output
            
        print(f"Batch processing all JSON files from: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        success_count, fail_count = process_directory(input_dir, output_dir, args.analysis_only)
        
        print(f"\nBatch processing complete. Successfully processed {success_count} files, failed {fail_count} files.")
    else:
        # Single file mode
        input_file = args.input
        
        # Default output file if not specified
        if args.output is None:
            base_dir = os.path.dirname(input_file)
            file_name = os.path.basename(input_file)
            output_file = os.path.join(base_dir, "balanced", f"{os.path.splitext(file_name)[0]}_balanced.json")
        else:
            output_file = args.output
            
        process_file(input_file, output_file, args.analysis_only)

if __name__ == "__main__":
    main()