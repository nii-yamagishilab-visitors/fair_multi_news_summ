"""
Filter events with ≤ 5000 words total (across all articles)
Randomise the order of articles within each event, with option to specify first article bias
Output a single JSON file with the qualified events
"""

import json
import os
import glob
import argparse
import random
from collections import defaultdict

def process_json_files(input_directory, output_file, max_word_count=5000, first_article_bias=None):
    """
    Process JSON files to filter events with <= max_word_count words and randomize article order,
    with option to place a specific bias (left, right, center) article first.
    
    Args:
        input_directory (str): Path to directory containing JSON files
        output_file (str): Path to output JSON file
        max_word_count (int): Maximum word count threshold for events to keep
        first_article_bias (str, optional): Political leaning of the first article (left, right, center)
    
    Returns:
        tuple: (num_total_events, num_qualified_events, bias_stats) - statistics about processing
    """
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_directory, "*.json"))
    
    qualified_events = {}
    total_events = 0
    qualified_count = 0
    bias_stats = defaultdict(int)  # To track stats about bias placement
    
    for json_file in json_files:
        file_name = os.path.basename(json_file)
        
        try:
            # Load the JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each event
            for event_id, event_data in data.items():
                total_events += 1
                
                if "articles" in event_data:
                    # Calculate total word count for the event
                    total_word_count = 0
                    for article in event_data["articles"]:
                        # Extract article text
                        article_text = ""
                        if "article" in article:
                            article_text = article["article"]
                        elif "text" in article:  # Fallback in case some files use "text" instead
                            article_text = article["text"]
                        
                        # Count words
                        word_count = len(article_text.split())
                        total_word_count += word_count
                    
                    # Keep only events with <= max_word_count words
                    if total_word_count <= max_word_count:
                        # Create a deep copy of the event data
                        qualified_event = event_data.copy()
                        
                        # Handle article ordering based on bias preference
                        if "articles" in qualified_event and first_article_bias:
                            articles = qualified_event["articles"].copy()
                            
                            # First, shuffle all articles to randomize them
                            random.shuffle(articles)
                            
                            # If bias is specified, try to find an article with that bias to place first
                            if first_article_bias.lower() in ['left', 'right', 'center']:
                                bias_articles = [
                                    article for article in articles 
                                    if article.get('bias_rating', '').lower() == first_article_bias.lower()
                                ]
                                
                                if bias_articles:
                                    # Remove the first found bias article from the list
                                    first_article = bias_articles[0]
                                    articles.remove(first_article)
                                    
                                    # Place it at the front
                                    articles.insert(0, first_article)
                                    bias_stats["success"] += 1
                                else:
                                    # No article with the specified bias was found
                                    bias_stats["not_found"] += 1
                            
                            qualified_event["articles"] = articles
                        else:
                            # Just randomize without any bias preference
                            random.shuffle(qualified_event["articles"])
                        
                        # Add to our qualified events
                        qualified_events[event_id] = qualified_event
                        qualified_count += 1
            
            print(f"Processed {file_name}: {len(data)} events")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Write qualified events to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qualified_events, f, indent=2)
    
    return total_events, qualified_count, bias_stats

def main():
    parser = argparse.ArgumentParser(description="Filter events by word count and randomize article order")
    parser.add_argument("input_dir", help="Directory containing the JSON files to process")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument("--max_words", type=int, default=5000, 
                        help="Maximum word count for qualified events (default: 5000)")
    parser.add_argument("--first_bias", type=str, choices=['left', 'right', 'center'], 
                        help="Specify the political bias of the first article (left, right, center)")
    
    args = parser.parse_args()
    
    # Modify output filename if bias is specified
    if args.first_bias:
        base_name, ext = os.path.splitext(args.output_file)
        output_file = f"{base_name}_{args.first_bias}{ext}"
    else:
        output_file = args.output_file
    
    print(f"Processing JSON files in {args.input_dir}...")
    print(f"First article bias: {args.first_bias if args.first_bias else 'None (fully randomized)'}")
    
    total_events, qualified_events, bias_stats = process_json_files(
        args.input_dir, output_file, args.max_words, args.first_bias
    )
    
    print("\nSummary:")
    print(f"Total events processed: {total_events}")
    print(f"Qualified events (≤ {args.max_words} words): {qualified_events} ({qualified_events/max(1, total_events)*100:.1f}%)")
    
    if args.first_bias:
        print(f"\nBias placement statistics:")
        print(f"  Events with {args.first_bias} bias as first article: {bias_stats['success']}")
        print(f"  Events where {args.first_bias} bias article was not found: {bias_stats['not_found']}")
    
    print(f"\nQualified events saved to: {output_file}")

if __name__ == "__main__":
    # Set random seed for reproducibility if needed
    random.seed(42)
    main()