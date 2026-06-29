import json
import os
from collections import defaultdict
import shutil
from datetime import datetime

def find_connected_events(input_dir, output_dir, simplify_labels=False):
    """
    Find connected events based on overlapping articles and merge them.
    
    Args:
        input_dir (str): Directory containing the original JSON files
        output_dir (str): Directory to save the new combined JSON files
        simplify_labels (bool, optional): If True, simplify political labels to 3 classes:
                                         "left", "center", and "right". Defaults to False.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dictionary mapping articles to the events they appear in
    article_to_events = defaultdict(set)
    
    # Dictionary to store all events by event_id
    all_events = {}
    
    # Process each JSON file in the input directory
    print("Reading files and identifying overlaps...")
    file_count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Process each event in the JSON file
                    for event_id, event_data in data.items():
                        # Skip if it's not an event object or doesn't have the expected structure
                        if not isinstance(event_data, dict) or 'articles' not in event_data:
                            continue
                        
                        # Store the event
                        all_events[event_id] = event_data
                        
                        # Process each article in the event
                        for article in event_data['articles']:
                            article_id = article.get('article_id')
                            if article_id:
                                article_to_events[article_id].add(event_id)
                file_count += 1
                if file_count % 10 == 0:
                    print(f"Processed {file_count} files...")
            
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    print(f"Completed reading {file_count} files.")
    print(f"Found {len(all_events)} events and {len(article_to_events)} unique articles.")
    
    # Build the event graph: events connected if they share articles
    event_connections = defaultdict(set)
    for article_id, events in article_to_events.items():
        # If an article appears in multiple events, those events are connected
        if len(events) > 1:
            for event_id in events:
                event_connections[event_id].update(events - {event_id})
    
    print(f"Found {sum(1 for events in event_connections.values() if events)} events with connections.")
    
    # # Find connected components (groups of events that should be merged)
    # def find_connected_component(event_id, visited=None):
    #     if visited is None:
    #         visited = set()
        
    #     component = {event_id}
    #     visited.add(event_id)
        
    #     # Explore connections
    #     for connected_event in event_connections.get(event_id, set()):
    #         if connected_event not in visited:
    #             component.update(find_connected_component(connected_event, visited))
        
    #     return component
    
    def find_connected_component(event_id, visited=None):
        if visited is None:
            visited = set()
        
        # Use a stack for depth-first traversal
        stack = [event_id]
        component = set()
        
        while stack:
            current_event = stack.pop()
            
            if current_event not in visited:
                visited.add(current_event)
                component.add(current_event)
                
                # Add connected events to the stack
                for connected_event in event_connections.get(current_event, set()):
                    if connected_event not in visited:
                        stack.append(connected_event)
        
        return component

    # Find all connected components
    all_visited = set()
    connected_components = []
    
    for event_id in all_events.keys():
        if event_id not in all_visited:
            component = find_connected_component(event_id, all_visited)
            if len(component) > 1:  # Only consider components with multiple events
                connected_components.append(component)
    
    print(f"Found {len(connected_components)} groups of connected events to merge.")
    
    # Function to simplify political labels if requested
    def simplify_political_label(label):
        if not simplify_labels:
            return label
            
        if label and isinstance(label, str):
            lower_label = label.lower()
            if 'lean left' in lower_label:
                return 'Left'
            elif 'lean right' in lower_label:
                return 'Right'
            elif 'left' in lower_label:
                return 'Left'
            elif 'right' in lower_label:
                return 'Right'
            elif 'center' in lower_label:
                return 'Center'
        return label
    
    # Merge connected events and write new JSON files
    merged_events = {}
    merged_count = 0
    unchanged_count = 0
    
    # First, handle events that need to be merged
    for i, component in enumerate(connected_components):
        # Create a new merged event ID
        merged_event_id = f"MERGED_EVENT_{i:06d}"
        
        # Collect all articles from component events
        all_articles = []
        article_ids = set()
        earliest_date = None
        
        for event_id in component:
            event_data = all_events[event_id]
            
            # Update earliest date
            event_date = event_data.get('event_metadata', {}).get('event_date')
            if event_date and (not earliest_date or event_date < earliest_date):
                earliest_date = event_date
            
            # Collect articles, avoiding duplicates
            for article in event_data.get('articles', []):
                article_id = article.get('article_id')
                
                # Apply label simplification if requested
                if simplify_labels and 'bias_rating' in article:
                    article['bias_rating'] = simplify_political_label(article['bias_rating'])
                
                if article_id and article_id not in article_ids:
                    all_articles.append(article)
                    article_ids.add(article_id)
        
        # Create merged event
        merged_event = {
            "event_metadata": {
                "event_id": merged_event_id,
                "event_date": earliest_date or datetime.now().strftime("%Y-%m-%d"),
                "num_articles": len(all_articles),
                "article_ids": list(article_ids),
                "merged_from": list(component)
            },
            "articles": all_articles
        }
        
        merged_events[merged_event_id] = merged_event
        merged_count += 1
    
    # Now handle events that don't need to be merged (no overlapping articles)
    for event_id, event_data in all_events.items():
        # Skip if this event was part of a merged component
        if any(event_id in component for component in connected_components):
            continue
        
        # Apply label simplification if requested
        if simplify_labels and 'articles' in event_data:
            for article in event_data['articles']:
                if 'bias_rating' in article:
                    article['bias_rating'] = simplify_political_label(article['bias_rating'])
            
        # Keep this event as is
        merged_events[event_id] = event_data
        unchanged_count += 1
    
    print(f"Created {merged_count} merged events and kept {unchanged_count} unchanged events.")
    if simplify_labels:
        print("Applied label simplification: 'Lean Left' → 'Left', 'Lean Right' → 'Right'")
    
    # Write the merged events to the output directory
    # Split into multiple files to manage file size
    max_events_per_file = 1000
    file_index = 0
    events_in_current_file = 0
    current_file_data = {}
    
    for event_id, event_data in merged_events.items():
        current_file_data[event_id] = event_data
        events_in_current_file += 1
        
        # Write file when it reaches the maximum size
        if events_in_current_file >= max_events_per_file:
            output_file = os.path.join(output_dir, f"merged_events_{file_index:03d}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_file_data, f, indent=2)
            
            print(f"Wrote {events_in_current_file} events to {output_file}")
            file_index += 1
            events_in_current_file = 0
            current_file_data = {}
    
    # Write any remaining events
    if events_in_current_file > 0:
        output_file = os.path.join(output_dir, f"merged_events_{file_index:03d}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_file_data, f, indent=2)
        
        print(f"Wrote {events_in_current_file} events to {output_file}")
    
    # Write a summary file
    summary = {
        "total_original_events": len(all_events),
        "total_merged_events": merged_count,
        "total_unchanged_events": unchanged_count,
        "total_final_events": len(merged_events),
        "merged_groups": len(connected_components),
        "largest_merge_size": max([len(c) for c in connected_components]) if connected_components else 0,
        "labels_simplified": simplify_labels,
        "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("Event merging complete!")
    print(f"Original events: {len(all_events)}")
    print(f"Final events after merging: {len(merged_events)}")
    print(f"Merged {merged_count} events into {len(connected_components)} new events")

if __name__ == "__main__":
    # Paths
    input_directory = "./multi_news_dataset_output/balanced_json"
    output_directory = "./multi_news_dataset_output/merged_events_json"
    
    # Set this to True to simplify political labels to 3 classes (left, center, right)
    simplify_political_labels = True
    
    find_connected_events(input_directory, output_directory, simplify_political_labels)