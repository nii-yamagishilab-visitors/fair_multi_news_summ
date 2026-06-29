import spacy
import json
import os
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Set
import pandas as pd
import matplotlib.pyplot as plt

# Define entity types to ignore
IGNORED_ENTITY_TYPES = {
    "DATE",        # Absolute or relative dates or periods
    "TIME",        # Times smaller than a day
    "PERCENT",     # Percentage, including "%"
    "MONEY",       # Monetary values, including unit
    "QUANTITY",    # Measurements, as of weight or distance
    "ORDINAL",     # "first", "second", etc.
    "CARDINAL",    # Numerals that do not fall under another type
}

def load_spacy_model_with_coref(model: str = "en_core_web_sm") -> spacy.language.Language:
    """
    Load spaCy model with coreference resolution capability.
    
    Args:
        model (str): The spaCy model to use
    
    Returns:
        spacy.language.Language: Loaded spaCy model with coref component
    """
    # Load base spaCy model
    nlp = spacy.load(model)
    
    # Add neuralcoref component if it's installed
    try:
        import neuralcoref
        neuralcoref.add_to_pipe(nlp)
        print("Using neuralcoref for coreference resolution")
    except ImportError:
        try:
            # Try to use spaCy's experimental coref component as fallback
            if not nlp.has_pipe("coreferee"):
                try:
                    import coreferee
                    nlp.add_pipe("coreferee")
                    print("Using coreferee for coreference resolution")
                except ImportError:
                    print("WARNING: Neither neuralcoref nor coreferee is installed. Running without coreference resolution.")
        except:
            print("WARNING: Could not add coreference resolution. Running without it.")
    
    return nlp

def resolve_coreferences(doc) -> str:
    """
    Resolve coreferences in a spaCy document.
    
    Args:
        doc: spaCy document with coreference information
    
    Returns:
        str: Text with resolved coreferences
    """
    # Check if we're using neuralcoref
    if hasattr(doc, '_.coref_clusters'):
        # neuralcoref style resolution
        resolved = doc._.coref_resolved
        return resolved
    
    # Check if we're using coreferee
    elif hasattr(doc, '_.coref_chains'):
        # coreferee style resolution
        resolved_text = doc.text
        for chain in doc._.coref_chains:
            # Get the most representative mention (usually the first and most complete one)
            if len(chain) > 0:
                representative_mention = None
                for mention in chain:
                    mention_span = doc[mention.root_index:mention.root_index + 1]
                    
                    # Try to find a named entity mention as the representative
                    is_entity = False
                    for ent in doc.ents:
                        if mention.root_index >= ent.start and mention.root_index < ent.end:
                            representative_mention = doc[ent.start:ent.end].text
                            is_entity = True
                            break
                            
                    if is_entity:
                        break
                    
                    # If no entity found, use the first mention
                    if representative_mention is None:
                        # Get the full mention span
                        span_start = min(token.i for token in mention.span)
                        span_end = max(token.i for token in mention.span) + 1
                        representative_mention = doc[span_start:span_end].text
                
                if representative_mention:
                    # Replace other mentions with the representative mention
                    for mention in chain:
                        if doc[mention.root_index].text.lower() in ['he', 'she', 'it', 'they', 'this', 'that', 'these', 'those', 'him', 'her', 'them']:
                            # Replace pronouns only
                            span_start = min(token.i for token in mention.span)
                            span_end = max(token.i for token in mention.span) + 1
                            mention_text = doc[span_start:span_end].text
                            resolved_text = resolved_text.replace(f" {mention_text} ", f" {representative_mention} ")
        
        return resolved_text
    
    # Fallback: no coreference resolution available
    return doc.text

def extract_entities_with_frequency(text: str, nlp=None, use_coref: bool = True) -> Dict:
    """
    Extract named entities from text and calculate their frequency,
    ignoring specified entity types. Optionally resolve coreferences.
    
    Args:
        text (str): The input text to analyze
        nlp: Pre-loaded spaCy model (if None, will load default)
        use_coref (bool): Whether to use coreference resolution
    
    Returns:
        Dict: Dictionary containing entity frequencies by type and overall
    """
    # Load spaCy model if not provided
    if nlp is None:
        nlp = load_spacy_model_with_coref()
    
    # Process the text
    doc = nlp(text)
    
    # Resolve coreferences if enabled
    if use_coref:
        resolved_text = resolve_coreferences(doc)
        # Re-process with resolved coreferences
        doc = nlp(resolved_text)
    
    # Extract all entities, excluding ignored types
    entities = [(ent.text.strip(), ent.label_) 
                for ent in doc.ents 
                if ent.label_ not in IGNORED_ENTITY_TYPES]
    
    # Count entity occurrences
    entity_counts = Counter([entity[0] for entity in entities])
    
    # Count entity types
    entity_type_counts = Counter([entity[1] for entity in entities])
    
    # Group entities by type
    entities_by_type = {}
    for entity_text, entity_type in entities:
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = Counter()
        entities_by_type[entity_type][entity_text] += 1
    
    # Prepare the results
    results = {
        "total_entities": len(entities),
        "unique_entities": len(entity_counts),
        "entity_counts": dict(entity_counts.most_common()),
        "entity_type_counts": dict(entity_type_counts.most_common()),
        "entities_by_type": {etype: dict(ecounter.most_common()) 
                             for etype, ecounter in entities_by_type.items()}
    }
    
    return results

def calculate_entity_coverage_ratio(source_text: str, summary_text: str, nlp=None, use_coref: bool = True) -> Dict:
    """
    Calculate what percentage of source entities appear in the summary,
    ignoring specified entity types. Optionally use coreference resolution.
    
    Args:
        source_text (str): The original source text
        summary_text (str): The summary text to evaluate
        nlp: Pre-loaded spaCy model (if None, will load default)
        use_coref (bool): Whether to use coreference resolution
        
    Returns:
        Dict: Dictionary with entity coverage metrics
    """
    # Load spaCy model if not provided
    if nlp is None:
        nlp = load_spacy_model_with_coref()
    
    # Extract entities from both texts
    source_entities = extract_entities_with_frequency(source_text, nlp, use_coref)
    summary_entities = extract_entities_with_frequency(summary_text, nlp, use_coref)
    
    # Get sets of unique entities
    source_entity_set = set(source_entities["entity_counts"].keys())
    summary_entity_set = set(summary_entities["entity_counts"].keys())
    
    # Calculate overlap
    common_entities = source_entity_set.intersection(summary_entity_set)
    
    # Calculate entity coverage ratio
    if len(source_entity_set) > 0:
        coverage_ratio = len(common_entities) / len(source_entity_set)
    else:
        coverage_ratio = 0.0
    
    # Calculate entity types in source vs summary
    source_entity_types = set(source_entities["entity_type_counts"].keys())
    summary_entity_types = set(summary_entities["entity_type_counts"].keys())
    common_entity_types = source_entity_types.intersection(summary_entity_types)
    
    if len(source_entity_types) > 0:
        type_coverage_ratio = len(common_entity_types) / len(source_entity_types)
    else:
        type_coverage_ratio = 0.0
    
    # Calculate metrics for each entity type
    entity_type_metrics = {}
    for entity_type in source_entity_types:
        if entity_type in source_entities["entities_by_type"] and entity_type in summary_entities["entities_by_type"]:
            source_type_entities = set(source_entities["entities_by_type"][entity_type].keys())
            summary_type_entities = set(summary_entities["entities_by_type"][entity_type].keys())
            common_type_entities = source_type_entities.intersection(summary_type_entities)
            
            if len(source_type_entities) > 0:
                type_ratio = len(common_type_entities) / len(source_type_entities)
            else:
                type_ratio = 0.0
                
            entity_type_metrics[entity_type] = {
                "source_count": len(source_type_entities),
                "summary_count": len(summary_type_entities),
                "common_count": len(common_type_entities),
                "coverage_ratio": type_ratio
            }
    
    return {
        "coverage_ratio": coverage_ratio,
        "type_coverage_ratio": type_coverage_ratio,
        "source_entity_count": len(source_entity_set),
        "summary_entity_count": len(summary_entity_set),
        "common_entity_count": len(common_entities),
        "source_entity_type_count": len(source_entity_types),
        "summary_entity_type_count": len(summary_entity_types),
        "common_entity_type_count": len(common_entity_types),
        "entity_type_metrics": entity_type_metrics,
        "source_entities": source_entities,
        "summary_entities": summary_entities
    }

def process_file(input_path: str, use_coref: bool = True) -> Dict:
    """
    Process a single JSON file containing summaries
    
    Args:
        input_path (str): Path to the input JSON file
        use_coref (bool): Whether to use coreference resolution
        
    Returns:
        Dict: Dictionary with entity diversity metrics for each event
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    
    # Load the model once to be reused across all events
    nlp = load_spacy_model_with_coref()
    
    # Process each event in the JSON file
    for event_id, event_data in data.items():
        if 'input' in event_data and 'summary' in event_data:
            # Calculate entity diversity
            metrics = calculate_entity_coverage_ratio(
                event_data['input'],
                event_data['summary'],
                nlp,
                use_coref
            )
            
            # Store results
            results[event_id] = {
                "entity_coverage_ratio": metrics["coverage_ratio"],
                "type_coverage_ratio": metrics["type_coverage_ratio"],
                "source_entity_count": metrics["source_entity_count"],
                "summary_entity_count": metrics["summary_entity_count"],
                "common_entity_count": metrics["common_entity_count"],
                "entity_type_metrics": metrics["entity_type_metrics"]
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Calculate entity diversity metrics for summaries')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file with summaries')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file for results')
    parser.add_argument('--no-coref', action='store_true', help='Disable coreference resolution')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization of results')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process the input file
    results = process_file(args.input, not args.no_coref)
    
    # Calculate average metrics across all events
    if results:
        avg_entity_coverage = sum(item["entity_coverage_ratio"] for item in results.values()) / len(results)
        avg_type_coverage = sum(item["type_coverage_ratio"] for item in results.values()) / len(results)
    else:
        avg_entity_coverage = 0.0
        avg_type_coverage = 0.0
    
    # Add summary metrics
    output_data = {
        "events": results,
        "summary": {
            "avg_entity_coverage_ratio": avg_entity_coverage,
            "avg_type_coverage_ratio": avg_type_coverage,
            "total_events_processed": len(results),
            "coreference_resolution_used": not args.no_coref
        }
    }
    
    # Save the results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Processed {len(results)} events")
    print(f"Average entity coverage ratio: {avg_entity_coverage:.4f}")
    print(f"Average entity type coverage ratio: {avg_type_coverage:.4f}")
    print(f"Coreference resolution: {'Disabled' if args.no_coref else 'Enabled'}")
    print(f"Results saved to {args.output}")
    
    # Generate visualizations if requested
    if args.visualize and results:
        output_dir = os.path.dirname(args.output)
        visualization_path = os.path.join(output_dir, "entity_coverage_visualization.png")
        
        # Create DataFrame for plotting
        entity_ratios = [item["entity_coverage_ratio"] for item in results.values()]
        type_ratios = [item["type_coverage_ratio"] for item in results.values()]
        
        df = pd.DataFrame({
            'Event ID': list(results.keys()),
            'Entity Coverage': entity_ratios,
            'Type Coverage': type_ratios
        })
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(entity_ratios, bins=10, alpha=0.7)
        plt.axvline(avg_entity_coverage, color='r', linestyle='dashed', linewidth=1)
        plt.title(f'Entity Coverage Distribution\nAverage: {avg_entity_coverage:.4f}')
        plt.xlabel('Coverage Ratio')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(type_ratios, bins=10, alpha=0.7)
        plt.axvline(avg_type_coverage, color='r', linestyle='dashed', linewidth=1)
        plt.title(f'Entity Type Coverage Distribution\nAverage: {avg_type_coverage:.4f}')
        plt.xlabel('Coverage Ratio')
        
        plt.tight_layout()
        plt.savefig(visualization_path)
        print(f"Visualization saved to {visualization_path}")

if __name__ == "__main__":
    main()