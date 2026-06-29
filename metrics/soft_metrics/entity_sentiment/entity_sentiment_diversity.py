"""
top-k
- Work out common entities
- Select top appearing in source documents
"""
from NewsSentiment import TargetSentimentClassifier
import json
import os
import argparse
import time
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import ot
import scipy.stats

def load_json_from_file(file_path):
    """Load JSON data from a file with all events"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_entities(text, nlp):
    """
    Extract named entities from text
    
    Args:
        text (str): Text to extract entities from
        nlp: spaCy model
        
    Returns:
        dict: Entity counts and mapping to sentences
    """
    # Define entity types to ignore
    ignore_types = [
        "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
    ]
    
    # Process the text
    doc = nlp(text)
    
    # Extract entities, excluding ignored types
    entities = []
    for ent in doc.ents:
        if ent.label_ not in ignore_types:
            entities.append((ent.text.strip(), ent.label_))
    
    # Count entity occurrences
    entity_counts = Counter([entity[0] for entity in entities])
    
    # Get sentences containing entities
    sentences = sent_tokenize(text)
    entity_to_sentences = defaultdict(list)
    
    # Map entities to sentences they appear in
    for i, sentence in enumerate(sentences):
        doc_sent = nlp(sentence)
        for ent in doc_sent.ents:
            entity_text = ent.text.strip()
            if entity_text and ent.label_ not in ignore_types:
                entity_to_sentences[entity_text].append((i, sentence))
    
    return {
        "entity_counts": entity_counts,
        "entity_to_sentences": entity_to_sentences,
        "sentences": sentences
    }

def get_top_k_entities(entity_data, k=None):
    """
    Get top-k entities based on frequency
    
    Args:
        entity_data (dict): Entity data from extract_entities
        k (int, optional): Number of top entities to return
        
    Returns:
        list: Top-k entities
    """
    # Get all entities sorted by frequency
    sorted_entities = entity_data["entity_counts"].most_common()
    
    # Return all if k is None or greater than available entities
    if k is None or k >= len(sorted_entities):
        return [entity for entity, _ in sorted_entities]
    
    # Return top-k entities
    return [entity for entity, _ in sorted_entities[:k]]

def calculate_entity_sentiment(entity, sentences, tsc):
    """
    Calculate sentiment distribution for a single entity
    
    Args:
        entity (str): Entity name
        sentences (list): List of (idx, sentence) pairs containing the entity
        tsc: Target Sentiment Classifier instance
        
    Returns:
        dict: Sentiment distribution and counts for the entity
    """
    entity_sentiments = []
    
    print(f"Analyzing sentiment for entity '{entity}' in {len(sentences)} sentences")
    
    for idx, sentence in sentences:
        try:
            # Make sure the sentence isn't too long
            if len(sentence) > 1000:
                sentence = sentence[:1000]
            
            # Properly format input for TargetSentimentClassifier
            entity_lower = entity.lower()
            sentence_lower = sentence.lower()
            
            if entity_lower in sentence_lower:
                # Find the position of the entity in the sentence (case insensitive)
                start_pos = sentence_lower.find(entity_lower)
                end_pos = start_pos + len(entity_lower)
                
                # Extract prefix and suffix
                prefix = sentence[:start_pos]
                suffix = sentence[end_pos:]
                
                # Prepare data for sentiment analysis with correct prefix/suffix
                data = [(prefix, entity, suffix)]
            else:
                # Fallback if entity not found exactly as is in the sentence
                data = [("", sentence, "")]
                
            # Get sentiment for the entity in this sentence
            sentiment_result = tsc.infer(targets=data)
            
            # Extract sentiment class
            try:
                sentiment_class = sentiment_result[0][0]['class_label']
                confidence = sentiment_result[0][0].get('confidence', 0.0)
                
                entity_sentiments.append({
                    "sentence_idx": idx,
                    "sentence": sentence,
                    "sentiment": sentiment_class,
                    "confidence": confidence
                })
            except (KeyError, IndexError, TypeError) as e:
                print(f"Error extracting sentiment data: {str(e)}")
                continue
            
        except Exception as e:
            print(f"Error processing entity '{entity}' in sentence: {str(e)}")
            continue
    
    # Calculate sentiment distribution
    sentiment_counts = Counter([s["sentiment"] for s in entity_sentiments])
    total = sum(sentiment_counts.values()) if sentiment_counts else 0
    
    # If no successful sentiment analysis, assign default neutral sentiment
    if total == 0:
        sentiment_counts['neutral'] = 1
        total = 1
        
        # Create a default sentiment entry with the first sentence if available
        if sentences:
            idx, sentence = sentences[0]
            entity_sentiments.append({
                "sentence_idx": idx,
                "sentence": sentence,
                "sentiment": "neutral",
                "confidence": 1.0,
                "is_default": True
            })
    
    # Ensure all sentiment types are present
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in sentiment_counts:
            sentiment_counts[sentiment] = 0
    
    # Normalize to proportions
    sentiment_distribution = {
        "positive": sentiment_counts.get("positive", 0) / total if total > 0 else 0,
        "negative": sentiment_counts.get("negative", 0) / total if total > 0 else 0,
        "neutral": sentiment_counts.get("neutral", 0) / total if total > 0 else 0
    }
    
    # Create formatted list of entity-sentiment pairs for display/reporting
    sentiment_list = []
    for sentiment_type in ['positive', 'negative', 'neutral']:
        count = sentiment_counts.get(sentiment_type, 0)
        for _ in range(count):
            sentiment_list.append(f"{entity}, {sentiment_type}")
    
    return {
        "sentiment_distribution": sentiment_distribution,
        "sentiment_counts": dict(sentiment_counts),
        "total_mentions": total,
        "detailed_sentiments": entity_sentiments,
        "sentiment_list": sentiment_list,
        "had_successful_analysis": len(sentiment_counts) > 0
    }

def custom_wasserstein_distance(p, q):
    """Calculate Wasserstein distance between two distributions using uniform costs
    
    This treats categories as unordered with uniform cost of 1 to move between different bins
    and 0 cost to stay in the same bin.
    
    Args:
        p (array-like): First distribution
        q (array-like): Second distribution
        
    Returns:
        float: Wasserstein distance between the distributions
    """
    p = np.asarray(p)
    q = np.asarray(q)
    
    # Use optimal transport with uniform costs between bins
    n = len(p)
    cost_matrix = np.ones((n, n)) - np.eye(n)  # Cost 1 to move between different bins, 0 for same bin
    return ot.emd2(p, q, cost_matrix)

def kl_divergence(p, q, epsilon=1e-10):
    """
    Calculate the Kullback-Leibler divergence between two distributions
    Adding epsilon to avoid division by zero or log of zero
    
    Args:
        p (list): First distribution
        q (list): Second distribution
        epsilon (float): Small value to avoid numerical issues
        
    Returns:
        float: KL divergence value
    """
    # Add small epsilon to avoid log(0) or division by zero
    p = np.asarray(p) + epsilon
    q = np.asarray(q) + epsilon
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate KL divergence
    return float(np.sum(p * np.log(p / q)))

def determine_sentiment_shift(source_dist, summary_dist, threshold=0.05):
    """
    Determine the sentiment shift from source to summary
    
    Args:
        source_dist (list): Source sentiment distribution [pos, neg, neutral]
        summary_dist (list): Summary sentiment distribution [pos, neg, neutral]
        threshold (float): Threshold for considering a shift significant
        
    Returns:
        str: Sentiment shift category
    """
    # Calculate differences
    pos_diff = summary_dist[0] - source_dist[0]  # positive diff
    neg_diff = summary_dist[1] - source_dist[1]  # negative diff
    neu_diff = summary_dist[2] - source_dist[2]  # neutral diff
    
    # Calculate absolute differences
    abs_pos_diff = abs(pos_diff)
    abs_neg_diff = abs(neg_diff)
    abs_neu_diff = abs(neu_diff)
    
    # Check if any difference exceeds the threshold
    if max(abs_pos_diff, abs_neg_diff, abs_neu_diff) <= threshold:
        return "balanced"
    
    # Determine which sentiment shows the most significant increase
    if pos_diff > threshold and pos_diff > neg_diff and pos_diff > neu_diff:
        return "more_positive"
    elif neg_diff > threshold and neg_diff > pos_diff and neg_diff > neu_diff:
        return "more_negative"
    elif neu_diff > threshold and neu_diff > pos_diff and neu_diff > neg_diff:
        return "more_neutral"
    
    # Default fallback
    return "balanced"

def process_file(input_path, top_k=5, tsc=None, nlp=None):
    """
    Process sentiment for top-k common entities in each event in a file
    
    Args:
        input_path (str): Path to input JSON
        top_k (int): Number of top common entities to analyze (by source frequency)
        tsc: Target Sentiment Classifier instance
        nlp: spaCy model
        
    Returns:
        dict: Dictionary with sentiment diversity results
    """
    # Load the input JSON
    events_data = load_json_from_file(input_path)
    results = {}
    
    # Process each event
    total_events = len(events_data)
    print(f"Processing {total_events} events with top-{top_k} common entities...")
    
    # Use tqdm for a progress bar
    for event_id, event in tqdm(events_data.items(), total=total_events):
        source_text = event.get('input', '')
        summary_text = event.get('summary', '')
        
        # Skip events without input or summary
        if not source_text or not summary_text:
            print(f"Skipping event {event_id}: Missing input or summary text")
            continue
        
        # Extract entities from source and summary
        source_entities = extract_entities(source_text, nlp)
        summary_entities = extract_entities(summary_text, nlp)
        
        # Find common entities between source and summary
        common_entity_names = set(source_entities["entity_to_sentences"].keys()) & set(summary_entities["entity_to_sentences"].keys())
        if not common_entity_names:
            print(f"No common entities found between source and summary for event {event_id}")
            continue
        
        # Calculate frequency for each common entity in source
        common_entity_counts = {entity: source_entities["entity_counts"][entity] 
                               for entity in common_entity_names}
        
        # Sort common entities by source frequency
        sorted_common_entities = sorted(common_entity_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Apply top-k filter to common entities
        if top_k and top_k > 0 and top_k < len(sorted_common_entities):
            top_common_entities = [entity for entity, count in sorted_common_entities[:top_k]]
        else:
            top_common_entities = [entity for entity, count in sorted_common_entities]
        
        # Calculate sentiment for each common entity in source and summary
        source_sentiments = {}
        summary_sentiments = {}
        
        for entity in top_common_entities:
            # Get sentences containing this entity
            source_sentences = source_entities["entity_to_sentences"].get(entity, [])
            summary_sentences = summary_entities["entity_to_sentences"].get(entity, [])
            
            # Calculate sentiment distributions
            if source_sentences:
                source_sentiments[entity] = calculate_entity_sentiment(
                    entity, source_sentences, tsc)
            
            if summary_sentences:
                summary_sentiments[entity] = calculate_entity_sentiment(
                    entity, summary_sentences, tsc)
        
        # Calculate distance metrics between distributions
        entity_distances = {}
        
        for entity in top_common_entities:
            # Skip if entity doesn't have sentiment in both source and summary
            if (entity not in source_sentiments or 
                entity not in summary_sentiments or
                source_sentiments[entity]["total_mentions"] == 0 or
                summary_sentiments[entity]["total_mentions"] == 0):
                continue
            
            source_dist = [
                source_sentiments[entity]["sentiment_distribution"]["positive"],
                source_sentiments[entity]["sentiment_distribution"]["negative"],
                source_sentiments[entity]["sentiment_distribution"]["neutral"]
            ]
            
            summary_dist = [
                summary_sentiments[entity]["sentiment_distribution"]["positive"],
                summary_sentiments[entity]["sentiment_distribution"]["negative"],
                summary_sentiments[entity]["sentiment_distribution"]["neutral"]
            ]
            
            # Calculate distance metrics
            wasserstein = custom_wasserstein_distance(source_dist, summary_dist)
            kl_src_to_sum = kl_divergence(source_dist, summary_dist)
            kl_sum_to_src = kl_divergence(summary_dist, source_dist)
            
            # Calculate sentiment shifts
            pos_shift = summary_dist[0] - source_dist[0]
            neg_shift = summary_dist[1] - source_dist[1]
            neu_shift = summary_dist[2] - source_dist[2]
            
            # Determine the overall sentiment shift category
            sentiment_shift = determine_sentiment_shift(source_dist, summary_dist)
            
            entity_distances[entity] = {
                "wasserstein": float(wasserstein),
                "kl_divergence_src_to_sum": float(kl_src_to_sum),
                "kl_divergence_sum_to_src": float(kl_sum_to_src),
                "source_distribution": source_sentiments[entity]["sentiment_distribution"],
                "summary_distribution": summary_sentiments[entity]["sentiment_distribution"],
                "source_mentions": source_sentiments[entity]["total_mentions"],
                "summary_mentions": summary_sentiments[entity]["total_mentions"],
                "positive_shift": float(pos_shift),
                "negative_shift": float(neg_shift),
                "neutral_shift": float(neu_shift),
                "sentiment_shift": sentiment_shift
            }
        
        # Calculate average metrics and collect shift categories
        if entity_distances:
            avg_wasserstein = np.mean([d["wasserstein"] for d in entity_distances.values()])
            avg_kl_src_to_sum = np.mean([d["kl_divergence_src_to_sum"] for d in entity_distances.values()])
            avg_kl_sum_to_src = np.mean([d["kl_divergence_sum_to_src"] for d in entity_distances.values()])
            
            # Calculate average directional shifts
            avg_pos_shift = np.mean([d["positive_shift"] for d in entity_distances.values()])
            avg_neg_shift = np.mean([d["negative_shift"] for d in entity_distances.values()])
            avg_neu_shift = np.mean([d["neutral_shift"] for d in entity_distances.values()])
            
            # Count shift categories
            shift_categories = Counter([d["sentiment_shift"] for d in entity_distances.values()])
            
            avg_distances = {
                "wasserstein": float(avg_wasserstein),
                "kl_divergence_src_to_sum": float(avg_kl_src_to_sum),
                "kl_divergence_sum_to_src": float(avg_kl_sum_to_src),
                "avg_positive_shift": float(avg_pos_shift),
                "avg_negative_shift": float(avg_neg_shift),
                "avg_neutral_shift": float(avg_neu_shift),
                "shift_categories": dict(shift_categories),
                "most_common_shift": shift_categories.most_common(1)[0][0] if shift_categories else "balanced"
            }
        else:
            # Default values if no entity distances
            avg_distances = {
                "wasserstein": 0.0,
                "kl_divergence_src_to_sum": 0.0,
                "kl_divergence_sum_to_src": 0.0,
                "avg_positive_shift": 0.0,
                "avg_negative_shift": 0.0,
                "avg_neutral_shift": 0.0,
                "shift_categories": {},
                "most_common_shift": "balanced"
            }
        
        # Store results for this event
        results[event_id] = {
            "source_entities_count": len(source_entities["entity_counts"]),
            "summary_entities_count": len(summary_entities["entity_counts"]),
            "common_entities_count": len(common_entity_names),
            "top_k_common_entities_used": min(top_k if top_k else len(common_entity_names), len(common_entity_names)),
            "entity_distances": entity_distances,
            "average_distances": avg_distances
        }
        
    # Calculate aggregate metrics across all events
    all_wasserstein = [event["average_distances"]["wasserstein"] for event in results.values() if event["average_distances"]["wasserstein"] > 0]
    all_kl_src_to_sum = [event["average_distances"]["kl_divergence_src_to_sum"] for event in results.values() if event["average_distances"]["kl_divergence_src_to_sum"] > 0]
    all_kl_sum_to_src = [event["average_distances"]["kl_divergence_sum_to_src"] for event in results.values() if event["average_distances"]["kl_divergence_sum_to_src"] > 0]
    
    # Collect directional shift data
    all_pos_shifts = [event["average_distances"].get("avg_positive_shift", 0) for event in results.values()]
    all_neg_shifts = [event["average_distances"].get("avg_negative_shift", 0) for event in results.values()]
    all_neu_shifts = [event["average_distances"].get("avg_neutral_shift", 0) for event in results.values()]
    
    # Collect shift categories
    all_shifts = []
    for event in results.values():
        if "shift_categories" in event["average_distances"]:
            for shift, count in event["average_distances"]["shift_categories"].items():
                all_shifts.extend([shift] * count)
    
    # Get most common shift across all events
    shift_counter = Counter(all_shifts)
    most_common_overall = shift_counter.most_common(1)[0][0] if shift_counter else "balanced"
    
    aggregate_metrics = {
        "average_distances": {
            "wasserstein": float(np.mean(all_wasserstein)) if all_wasserstein else 0,
            "kl_divergence_src_to_sum": float(np.mean(all_kl_src_to_sum)) if all_kl_src_to_sum else 0,
            "kl_divergence_sum_to_src": float(np.mean(all_kl_sum_to_src)) if all_kl_sum_to_src else 0,
            "avg_positive_shift": float(np.mean(all_pos_shifts)) if all_pos_shifts else 0,
            "avg_negative_shift": float(np.mean(all_neg_shifts)) if all_neg_shifts else 0, 
            "avg_neutral_shift": float(np.mean(all_neu_shifts)) if all_neu_shifts else 0,
            "overall_shift_counts": dict(shift_counter),
            "most_common_shift": most_common_overall
        },
        "events_with_common_entities": len([e for e in results.values() if e["entity_distances"]]),
        "total_events_processed": len(results)
    }
    
    # Prepare final output
    final_output = {
        "metadata": {
            "aggregate_metrics": aggregate_metrics,
            "total_events_analyzed": len(events_data),
            "events_with_results": len(results),
            "top_k_common_entities": top_k
        },
        "events": results
    }
    
    return final_output

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare entity sentiment between source and summary texts.')
    parser.add_argument('--input', required=True, help='Path to input JSON file with events')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--top_k', type=int, default=5, help='Only consider top-k entities from source by frequency')
    args = parser.parse_args()
    
    # Download the necessary data for tokenization
    nltk.download('punkt', quiet=True)
    
    # Load spaCy model
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Configure spaCy for better NER performance
    nlp.max_length = 2000000
    
    print("Loading sentiment model...")
    tsc = TargetSentimentClassifier()
    
    # Process file
    print(f"Processing events from {args.input}...")
    results = process_file(args.input, args.top_k, tsc, nlp)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save results to output file
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Processed {results['metadata']['events_with_results']} events with results out of {results['metadata']['total_events_analyzed']} total events")
    
    # Print aggregate metrics
    print("\nAggregate Metrics:")
    for metric, value in results["metadata"]["aggregate_metrics"]["average_distances"].items():
        if not isinstance(value, dict):
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

if __name__ == "__main__":
    main()