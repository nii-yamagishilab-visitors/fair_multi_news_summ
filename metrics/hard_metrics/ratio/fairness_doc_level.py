from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json
import argparse
import logging
from scipy import stats
import ot  
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate(text, tokenizer, model, device):
    """Calculate political bias distribution using pre-trained model"""
    try:
        # Handle empty text
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for bias calculation")
            return {"left": 0.33, "center": 0.33, "right": 0.33}
            
        # Tokenize with truncation to handle longer sequences
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # Get confidence scores using softmax
        confidence_scores = logits.softmax(dim=-1)[0].cpu().tolist()

        # Map scores to classes
        classes = ["left", "center", "right"]
        results = dict(zip(classes, confidence_scores))
        
        # Print scores with labels
        for cls, score in results.items():
            logger.info(f"{cls}: {score:.4f}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in bias calculation: {e}")
        # Return balanced distribution on error
        return {"left": 0.33, "center": 0.33, "right": 0.33}

def normalize_dict(d):
    """Ensure dictionary values sum to 1.0"""
    total = sum(d.values())
    return {k: v / total for k, v in d.items()} if total != 1.0 else d

def align_dicts(dict1, dict2):
    """Ensure both dictionaries have the same keys (case-insensitive)"""
    # Create mapping of lowercase -> original case for both dicts
    keys1_lower = {k.lower(): k for k in dict1.keys()}
    keys2_lower = {k.lower(): k for k in dict2.keys()}
    
    # Get the union of all keys (lowercase)
    all_keys_lower = set(keys1_lower.keys()) | set(keys2_lower.keys())
    
    # Create new dicts with aligned keys
    aligned1 = {}
    aligned2 = {}
    
    for k_lower in all_keys_lower:
        # Use original key if available, otherwise use lowercase
        k1 = keys1_lower.get(k_lower, k_lower)
        k2 = keys2_lower.get(k_lower, k_lower)
        
        # Add to aligned dicts (with 0 if key doesn't exist in original)
        aligned1[k_lower] = dict1.get(k1, 0)
        aligned2[k_lower] = dict2.get(k2, 0)
    
    return normalize_dict(aligned1), normalize_dict(aligned2)

def kl_divergence(dict1, dict2):
    """Calculate KL divergence: KL(dict1 || dict2)"""
    aligned1, aligned2 = align_dicts(dict1, dict2)
    
    # Convert to arrays in the same order
    keys = sorted(aligned1.keys())
    p = np.array([aligned1[k] for k in keys])
    q = np.array([aligned2[k] for k in keys])
    
    # Avoid division by zero in log
    epsilon = 1e-10
    q = np.maximum(q, epsilon)
    
    return np.sum(p * np.log(p / q + epsilon))

def wasserstein_distance(dict1, dict2):
    """Calculate Wasserstein distance between two distributions"""
    aligned1, aligned2 = align_dicts(dict1, dict2)
    
    # Convert to arrays in the same order
    keys = sorted(aligned1.keys())
    p = np.array([aligned1[k] for k in keys])
    q = np.array([aligned2[k] for k in keys])
    
    # Use optimal transport with uniform costs between bins
    # because the other approach would treat the categories as if they were ordered on a line
    # we treat them as unordered
    n = len(keys)
    cost_matrix = np.ones((n, n)) - np.eye(n)  # Cost 1 to move between different bins, 0 for same bin
    return ot.emd2(p, q, cost_matrix)

def load_json_from_file(file_path):
    """Load JSON data from a file with all events"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    events = []
    for event_id, event in data.items():
        # Directly access the bias_distribution and summary for each event
        input_dist = event.get('bias_distribution')
        summary_text = event.get('summary', '')
        
        if input_dist:  # Only include events that have a bias_distribution
            events.append({
                "event_id": event_id,
                "input_dist": input_dist,
                "summary_text": summary_text
            })
    
    return events

def load_model(model_name="bucketresearch/politicalBiasBERT", tokenizer_name="bert-base-cased", use_cache=True):
    """Load model and tokenizer with caching option"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    
    # Create cache directory if it doesn't exist
    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Load tokenizer and model
    if use_cache:
        logger.info(f"Loading tokenizer and model with cache at {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        logger.info("Loading tokenizer and model without cache")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return tokenizer, model.to(device), device

def main():
    parser = argparse.ArgumentParser(description='Calculate bias metrics from JSON file')
    parser.add_argument('json_file', help='Path to the JSON file containing input distribution and text')
    parser.add_argument('--no-cache', action='store_true', help='Disable model caching')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--output', '-o', help='Output JSON file to save results')
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load model and tokenizer
        tokenizer, model, device = load_model(use_cache=not args.no_cache)
        
        # Load all events from JSON file
        events = load_json_from_file(args.json_file)
        
        if not events:
            logger.error("No valid events with bias_distribution found in the JSON file")
            return
        
        logger.info(f"Found {len(events)} events to process")
        
        # Dictionary to store results for all events
        all_results = {}
        
        # Process each event
        for i, event in enumerate(events):
            event_id = event["event_id"]
            input_dist = event["input_dist"]
            summary_text = event["summary_text"]
            
            logger.info(f"\nProcessing event {i+1}/{len(events)}: {event_id}")
            logger.info(f"Input Distribution: {input_dist}")
            logger.info(f"Summary Text Length: {len(summary_text)} characters")
            logger.debug(f"Summary Text Sample: {summary_text[:100]}...")
            
            # Calculate output distribution from text using the model
            output_dist = calculate(summary_text, tokenizer, model, device)
            logger.info(f"Output Distribution: {output_dist}")
            
            # Calculate metrics
            kl = kl_divergence(input_dist, output_dist)
            wd = wasserstein_distance(input_dist, output_dist)
            
            # Print results for this event
            print(f"\nResults for event {event_id}:")
            print(f"Input Distribution: {input_dist}")
            print(f"Output Distribution: {output_dist}")
            print(f"KL Divergence: {kl:.6f}")
            print(f"Wasserstein Distance: {wd:.6f}")
            
            # Store results
            all_results[event_id] = {
                "input_dist": input_dist,
                "output_dist": output_dist,
                "kl_divergence": kl,
                "wasserstein_distance": wd
            }
        
        # Calculate average metrics
        avg_kl = sum(result["kl_divergence"] for result in all_results.values()) / len(all_results)
        avg_wd = sum(result["wasserstein_distance"] for result in all_results.values()) / len(all_results)

        
        print(f"\nSummary for {args.json_file}:")
        print(f"Processed {len(events)} events")
        print(f"Average KL Divergence: {avg_kl:.6f}")
        print(f"Average Wasserstein Distance: {avg_wd:.6f}")
                
        # Save results to file if output path is provided
        if args.output:
            try:
                # Create directory if it doesn't exist
                output_dir = os.path.dirname(args.output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    logger.info(f"Created directory: {output_dir}")
                
                # Create a results dictionary that includes per-event results and summary metrics
                final_results = {
                    "events": all_results,
                    "metadata": {
                        "count": len(events),
                        "avg_kl": avg_kl,
                        "avg_wd": avg_wd
                    }
                }
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=2)
                print(f"Results saved to {args.output}")
            except Exception as e:
                logger.error(f"Error saving results to {args.output}: {e}")
                print(f"Error saving results: {e}")

        
        # Return values for potential use in other scripts
        return {
            "results": all_results,
            "metadata": {
                "count": len(events),
                "avg_kl": avg_kl,
                "avg_wd": avg_wd
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing the JSON file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

    # python fairness_doc_level.py "/home/smg/v-amber/all_news_generation/generated_summary/random/Llama-3.3-70B-Instruct/Llama-3.3-70B-Instruct/summaries.json" --output /home/smg/v-amber/evaluation/output/hard/ratio/results.json --no-cache
    # python fairness_doc_level.py "/home/smg/v-amber/all_news_generation/generated_summary/random/Llama-3.3-70B-Instruct/summaries.json" --output /home/smg/v-amber/evaluation/output/hard/ratio/random/Llama-3.3-70B-Instruct/results.json --no-cache
    # # Basic usage - process all events and show results
    # python political_bias_analyzer.py "/path/to/summaries.json"

    # # Save detailed results to a file
    # python political_bias_analyzer.py "/path/to/summaries.json" --output results.json 

    # # With verbose logging
    # python political_bias_analyzer.py "/path/to/summaries.json" --verbose