from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import json
import nltk
from nltk.tokenize import sent_tokenize
import argparse
import torch
import numpy as np
import os

def load_json_from_file(file_path):
    """Load JSON data from a file with all events"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def break_into_individual_sentences(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    return sentences

def calculate_bias_distribution(sentences, nlp, tokenizer, max_length=512):
    # Initialize counters for each bias category
    counts = {"LEFT": 0, "RIGHT": 0, "CENTER": 0}
    processed_sentences = 0
    skipped_sentences = 0
    
    if len(sentences) == 0:
        return {
            "Left": 0,
            "Right": 0,
            "Center": 0,
            "equality_gap": 0
        }
    
    # Process each sentence and count the biases
    for sent in sentences:
        # Check if the sentence is too long for the model
        tokens = tokenizer.encode(sent, add_special_tokens=True)
        if len(tokens) > max_length:
            # Skip sentences that are too long
            skipped_sentences += 1
            continue
            
        try:
            result = nlp(sent)
            label = result[0]['label']
            counts[label] += 1
            processed_sentences += 1
        except Exception as e:
            print(f"Error processing sentence: {e}")
            skipped_sentences += 1
    
    # If no sentences were processed successfully, return zeros
    if processed_sentences == 0:
        print(f"Warning: No sentences were processed (skipped {skipped_sentences} sentences)")
        return {
            "Left": 0,
            "Right": 0,
            "Center": 0,
            "equality_gap": 0
        }
    
    # Calculate distribution percentages (rounded to 2 decimal places)
    distribution = {
        "Left": round(counts["LEFT"] / processed_sentences, 2),
        "Right": round(counts["RIGHT"] / processed_sentences, 2),
        "Center": round(counts["CENTER"] / processed_sentences, 2)
    }
    
    # Calculate the gap between highest and lowest distribution
    values = list(distribution.values())
    highest = max(values)
    lowest = min(values)
    gap = highest - lowest
    
    # Add the equality_gap field to the result
    distribution["equality_gap"] = round(gap, 2)
    
    if skipped_sentences > 0:
        print(f"Warning: Skipped {skipped_sentences} sentences due to length limitations")
    
    return distribution

def process_and_save_bias_data(input_path, output_path, nlp, tokenizer):
    # Load the input JSON
    events_data = load_json_from_file(input_path)
    results = {}
    equality_gaps = []
    
    # Process each event
    total_events = len(events_data)
    for i, (event_id, event) in enumerate(events_data.items()):
        if i % 50 == 0:
            print(f"Processing event {i+1}/{total_events}...")
            
        summary_text = event.get('summary', '')
        
        # Skip events without a summary
        if not summary_text:
            results[event_id] = {
                "bias_distribution": {
                    "Left": 0,
                    "Right": 0,
                    "Center": 0,
                    "equality_gap": 0
                }
            }
            continue
        
        # Break the summary into sentences
        sentences = break_into_individual_sentences(summary_text)
        
        # Calculate bias distribution
        bias_distribution = calculate_bias_distribution(sentences, nlp, tokenizer)
        
        # Track equality gaps for average calculation
        equality_gaps.append(bias_distribution["equality_gap"])
        
        # Add to results
        results[event_id] = {
            "bias_distribution": bias_distribution
        }
    
    # Calculate average equality gap
    avg_equality_gap = np.mean(equality_gaps) if equality_gaps else 0
    
    # Add metadata to results
    metadata = {
        "metadata": {
            "average_equality_gap": round(float(avg_equality_gap), 4),
            "total_events_processed": total_events,
            "events_with_summaries": len(equality_gaps)
        }
    }
    
    # Combine metadata and results
    final_output = {**metadata, "events": results}
    
    # Save results to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_output, file, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"Average equality gap: {round(float(avg_equality_gap), 4)}")
    print(f"Processed {len(equality_gaps)} events with summaries out of {total_events} total events")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process political bias in event summaries.')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    args = parser.parse_args()
    
    # Download the necessary data for tokenization (only needed once)
    nltk.download('punkt', quiet=True)
    
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("Using GPU for inference...")
    else:
        print("GPU not available, using CPU...")
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
    tokenizer = AutoTokenizer.from_pretrained("bucketresearch/politicalBiasBERT")
    
    # Set up pipeline with batch size for efficiency
    nlp = pipeline("text-classification", 
                  model=model, 
                  tokenizer=tokenizer, 
                  device=device,
                  batch_size=args.batch_size)
    
    print(f"Processing events from {args.input}...")
    process_and_save_bias_data(args.input, args.output, nlp, tokenizer)

if __name__ == "__main__":
    main()