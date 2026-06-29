from NewsSentiment import TargetSentimentClassifier
import json
import nltk
from nltk.tokenize import sent_tokenize
import argparse
import torch
import numpy as np
import os
import time
from tqdm import tqdm

def load_json_from_file(file_path):
    """Load JSON data from a file with all events"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def break_into_individual_sentences(text):
    # Handle empty or None text
    if not text:
        return []
    
    # Tokenize the text into sentences
    try:
        sentences = sent_tokenize(text)
        return sentences
    except Exception as e:
        print(f"Error tokenizing text: {str(e)}")
        return []

def calculate_sentiment_distribution(sentences, tsc):
    # If no sentences, return empty distribution
    total_sentences = len(sentences)
    if total_sentences == 0:
        return {
            "sentiment_counts": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            },
            "sentiment_proportions": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            },
            "total_sentences": 0,
            "skipped_sentences": 0
        }
    
    # Initialize sentiment counts
    sentiment_counts = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }
    
    processed_sentences = 0
    skipped_sentences = 0
    
    # Process sentences in smaller batches to avoid TooLongTextException
    batch_size = 10  # Adjust this value based on your needs
    
    for i in range(0, total_sentences, batch_size):
        batch_sentences = sentences[i:i+batch_size]
        
        # Process each sentence individually to prevent exceptions
        for sent in batch_sentences:
            try:
                # Skip empty sentences
                if not sent or len(sent.strip()) == 0:
                    skipped_sentences += 1
                    continue
                
                # Make sure the sentence isn't too long
                if len(sent) > 500:  # Limit sentence length
                    sent = sent[:500]
                
                # Prepare data for sentiment analysis - format required by NewsSentiment
                data = [("", sent, "")]
                
                # Get sentiment for the sentence
                sentiment_result = tsc.infer(targets=data)
                sentiment_class = sentiment_result[0][0]['class_label']
                
                # Count sentiment class
                if sentiment_class in sentiment_counts:
                    sentiment_counts[sentiment_class] += 1
                
                processed_sentences += 1
                
            except Exception as e:
                print(f"Error processing sentence: {str(e)}")
                skipped_sentences += 1
                # Continue with the next sentence
                continue
    
    # Calculate proportions (rounded to 4 decimal places)
    sentiment_proportions = {}
    
    if processed_sentences > 0:
        for sentiment, count in sentiment_counts.items():
            sentiment_proportions[sentiment] = round(count / processed_sentences, 4)
    else:
        sentiment_proportions = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
    
    return {
        "sentiment_counts": sentiment_counts,
        "sentiment_proportions": sentiment_proportions,
        "total_sentences": processed_sentences,
        "skipped_sentences": skipped_sentences
    }
    
def process_and_save_sentiment_data(input_path, output_path, tsc):
    # Load the input JSON
    events_data = load_json_from_file(input_path)
    results = {}
    
    # Track aggregate sentiment data
    summary_sentiment_data = {
        "positive": [],
        "negative": [],
        "neutral": []
    }
    
    input_sentiment_data = {
        "positive": [],
        "negative": [],
        "neutral": []
    }
    
    # Process each event
    total_events = len(events_data)
    print(f"Processing {total_events} events...")
    
    # Use tqdm for a progress bar
    for event_id, event in tqdm(events_data.items(), total=total_events):
        # Initialize result structure for this event
        results[event_id] = {
            "sentiment_analysis": {}
        }
        
        # Process summary text
        summary_text = event.get('summary', '')
        if summary_text:
            # Break the summary into sentences
            summary_sentences = break_into_individual_sentences(summary_text)
            
            # Calculate sentiment distribution for summary
            summary_sentiment_result = calculate_sentiment_distribution(summary_sentences, tsc)
            
            # Track sentiment proportions for average calculation
            for sentiment, proportion in summary_sentiment_result["sentiment_proportions"].items():
                summary_sentiment_data[sentiment].append(proportion)
            
            # Add to results
            results[event_id]["sentiment_analysis"]["summary"] = summary_sentiment_result
        else:
            # Add empty results for events without summaries
            results[event_id]["sentiment_analysis"]["summary"] = {
                "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
                "sentiment_proportions": {"positive": 0, "negative": 0, "neutral": 0},
                "total_sentences": 0,
                "skipped_sentences": 0
            }
        
        # Process input text
        input_text = event.get('input', '')
        if input_text:
            # Break the input into sentences
            input_sentences = break_into_individual_sentences(input_text)
            
            # Calculate sentiment distribution for input
            input_sentiment_result = calculate_sentiment_distribution(input_sentences, tsc)
            
            # Track sentiment proportions for average calculation
            for sentiment, proportion in input_sentiment_result["sentiment_proportions"].items():
                input_sentiment_data[sentiment].append(proportion)
            
            # Add to results
            results[event_id]["sentiment_analysis"]["input"] = input_sentiment_result
        else:
            # Add empty results for events without input
            results[event_id]["sentiment_analysis"]["input"] = {
                "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
                "sentiment_proportions": {"positive": 0, "negative": 0, "neutral": 0},
                "total_sentences": 0,
                "skipped_sentences": 0
            }
        
        # Save intermediate results every 100 events for safety
        if len(results) % 100 == 0:
            intermediate_output = {
                "metadata": {
                    "intermediate_save": True,
                    "events_processed_so_far": len(results),
                    "total_events": total_events,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "events": results
            }
            
            temp_output_path = output_path + ".temp"
            os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
            with open(temp_output_path, 'w', encoding='utf-8') as file:
                json.dump(intermediate_output, file, indent=2)
            
            print(f"Intermediate results saved after processing {len(results)} events")
    
    # Calculate average sentiment proportions
    avg_sentiment_proportions = {
        "summary": {},
        "input": {}
    }
    
    for sentiment in ["positive", "negative", "neutral"]:
        # Calculate for summary
        if summary_sentiment_data[sentiment]:
            avg_summary_sentiment = np.mean(summary_sentiment_data[sentiment])
            avg_sentiment_proportions["summary"][sentiment] = round(float(avg_summary_sentiment), 4)
        else:
            avg_sentiment_proportions["summary"][sentiment] = 0
        
        # Calculate for input
        if input_sentiment_data[sentiment]:
            avg_input_sentiment = np.mean(input_sentiment_data[sentiment])
            avg_sentiment_proportions["input"][sentiment] = round(float(avg_input_sentiment), 4)
        else:
            avg_sentiment_proportions["input"][sentiment] = 0
    
    # Add metadata to results
    metadata = {
        "metadata": {
            "average_sentiment_proportions": avg_sentiment_proportions,
            "total_events_processed": total_events,
            "events_with_summaries": len(summary_sentiment_data["neutral"]),
            "events_with_input": len(input_sentiment_data["neutral"])
        }
    }
    
    # Combine metadata and results
    final_output = {**metadata, "events": results}
    
    # Save results to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_output, file, indent=2)
    
    print(f"Results saved to {output_path}")
    print("Average sentiment proportions:")
    print(f"  Summary: Positive: {avg_sentiment_proportions['summary']['positive']}, " 
          f"Negative: {avg_sentiment_proportions['summary']['negative']}, "
          f"Neutral: {avg_sentiment_proportions['summary']['neutral']}")
    print(f"  Input: Positive: {avg_sentiment_proportions['input']['positive']}, "
          f"Negative: {avg_sentiment_proportions['input']['negative']}, "
          f"Neutral: {avg_sentiment_proportions['input']['neutral']}")
    print(f"Processed {len(summary_sentiment_data['neutral'])} events with summaries out of {total_events} total events")
    print(f"Processed {len(input_sentiment_data['neutral'])} events with input out of {total_events} total events")
    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process sentiment distribution in event summaries and inputs.')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()
    
    # Download the necessary data for tokenization (only needed once)
    nltk.download('punkt', quiet=True)
    
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("Using GPU for inference...")
    else:
        print("GPU not available, using CPU...")
    
    print("Loading sentiment model...")
    tsc = TargetSentimentClassifier()
    
    print(f"Processing events from {args.input}...")
    process_and_save_sentiment_data(args.input, args.output, tsc)

if __name__ == "__main__":
    main()


    # python test_sentiment_political.py --input "/home/smg/v-amber/all_news_generation/generated_summary/random/Llama-3.3-70B-Instruct/Llama-3.3-70B-Instruct/summaries.json" --output /home/smg/v-amber/evaluation/output/hard/neutralisation/results.json