"""
Text Preprocessing:

Articles are first preprocessed using the preprocess_text method
This converts text to lowercase, removes punctuation
Tokenizes the text, removes stopwords, and applies stemming to normalize words


TF-IDF Vectorization:

The preprocessed article titles are converted to TF-IDF vectors using scikit-learn's TfidfVectorizer
This transforms the text into numerical vectors that capture word importance


Cosine Similarity Calculation:

The code calculates the cosine similarity between all pairs of article TF-IDF vectors
This creates a similarity matrix where each cell [i,j] represents how similar article i is to article j
Values range from 0 (completely different) to 1 (identical)

Threshold Application:

Articles are grouped together only if their similarity score is greater than or equal to the threshold (0.6)

Clustering Logic:

The algorithm groups articles in a greedy fashion
It starts with one article and adds all others that are sufficiently similar to it
Only clusters with at least 2 articles are kept (only cluster with more than a single article will be kept, because this is a multi document summarisation)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from datetime import datetime, timedelta
import re
from collections import defaultdict
import hashlib

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

class MultiNewsDatasetCreator:
    def __init__(self, news_dataset_path, similarity_threshold=0.5, time_window_days=3):
        """
        Initialize the Multi-News Dataset Creator
        
        Parameters:
        -----------
        news_dataset_path : str
            Path to the All the News 2.0 dataset CSV file
        similarity_threshold : float
            Threshold for cosine similarity to consider articles related (0-1)
        time_window_days : int
            Number of days to consider for grouping related articles
        """
        self.news_dataset_path = news_dataset_path
        self.similarity_threshold = similarity_threshold
        self.time_window_days = time_window_days
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def preprocess_text(self, text):
        """Preprocess text for better matching"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        
        return " ".join(tokens)
    
    def compute_text_hash(self, text):
        """Compute a hash for text to quickly identify exact duplicates"""
        if not isinstance(text, str):
            return ""
        
        # Normalize text: lowercase, remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Create hash
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def remove_exact_duplicates(self, df):
        """Remove exact duplicate articles from the dataset"""
        print("Removing exact duplicates...")
        initial_count = len(df)
        
        # Remove exact duplicates using article content hash
        df['content_hash'] = df['article'].apply(self.compute_text_hash)
        df = df.drop_duplicates(subset=['content_hash'])
        
        # Also deduplicate by title
        df['title_hash'] = df['title'].apply(self.compute_text_hash)
        df = df.drop_duplicates(subset=['title_hash'])
        
        # Clean up temporary columns
        df = df.drop(columns=['content_hash', 'title_hash'])
        
        exact_duplicate_count = initial_count - len(df)
        print(f"Removed {exact_duplicate_count} exact duplicates ({exact_duplicate_count/initial_count*100:.2f}% of original data)")
        
        return df
    
    def load_and_preprocess_data(self, sample_size=None):
        """Load and preprocess the news dataset"""
        print("Loading news dataset...")
        
        # Use chunksize for large datasets
        chunks = []
        # Define columns to read - include all relevant fields from the dataset
        columns = ['idx', 'article_idx', 'title', 'date', 'article', 'publication', 
                   'author', 'url', 'year', 'month', 'day', 'section']
        
        for chunk in pd.read_csv(self.news_dataset_path, chunksize=10000):
            # Keep only columns that exist in the dataset
            available_columns = [col for col in columns if col in chunk.columns]
            chunk = chunk[available_columns]
            
            # Filter out rows with missing values in essential columns
            chunk = chunk.dropna(subset=['title', 'date', 'article'])
            
            # Convert date strings to datetime objects
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk = chunk.dropna(subset=['date'])
            
            # Take a sample if specified
            if sample_size and len(chunks) * 10000 + len(chunk) > sample_size:
                chunk = chunk.sample(max(0, sample_size - len(chunks) * 10000))
                chunks.append(chunk)
                break
                
            chunks.append(chunk)
            
            if sample_size and len(chunks) * 10000 >= sample_size:
                break
        
        df = pd.concat(chunks)
        print(f"Loaded {len(df)} articles")
        
        # Remove exact duplicates
        df = self.remove_exact_duplicates(df)
        
        # Add preprocessed text columns
        df['preprocessed_title'] = df['title'].apply(self.preprocess_text)
        
        # Sort by date for easier time window processing
        df = df.sort_values('date')
        
        return df
    
    def cluster_articles_by_date_and_similarity(self, df):
        """Group articles by date window and then by content similarity"""
        print("Clustering articles...")
        
        # Initialize event clusters
        event_clusters = []
        
        # Process articles in chronological order
        date_groups = defaultdict(list)
        
        # Group articles by date
        for _, article in df.iterrows():
            date_key = article['date'].strftime('%Y-%m-%d')
            date_groups[date_key].append(article)
        
        # Process each date group
        for date_key, articles in date_groups.items():
            current_date = datetime.strptime(date_key, '%Y-%m-%d')
            
            # Find related articles within the time window
            window_articles = []
            for d in range(self.time_window_days):
                window_date = (current_date - timedelta(days=d)).strftime('%Y-%m-%d')
                if window_date in date_groups:
                    window_articles.extend(date_groups[window_date])
            
            if len(window_articles) < 2:
                continue
                
            # Create TF-IDF vectors for title comparison
            vectorizer = TfidfVectorizer()
            titles = [article['preprocessed_title'] for article in window_articles]
            
            # Skip if all titles are empty after preprocessing
            if all(not title for title in titles):
                continue
                
            tfidf_matrix = vectorizer.fit_transform(titles)
            
            # Calculate pairwise similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Group similar articles
            processed_indices = set()
            
            for i in range(len(window_articles)):
                if i in processed_indices:
                    continue
                    
                # Find similar articles
                cluster = [i]
                processed_indices.add(i)
                
                for j in range(i+1, len(window_articles)):
                    if j not in processed_indices and similarity_matrix[i, j] >= self.similarity_threshold:
                        cluster.append(j)
                        processed_indices.add(j)
                
                # Only keep clusters with multiple articles
                if len(cluster) >= 2:
                    cluster_articles = [window_articles[idx] for idx in cluster]
                    event_clusters.append(cluster_articles)
        
        print(f"Created {len(event_clusters)} event clusters")
        return event_clusters
    
    def create_multi_news_dataset(self, event_clusters):
        """Create the final multi-news dataset from event clusters"""
        print("Creating multi-news dataset...")
        
        multi_news_data = []
        event_data = []
        
        # First, assign unique article IDs to all articles regardless of event
        all_articles = []
        for cluster in event_clusters:
            all_articles.extend(cluster)
        
        # Create a dictionary to map each article to its unique ID
        article_id_map = {}
        for article_idx, article in enumerate(all_articles, 1):
            # Create a unique identifier for the article
            # Using the article's hash as part of the ID to ensure uniqueness
            article_hash = self.compute_text_hash(article['title'] + str(article['date']))[:8]
            unique_article_id = f"ARTICLE_{article_idx:08d}_{article_hash}"
            
            # Store the mapping
            article_id_map[id(article)] = unique_article_id
        
        # Create a unique event ID for each cluster
        for event_id, cluster in enumerate(event_clusters, 1):
            # Sort articles by length (descending) to prioritize more detailed articles
            cluster.sort(key=lambda x: len(x['article']), reverse=True)
            
            # Create an event entry
            event_entry = {
                'event_id': f"EVENT_{event_id:06d}",
                'event_date': min(article['date'] for article in cluster),
                'num_articles': len(cluster),
                'article_ids': [article_id_map[id(article)] for article in cluster]
            }
            
            event_data.append(event_entry)
            
            # Create individual article entries
            for article in cluster:
                article_entry = {
                    'article_id': article_id_map[id(article)],
                    'event_id': f"EVENT_{event_id:06d}",  # Link to event
                    'date': article['date'],
                    'title': article['title'],
                    'article': article['article'],
                    'publication': article.get('publication', '')
                }
                
                # Add optional fields if they exist in the original data
                if 'url' in article and not pd.isna(article['url']):
                    article_entry['url'] = article['url']
                
                if 'author' in article and not pd.isna(article['author']):
                    article_entry['author'] = article['author']
                
                if 'section' in article and not pd.isna(article['section']):
                    article_entry['section'] = article['section']
                
                # Only include year, month, day if they are present in the dataset
                date_components = {}
                for comp in ['year', 'month', 'day']:
                    if comp in article and not pd.isna(article[comp]):
                        article_entry[comp] = article[comp]
                
                multi_news_data.append(article_entry)
        
        # Create two DataFrames - one for events and one for articles
        event_df = pd.DataFrame(event_data)
        article_df = pd.DataFrame(multi_news_data)
        
        # Sort by date
        event_df = event_df.sort_values('event_date')
        article_df = article_df.sort_values(['date'])  # Sort by date only, not by event_id
        
        return event_df, article_df
    
    def run_pipeline(self, output_path_prefix, sample_size=None):
        """Run the complete pipeline to create the multi-news dataset"""
        # Load and preprocess data
        df = self.load_and_preprocess_data(sample_size)
        
        # Cluster articles by date and similarity
        event_clusters = self.cluster_articles_by_date_and_similarity(df)
        
        # Create the multi-news dataset
        event_df, article_df = self.create_multi_news_dataset(event_clusters)
        
        # Save the datasets
        events_path = f"{output_path_prefix}_events.csv"
        articles_path = f"{output_path_prefix}_articles.csv"
        
        event_df.to_csv(events_path, index=False)
        article_df.to_csv(articles_path, index=False)
        
        print(f"Multi-news dataset created with {len(event_df)} events and {len(article_df)} articles")
        print(f"Event data saved to: {events_path}")
        print(f"Article data saved to: {articles_path}")
        
        return event_df, article_df

# Method for date filtering (to be used in the batch processing)
def load_and_preprocess_data_with_date_filter(self, start_date, end_date):
    """Load and preprocess the news dataset with date filtering"""
    print(f"Loading news from {start_date.date()} to {end_date.date()}...")
    
    # Use chunksize for large datasets
    chunks = []
    # Define columns to read - include all relevant fields from the dataset
    columns = ['title', 'date', 'article', 'publication', 
               'author', 'url', 'year', 'month', 'day', 'section']
    
    for chunk in pd.read_csv(self.news_dataset_path, chunksize=100000):
        # Keep only columns that exist in the dataset
        available_columns = [col for col in columns if col in chunk.columns]
        chunk = chunk[available_columns]
        
        # Filter out rows with missing values in essential columns
        chunk = chunk.dropna(subset=['title', 'date', 'article'])
        
        # Convert date strings to datetime objects
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        
        # Apply date filter
        chunk = chunk[(chunk['date'] >= start_date) & (chunk['date'] <= end_date)]
        
        if not chunk.empty:
            chunks.append(chunk)
    
    if not chunks:
        return pd.DataFrame()
        
    df = pd.concat(chunks)
    print(f"Loaded {len(df):,} articles within date range")
    
    # Remove exact duplicates
    df = self.remove_exact_duplicates(df)
    
    # Add preprocessed text columns
    df['preprocessed_title'] = df['title'].apply(self.preprocess_text)
    
    # Sort by date for easier time window processing
    df = df.sort_values('date')
    
    return df

# Example usage
if __name__ == "__main__":
    creator = MultiNewsDatasetCreator(
        news_dataset_path="all-the-news-2-1.csv",
        similarity_threshold=0.6,      # Threshold for event clustering
        time_window_days=3             # Time window for related articles
    )
    
    event_df, article_df = creator.run_pipeline(
        output_path_prefix="multi_news_dataset",
        sample_size=100000  # Adjust based on available memory
    )
    
    # Print statistics
    print("\nEvent Statistics:")
    print(f"Total number of events: {len(event_df)}")
    print(f"Average number of articles per event: {event_df['num_articles'].mean():.2f}")
    print(f"Max number of articles for an event: {event_df['num_articles'].max()}")
    print(f"Min number of articles for an event: {event_df['num_articles'].min()}")
    
    # Print a sample of the data structure
    print("\nSample Event Record:")
    print(event_df.iloc[0].to_dict())
    
    print("\nCorresponding Article Records:")
    sample_event_id = event_df.iloc[0]['event_id']
    sample_articles = article_df[article_df['event_id'] == sample_event_id]
    for i, article in enumerate(sample_articles.itertuples(), 1):
        print(f"\nArticle {i}:")
        print(f"  ID: {article.article_id}")
        print(f"  Publication: {article.publication}")
        print(f"  Title: {article.title}")
        print(f"  Date: {article.date}")
        # Truncate the article text for display
        article_preview = article.article[:100] + "..." if len(article.article) > 100 else article.article
        print(f"  Preview: {article_preview}")