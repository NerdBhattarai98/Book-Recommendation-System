"""
Training script for Book Recommendation System
Run this once to pre-train and save the models for faster startup.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz
import time

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)

def load_books():
    """Load and preprocess books data"""
    print("Loading books data...")
    books_path = os.path.join(DATA_DIR, 'books.csv')
    
    books = pd.read_csv(
        books_path, 
        sep=';', 
        encoding='latin-1', 
        on_bad_lines='skip',
        low_memory=False
    )
    
    # Standardize column names
    books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 
                     'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
    
    # Clean data
    books = books.dropna(subset=['ISBN', 'Book-Title'])
    books['ISBN'] = books['ISBN'].astype(str).str.strip()
    books['Book-Title'] = books['Book-Title'].fillna('').astype(str)
    books['Book-Author'] = books['Book-Author'].fillna('Unknown').astype(str)
    books['Publisher'] = books['Publisher'].fillna('Unknown').astype(str)
    
    # Remove duplicates
    books = books.drop_duplicates(subset=['ISBN'])
    
    print(f"Loaded {len(books)} books")
    return books

def load_ratings():
    """Load and preprocess ratings data"""
    print("Loading ratings data...")
    ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
    
    ratings = pd.read_csv(
        ratings_path, 
        sep=';', 
        encoding='latin-1',
        on_bad_lines='skip'
    )
    
    ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']
    ratings['ISBN'] = ratings['ISBN'].astype(str).str.strip()
    ratings['User-ID'] = pd.to_numeric(ratings['User-ID'], errors='coerce')
    ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')
    
    # Keep only explicit ratings (1-10, not 0)
    ratings = ratings[ratings['Book-Rating'] > 0]
    ratings = ratings.dropna()
    
    print(f"Loaded {len(ratings)} ratings")
    return ratings

def train_content_based_model(books, max_features=5000):
    """Train TF-IDF based content model"""
    print("\n--- Training Content-Based Model ---")
    start_time = time.time()
    
    # Create combined text features
    print("Creating text features...")
    books['combined_features'] = (
        books['Book-Title'].fillna('') + ' ' +
        books['Book-Author'].fillna('') + ' ' +
        books['Publisher'].fillna('')
    )
    
    # Build TF-IDF matrix
    print(f"Building TF-IDF matrix (max_features={max_features})...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(books['combined_features'])
    
    # Save models
    print("Saving content-based model...")
    with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    
    save_npz(os.path.join(MODELS_DIR, 'tfidf_matrix.npz'), csr_matrix(tfidf_matrix))
    
    # Save ISBN index mapping
    isbn_to_idx = {isbn: idx for idx, isbn in enumerate(books['ISBN'].values)}
    with open(os.path.join(MODELS_DIR, 'isbn_to_idx.pkl'), 'wb') as f:
        pickle.dump(isbn_to_idx, f)
    
    elapsed = time.time() - start_time
    print(f"Content-based model trained in {elapsed:.2f} seconds")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    return tfidf_matrix, isbn_to_idx

def train_collaborative_model(books, ratings, min_ratings=50):
    """Train collaborative filtering model using item-item similarity"""
    print("\n--- Training Collaborative Filtering Model ---")
    start_time = time.time()
    
    # Filter to books that exist in our books dataset
    valid_isbns = set(books['ISBN'].values)
    ratings_filtered = ratings[ratings['ISBN'].isin(valid_isbns)]
    
    print(f"Ratings after filtering: {len(ratings_filtered)}")
    
    # Get books with enough ratings
    book_rating_counts = ratings_filtered.groupby('ISBN').size()
    popular_books = book_rating_counts[book_rating_counts >= min_ratings].index
    ratings_filtered = ratings_filtered[ratings_filtered['ISBN'].isin(popular_books)]
    
    print(f"Books with >= {min_ratings} ratings: {len(popular_books)}")
    print(f"Ratings for popular books: {len(ratings_filtered)}")
    
    if len(ratings_filtered) == 0:
        print("Not enough ratings data for collaborative filtering!")
        return None, None
    
    # Create user-item matrix
    print("Building user-item matrix...")
    user_ids = ratings_filtered['User-ID'].unique()
    book_ids = ratings_filtered['ISBN'].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    book_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}
    idx_to_book = {idx: bid for bid, idx in book_to_idx.items()}
    
    rows = ratings_filtered['User-ID'].map(user_to_idx).values
    cols = ratings_filtered['ISBN'].map(book_to_idx).values
    data = ratings_filtered['Book-Rating'].values
    
    user_item_matrix = csr_matrix(
        (data, (rows, cols)), 
        shape=(len(user_ids), len(book_ids))
    )
    
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    
    # Compute item-item similarity (transpose for item-based)
    print("Computing item-item similarity matrix...")
    item_matrix = user_item_matrix.T  # items x users
    
    # Normalize for cosine similarity
    from sklearn.preprocessing import normalize
    item_matrix_normalized = normalize(item_matrix, axis=1)
    
    # Compute similarity (this can be memory intensive for large matrices)
    # For very large datasets, you might want to compute this in batches
    print("Computing cosine similarity...")
    item_similarity = cosine_similarity(item_matrix_normalized)
    
    # Save models
    print("Saving collaborative filtering model...")
    with open(os.path.join(MODELS_DIR, 'book_to_idx_collab.pkl'), 'wb') as f:
        pickle.dump(book_to_idx, f)
    
    with open(os.path.join(MODELS_DIR, 'idx_to_book_collab.pkl'), 'wb') as f:
        pickle.dump(idx_to_book, f)
    
    np.save(os.path.join(MODELS_DIR, 'item_similarity.npy'), item_similarity)
    
    elapsed = time.time() - start_time
    print(f"Collaborative model trained in {elapsed:.2f} seconds")
    print(f"Item similarity matrix shape: {item_similarity.shape}")
    
    return item_similarity, book_to_idx

def save_processed_books(books):
    """Save processed books dataframe"""
    print("\nSaving processed books data...")
    books.to_pickle(os.path.join(MODELS_DIR, 'books_processed.pkl'))
    print(f"Saved {len(books)} processed books")

def main():
    print("=" * 60)
    print("Book Recommendation System - Model Training")
    print("=" * 60)
    
    total_start = time.time()
    
    # Load data
    books = load_books()
    ratings = load_ratings()
    
    # Save processed books
    save_processed_books(books)
    
    # Train content-based model
    train_content_based_model(books, max_features=5000)
    
    # Train collaborative filtering model
    train_collaborative_model(books, ratings, min_ratings=20)
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"Training complete! Total time: {total_elapsed:.2f} seconds")
    print(f"Models saved to: {MODELS_DIR}")
    print("=" * 60)
    
    # List saved files
    print("\nSaved model files:")
    for f in os.listdir(MODELS_DIR):
        size = os.path.getsize(os.path.join(MODELS_DIR, f)) / (1024 * 1024)
        print(f"  - {f}: {size:.2f} MB")

if __name__ == '__main__':
    main()
