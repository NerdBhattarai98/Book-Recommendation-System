import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import os
import pickle
import json
import re


class RecommendationEngine:
    def __init__(self, data_path, mongo=None):
        self.data_path = data_path
        self.models_path = os.path.join(os.path.dirname(data_path), 'models')
        self.mongo = mongo  # MongoDB instance for accessing user reviews
        
        self.books_df = None
        self.ratings_df = None
        self.popular_books_df = None
        
        # Genre data - now loaded from CSV
        self.all_genres = set()
        
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.isbn_to_idx = None
        
        self.item_similarity = None
        self.book_to_idx_collab = None
        self.idx_to_book_collab = None
        
        if self._load_pretrained_models():
            print("Loaded pre-trained models")
        else:
            print("Building models from scratch...")
            self._load_data()
            self._build_models()
    
    def get_genre_for_book(self, isbn):
        """Get genre for a book by ISBN from loaded data"""
        if self.books_df is None:
            return ['fiction']
        book = self.books_df[self.books_df['isbn'] == isbn]
        if book.empty:
            return ['fiction']
        genre = book.iloc[0].get('genre', '')
        if pd.isna(genre) or not genre:
            return ['fiction']
        return [g.strip() for g in str(genre).split(',')[:3]]  # Top 3 genres
    
    def get_genre_for_title(self, title):
        """Try to infer genre from title keywords (fallback)"""
        title_lower = str(title).lower()
        genre_keywords = {
            'fiction': ['novel', 'story', 'tales'],
            'mystery': ['mystery', 'murder', 'detective', 'crime', 'thriller', 'suspense'],
            'fantasy': ['magic', 'dragon', 'wizard', 'witch', 'vampire', 'werewolf', 'fantasy', 'enchanted'],
            'romance': ['love', 'heart', 'passion', 'kiss', 'romance', 'wedding', 'bride'],
            'history': ['history', 'war', 'biography', 'memoir', 'life of', 'civil war'],
            'children': ['kids', 'children', 'child', 'boy', 'girl', 'little'],
            'young-adult': ['teen', 'young adult', 'high school'],
            'non-fiction': ['guide', 'how to', 'handbook', 'manual', 'introduction to'],
            'poetry': ['poems', 'poetry', 'verses'],
            'comics': ['comic', 'graphic novel', 'manga']
        }
        
        matched_genres = []
        for genre, keywords in genre_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    matched_genres.append(genre)
                    break
        
        return matched_genres if matched_genres else ['fiction']
    
    def _load_pretrained_models(self):
        required = ['books_processed.pkl', 'tfidf_vectorizer.pkl', 
                    'tfidf_matrix.npz', 'isbn_to_idx.pkl']
        
        for f in required:
            if not os.path.exists(os.path.join(self.models_path, f)):
                return False
        
        try:
            self.books_df = pd.read_pickle(os.path.join(self.models_path, 'books_processed.pkl'))
            
            with open(os.path.join(self.models_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            self.tfidf_matrix = load_npz(os.path.join(self.models_path, 'tfidf_matrix.npz'))
            
            with open(os.path.join(self.models_path, 'isbn_to_idx.pkl'), 'rb') as f:
                self.isbn_to_idx = pickle.load(f)
            
            collab_files = ['item_similarity.npy', 'book_to_idx_collab.pkl', 'idx_to_book_collab.pkl']
            if all(os.path.exists(os.path.join(self.models_path, f)) for f in collab_files):
                self.item_similarity = np.load(os.path.join(self.models_path, 'item_similarity.npy'))
                with open(os.path.join(self.models_path, 'book_to_idx_collab.pkl'), 'rb') as f:
                    self.book_to_idx_collab = pickle.load(f)
                with open(os.path.join(self.models_path, 'idx_to_book_collab.pkl'), 'rb') as f:
                    self.idx_to_book_collab = pickle.load(f)
            
            self._calculate_popular_books()
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def _load_data(self):
        """Load data from GoodReads_100k_books.csv"""
        books_file = os.path.join(self.data_path, 'GoodReads_100k_books.csv')
        if os.path.exists(books_file):
            try:
                print("Loading GoodReads_100k_books.csv...")
                self.books_df = pd.read_csv(books_file, encoding='utf-8', dtype=str)
                # Columns: author, bookformat, desc, genre, img, isbn, isbn13, link, pages, rating, reviews, title, totalratings
                
                # Rename columns to match expected format
                self.books_df = self.books_df.rename(columns={
                    'img': 'image_url',
                    'desc': 'description',
                    'totalratings': 'num_ratings'
                })
                
                # Clean and fill missing values
                self.books_df['isbn'] = self.books_df['isbn'].fillna(self.books_df['isbn13'])
                self.books_df = self.books_df.dropna(subset=['isbn'])
                self.books_df = self.books_df.drop_duplicates(subset=['isbn'])
                self.books_df['title'] = self.books_df['title'].fillna('Unknown')
                self.books_df['author'] = self.books_df['author'].fillna('Unknown')
                self.books_df['genre'] = self.books_df['genre'].fillna('')
                self.books_df['description'] = self.books_df['description'].fillna('')
                self.books_df['rating'] = pd.to_numeric(self.books_df['rating'], errors='coerce').fillna(0)
                self.books_df['num_ratings'] = pd.to_numeric(self.books_df['num_ratings'], errors='coerce').fillna(0)
                self.books_df['pages'] = self.books_df['pages'].fillna('N/A')
                self.books_df['bookformat'] = self.books_df['bookformat'].fillna('Unknown')
                
                # Extract all unique genres
                for genre_str in self.books_df['genre'].dropna():
                    for g in str(genre_str).split(','):
                        self.all_genres.add(g.strip())
                
                print(f"Loaded {len(self.books_df)} books with {len(self.all_genres)} unique genres")
            except Exception as e:
                print(f"Error loading books: {e}")
                self.books_df = pd.DataFrame()
        else:
            print(f"Books file not found: {books_file}")
            self.books_df = pd.DataFrame()
        
        self._calculate_popular_books()
    
    def _calculate_popular_books(self):
        """Calculate popular books based on rating and number of ratings in dataset"""
        if self.books_df is None or self.books_df.empty:
            self.popular_books_df = pd.DataFrame()
            return
        
        # Use rating and num_ratings from the CSV
        if 'rating' in self.books_df.columns and 'num_ratings' in self.books_df.columns:
            df = self.books_df.copy()
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
            df['num_ratings'] = pd.to_numeric(df['num_ratings'], errors='coerce').fillna(0)
            
            # Filter books with at least some ratings and sort by weighted score
            df = df[df['num_ratings'] >= 10]
            df['popularity_score'] = df['rating'] * np.log1p(df['num_ratings'])
            self.popular_books_df = df.sort_values('popularity_score', ascending=False).head(500)
        else:
            self.popular_books_df = self.books_df.head(100)
    
    def _build_models(self):
        self._build_content_based_model()
        self._build_collaborative_model()
    
    def _build_content_based_model(self):
        if self.books_df is None or self.books_df.empty:
            return
        
        print("Building content-based model...")
        
        # Use more books since we have 100k
        sample_size = min(15000, len(self.books_df))
        popular_isbns = set(self.popular_books_df['isbn'].head(5000).tolist()) if self.popular_books_df is not None and len(self.popular_books_df) > 0 else set()
        sample_isbns = set(self.books_df.sample(min(sample_size, len(self.books_df)))['isbn'].tolist())
        
        content_books = self.books_df[self.books_df['isbn'].isin(popular_isbns | sample_isbns)].copy().reset_index(drop=True)
        
        # Build rich features from new dataset
        # Genre is now directly in the CSV, use it with high weight
        def clean_genre(genre_str):
            if pd.isna(genre_str) or not genre_str:
                return ''
            # Take first few genres, repeat for weight
            genres = [g.strip() for g in str(genre_str).split(',')[:5]]
            return ' '.join(genres * 2)  # Repeat for higher weight
        
        content_books['genre_clean'] = content_books['genre'].apply(clean_genre)
        
        # Combine title, genre, description, and author for features
        content_books['features'] = (
            content_books['title'].fillna('') + ' ' + 
            content_books['title'].fillna('') + ' ' +  # Double weight on title
            content_books['genre_clean'] + ' ' +  # Genres from CSV
            content_books['description'].fillna('').str[:500] + ' ' +  # First 500 chars of description
            content_books['author'].fillna('')
        ).str.lower()
        
        # Store author separately for diversity filtering
        self.content_books_df = content_books[['isbn', 'author']].copy()
        
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_books['features'])
        self.isbn_to_idx = {isbn: idx for idx, isbn in enumerate(content_books['isbn'].values)}
        
        # Save models for faster loading next time
        self._save_models()
        
        print(f"Content model built with {len(content_books)} books")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs(self.models_path, exist_ok=True)
            
            # Save books dataframe
            self.books_df.to_pickle(os.path.join(self.models_path, 'books_processed.pkl'))
            
            # Save TF-IDF
            with open(os.path.join(self.models_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            from scipy.sparse import save_npz
            save_npz(os.path.join(self.models_path, 'tfidf_matrix.npz'), self.tfidf_matrix)
            
            with open(os.path.join(self.models_path, 'isbn_to_idx.pkl'), 'wb') as f:
                pickle.dump(self.isbn_to_idx, f)
            
            # Save collaborative model if exists
            if self.item_similarity is not None:
                np.save(os.path.join(self.models_path, 'item_similarity.npy'), self.item_similarity)
                with open(os.path.join(self.models_path, 'book_to_idx_collab.pkl'), 'wb') as f:
                    pickle.dump(self.book_to_idx_collab, f)
                with open(os.path.join(self.models_path, 'idx_to_book_collab.pkl'), 'wb') as f:
                    pickle.dump(self.idx_to_book_collab, f)
            
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def _build_collaborative_model(self):
        """Build collaborative model using ratings from the CSV"""
        if self.books_df is None or self.books_df.empty:
            print("Skipping collaborative model - using content-based only")
            return
        
        # Since ratings are per-book in this dataset (not per-user), 
        # we'll create a simulated collaborative model based on genre similarity
        print("Building genre-based collaborative model...")
        
        # Group books by primary genre for similarity
        def get_primary_genre(genre_str):
            if pd.isna(genre_str) or not genre_str:
                return 'Fiction'
            genres = str(genre_str).split(',')
            return genres[0].strip() if genres else 'Fiction'
        
        sample_books = self.books_df.head(3000).copy()
        sample_books['primary_genre'] = sample_books['genre'].apply(get_primary_genre)
        
        # Create genre-based book groupings
        self.book_to_idx_collab = {isbn: idx for idx, isbn in enumerate(sample_books['isbn'].values)}
        self.idx_to_book_collab = {idx: isbn for isbn, idx in self.book_to_idx_collab.items()}
        
        # Simple genre-based similarity
        genre_dummies = pd.get_dummies(sample_books['primary_genre'])
        from sklearn.preprocessing import normalize
        self.item_similarity = cosine_similarity(normalize(genre_dummies.values, axis=1))
        
        print(f"Collaborative model built with {len(sample_books)} books")
    
    def get_book(self, isbn):
        if self.books_df is None:
            return None
        book = self.books_df[self.books_df['isbn'] == isbn]
        return book.iloc[0].to_dict() if not book.empty else None
    
    def get_popular_books(self, limit=20):
        if self.popular_books_df is None or self.popular_books_df.empty:
            return self.books_df.head(limit).to_dict('records') if self.books_df is not None else []
        return self.popular_books_df.head(limit).to_dict('records')
    
    def search_books(self, query, limit=20):
        if self.books_df is None:
            return []
        q = re.escape(query.lower().strip())  # Escape special regex characters
        mask = (self.books_df['title'].str.lower().str.contains(q, na=False, regex=True) |
                self.books_df['author'].str.lower().str.contains(q, na=False, regex=True) |
                self.books_df['genre'].str.lower().str.contains(q, na=False, regex=True) |
                self.books_df['isbn'].str.lower().str.contains(q, na=False, regex=True))
        return self.books_df[mask].head(limit).to_dict('records')
    
    def get_content_based_recommendations(self, isbn, limit=10):
        if self.tfidf_matrix is None or self.isbn_to_idx is None:
            return []
        if isbn not in self.isbn_to_idx:
            return self._fallback_recommendations(isbn, limit)
        
        idx = self.isbn_to_idx[isbn]
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Get more candidates to allow for diversity filtering
        top_idx = sims.argsort()[::-1][1:limit*5]
        
        idx_to_isbn = {v: k for k, v in self.isbn_to_idx.items()}
        
        # Get the source book's author for diversity
        source_book = self.get_book(isbn)
        source_author = source_book.get('author', '').lower() if source_book else ''
        
        recs = []
        same_author_count = 0
        max_same_author = 2  # Limit books from same author
        
        for i in top_idx:
            if len(recs) >= limit:
                break
                
            rec_isbn = idx_to_isbn.get(i)
            book = self.get_book(rec_isbn)
            if book:
                book_author = book.get('author', '').lower()
                
                # Limit same-author recommendations for diversity
                if book_author == source_author:
                    if same_author_count >= max_same_author:
                        continue
                    same_author_count += 1
                
                book['similarity_score'] = float(sims[i])
                book['method'] = 'content-based'
                recs.append(book)
        
        return recs
    
    def get_collaborative_recommendations(self, isbn, limit=10):
        if self.item_similarity is None or self.book_to_idx_collab is None:
            return []
        if isbn not in self.book_to_idx_collab:
            return []
        
        idx = self.book_to_idx_collab[isbn]
        sims = self.item_similarity[idx]
        top_idx = sims.argsort()[::-1][1:limit+1]
        
        recs = []
        for i in top_idx:
            book = self.get_book(self.idx_to_book_collab.get(i))
            if book:
                book['similarity_score'] = float(sims[i])
                book['method'] = 'collaborative'
                recs.append(book)
        return recs
    
    def get_hybrid_recommendations(self, isbn, limit=10, content_weight=0.4):
        content = self.get_content_based_recommendations(isbn, limit * 2)
        collab = self.get_collaborative_recommendations(isbn, limit * 2)
        
        # Also get real-time collaborative recommendations from user activity
        realtime_collab = self.get_realtime_collaborative_recommendations(isbn, limit * 2)
        
        scores = {}
        for book in content:
            scores[book['isbn']] = {'book': book, 'content': book.get('similarity_score', 0), 'collab': 0, 'realtime': 0}
        for book in collab:
            if book['isbn'] in scores:
                scores[book['isbn']]['collab'] = book.get('similarity_score', 0)
            else:
                scores[book['isbn']] = {'book': book, 'content': 0, 'collab': book.get('similarity_score', 0), 'realtime': 0}
        
        # Add real-time collaborative scores (from user favorites/reviews)
        for book in realtime_collab:
            if book['isbn'] in scores:
                scores[book['isbn']]['realtime'] = book.get('similarity_score', 0)
            else:
                scores[book['isbn']] = {'book': book, 'content': 0, 'collab': 0, 'realtime': book.get('similarity_score', 0)}
        
        # Weighted combination: content 40%, static collab 30%, realtime collab 30%
        for data in scores.values():
            data['hybrid'] = (content_weight * data['content'] + 
                            0.3 * data['collab'] + 
                            0.3 * data['realtime'])
            data['book']['hybrid_score'] = data['hybrid']
            data['book']['method'] = 'hybrid'
        
        sorted_recs = sorted(scores.values(), key=lambda x: x['hybrid'], reverse=True)
        return [item['book'] for item in sorted_recs[:limit]]
    
    def get_personalized_recommendations(self, favorites=None, to_read=None, history=None, searches=None, limit=20):
        seeds = []
        if favorites:
            seeds.extend([(isbn, 1.0) for isbn in favorites[:5]])
        if to_read:
            seeds.extend([(isbn, 0.9) for isbn in to_read[:5]])
        if history:
            seeds.extend([(isbn, 0.7) for isbn in history[:5]])
        if searches:
            for q in searches[:3]:
                for book in self.search_books(q, 2):
                    seeds.append((book['isbn'], 0.5))
        
        if not seeds:
            return self.get_popular_books(limit)
        
        recs = {}
        for isbn, weight in seeds:
            for book in self.get_hybrid_recommendations(isbn, 5):
                if book['isbn'] not in [s[0] for s in seeds]:
                    if book['isbn'] not in recs:
                        recs[book['isbn']] = {'book': book, 'score': 0}
                    recs[book['isbn']]['score'] += weight * book.get('hybrid_score', 0.5)
        
        results = [item['book'] for item in sorted(recs.values(), key=lambda x: x['score'], reverse=True)][:limit]
        
        if len(results) < limit:
            existing = {r['isbn'] for r in results}
            for book in self.get_popular_books(limit - len(results)):
                if book['isbn'] not in existing:
                    results.append(book)
        
        return results
    
    def get_unified_recommendations(self, user_id=None, favorites=None, to_read=None, 
                                     history=None, searches=None, limit=20,
                                     previous_recommendations=None, rotation_ratio=0.4):
        """Get unified recommendations based on ALL user activity including searches.
        Returns a single list of recommended books without separate categories.
        
        Args:
            previous_recommendations: List of ISBNs previously shown (for rotation)
            rotation_ratio: Fraction of old recommendations to remove (0.0-1.0)
        """
        
        # Collect all seeds with weights
        seeds = []
        excluded_isbns = set()
        
        # Exclude some old recommendations to show fresh ones
        kept_old = set()
        if previous_recommendations:
            # Keep only some of the previous recommendations
            num_to_keep = int(len(previous_recommendations) * (1 - rotation_ratio))
            # Keep the most recently added ones (assuming order matters)
            kept_old = set(previous_recommendations[:num_to_keep])
            # Exclude the rest to force new recommendations
            rotated_out = set(previous_recommendations[num_to_keep:])
            excluded_isbns.update(rotated_out)
        
        # Favorites have highest weight
        if favorites:
            excluded_isbns.update(favorites)
            seeds.extend([(isbn, 1.0) for isbn in favorites[:5]])
        
        # To-read list
        if to_read:
            excluded_isbns.update(to_read)
            seeds.extend([(isbn, 0.9) for isbn in to_read[:5]])
        
        # Recent views/history
        if history:
            excluded_isbns.update(history)
            seeds.extend([(isbn, 0.7) for isbn in history[:5]])
        
        # Search queries - find books matching searches and use as seeds
        if searches:
            for query in searches[:5]:
                search_results = self.search_books(query, limit=3)
                for book in search_results:
                    if book['isbn'] not in excluded_isbns:
                        seeds.append((book['isbn'], 0.8))  # High weight for search interest
        
        if not seeds:
            return self.get_popular_books(limit)
        
        # Find similar books for each seed
        recommendation_scores = {}
        
        for seed_isbn, weight in seeds:
            # Get content-based similar books
            content_similar = self.get_content_based_recommendations(seed_isbn, limit=4)
            for book in content_similar:
                if book['isbn'] not in excluded_isbns:
                    if book['isbn'] not in recommendation_scores:
                        recommendation_scores[book['isbn']] = {'book': book, 'score': 0}
                    recommendation_scores[book['isbn']]['score'] += weight * book.get('similarity_score', 0.5) * 1.2
            
            # Get collaborative similar books
            collab_similar = self.get_collaborative_recommendations(seed_isbn, limit=3)
            for book in collab_similar:
                if book['isbn'] not in excluded_isbns:
                    if book['isbn'] not in recommendation_scores:
                        recommendation_scores[book['isbn']] = {'book': book, 'score': 0}
                    recommendation_scores[book['isbn']]['score'] += weight * book.get('similarity_score', 0.5)
        
        # Sort by score and get top recommendations
        sorted_recs = sorted(recommendation_scores.values(), key=lambda x: x['score'], reverse=True)
        
        results = []
        for item in sorted_recs[:limit]:
            book = item['book']
            book['recommendation_score'] = item['score']
            results.append(book)
        
        # Fill with popular books if needed
        if len(results) < limit:
            existing = {r['isbn'] for r in results} | excluded_isbns
            for book in self.get_popular_books(limit - len(results) + 5):
                if len(results) >= limit:
                    break
                if book['isbn'] not in existing:
                    results.append(book)
        
        return results[:limit]
    
    def _fallback_recommendations(self, isbn, limit):
        book = self.get_book(isbn)
        if not book:
            return self.get_popular_books(limit)
        
        author = book.get('author', '')
        title_words = book.get('title', '').lower().split()
        
        results = []
        seen_isbns = {isbn}
        
        # Get a few books from same author (max 2)
        same_author = self.books_df[
            (self.books_df['author'].str.lower() == author.lower()) & 
            (self.books_df['isbn'] != isbn)
        ].head(2)
        
        for _, row in same_author.iterrows():
            r = row.to_dict()
            r['method'] = 'same-author'
            r['similarity_score'] = 0.6
            results.append(r)
            seen_isbns.add(r['isbn'])
        
        # Get books with similar title words (different authors)
        if title_words:
            import re
            for word in title_words[:3]:
                # Skip short words and ensure word is alphanumeric only
                if len(word) > 3 and word.isalnum():
                    # Escape regex special characters
                    escaped_word = re.escape(word)
                    try:
                        similar_title = self.books_df[
                            (self.books_df['title'].str.lower().str.contains(escaped_word, na=False, regex=True)) &
                            (~self.books_df['isbn'].isin(seen_isbns)) &
                            (self.books_df['author'].str.lower() != author.lower())
                        ].head(3)
                        
                        for _, row in similar_title.iterrows():
                            if len(results) >= limit:
                                break
                            r = row.to_dict()
                            r['method'] = 'similar-title'
                            r['similarity_score'] = 0.4
                            results.append(r)
                            seen_isbns.add(r['isbn'])
                    except Exception:
                        continue
        
        # Get books with similar genres (based on title inference)
        if len(results) < limit:
            source_genres = set(self.get_genre_for_title(book.get('title', '')))
            genre_matches = []
            
            for _, row in self.books_df.sample(min(500, len(self.books_df))).iterrows():
                if row['isbn'] in seen_isbns:
                    continue
                if row['author'].lower() == author.lower():
                    continue
                    
                book_genres = set(self.get_genre_for_title(row['title']))
                overlap = len(source_genres & book_genres)
                if overlap > 0:
                    r = row.to_dict()
                    r['method'] = 'similar-genre'
                    r['similarity_score'] = 0.5 * (overlap / max(len(source_genres), 1))
                    r['matched_genres'] = list(source_genres & book_genres)
                    genre_matches.append(r)
            
            # Sort by similarity and add to results
            genre_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            for r in genre_matches[:limit - len(results)]:
                results.append(r)
                seen_isbns.add(r['isbn'])
        
        # Fill remaining with popular books
        if len(results) < limit:
            popular = self.get_popular_books(limit - len(results) + 5)
            for p in popular:
                if len(results) >= limit:
                    break
                if p['isbn'] not in seen_isbns:
                    p['method'] = 'popular'
                    p['similarity_score'] = 0.3
                    results.append(p)
        
        return results[:limit]
    
    def get_books_by_genre(self, genre, limit=20, exclude_isbns=None):
        """Get books that match a specific genre based on title inference"""
        if self.books_df is None:
            return []
        
        exclude_isbns = exclude_isbns or set()
        genre_lower = genre.lower()
        matches = []
        
        for _, row in self.books_df.sample(min(1000, len(self.books_df))).iterrows():
            if row['isbn'] in exclude_isbns:
                continue
            
            book_genres = self.get_genre_for_title(row['title'])
            for bg in book_genres:
                if genre_lower in bg.lower():
                    r = row.to_dict()
                    r['genre'] = bg
                    r['method'] = 'genre-match'
                    matches.append(r)
                    break
        
        return matches[:limit]
    
    def get_all_books(self, page=1, per_page=20):
        if self.books_df is None:
            return [], 0
        start = (page - 1) * per_page
        return self.books_df.iloc[start:start + per_page].to_dict('records'), len(self.books_df)
    
    def set_mongo(self, mongo):
        """Set MongoDB instance for accessing user reviews"""
        self.mongo = mongo
    
    def get_all_user_interactions(self):
        """Get all user interactions (reviews, favorites) to build collaborative filtering"""
        if self.mongo is None:
            return pd.DataFrame()
        
        try:
            interactions = []
            
            # Get all reviews
            reviews = list(self.mongo.db.reviews.find({}))
            for r in reviews:
                interactions.append({
                    'user_id': str(r.get('user_id', '')),
                    'isbn': r.get('isbn', ''),
                    'rating': r.get('rating', 0),
                    'type': 'review'
                })
            
            # Get all user favorites (implicit rating of 5)
            users = list(self.mongo.db.users.find({'favorites': {'$exists': True, '$ne': []}}))
            for user in users:
                for isbn in user.get('favorites', []):
                    interactions.append({
                        'user_id': str(user.get('_id', '')),
                        'isbn': isbn,
                        'rating': 5,  # Favorite = highest implicit rating
                        'type': 'favorite'
                    })
            
            # Get user to_read lists (implicit rating of 4)
            users_to_read = list(self.mongo.db.users.find({'to_read': {'$exists': True, '$ne': []}}))
            for user in users_to_read:
                for isbn in user.get('to_read', []):
                    interactions.append({
                        'user_id': str(user.get('_id', '')),
                        'isbn': isbn,
                        'rating': 4,  # To-read = high implicit rating
                        'type': 'to_read'
                    })
            
            if not interactions:
                return pd.DataFrame()
            
            df = pd.DataFrame(interactions)
            # If same user has multiple interactions with same book, take the max rating
            df = df.groupby(['user_id', 'isbn']).agg({'rating': 'max'}).reset_index()
            return df
        except Exception as e:
            print(f"Error getting user interactions: {e}")
            return pd.DataFrame()
    
    def get_realtime_collaborative_recommendations(self, isbn, limit=10):
        """Get collaborative recommendations using real-time user data from MongoDB"""
        if self.mongo is None:
            return []
        
        try:
            interactions_df = self.get_all_user_interactions()
            if interactions_df.empty or len(interactions_df) < 5:
                return []
            
            # Check if the book has any interactions
            if isbn not in interactions_df['isbn'].values:
                return []
            
            # Build a simple item-item similarity based on users who interacted with both
            book_users = set(interactions_df[interactions_df['isbn'] == isbn]['user_id'].values)
            
            if not book_users:
                return []
            
            # Find other books those users interacted with
            other_books = interactions_df[
                (interactions_df['user_id'].isin(book_users)) & 
                (interactions_df['isbn'] != isbn)
            ]
            
            if other_books.empty:
                return []
            
            # Score books by: how many users who liked target book also liked this book * avg rating
            book_scores = other_books.groupby('isbn').agg({
                'user_id': 'count',  # Number of shared users
                'rating': 'mean'     # Average rating
            }).reset_index()
            book_scores.columns = ['isbn', 'shared_users', 'avg_rating']
            book_scores['score'] = book_scores['shared_users'] * book_scores['avg_rating']
            book_scores = book_scores.sort_values('score', ascending=False).head(limit)
            
            books = []
            for _, row in book_scores.iterrows():
                book = self.get_book(row['isbn'])
                if book:
                    book['method'] = 'realtime-collaborative'
                    book['similarity_score'] = row['score'] / book_scores['score'].max() if book_scores['score'].max() > 0 else 0
                    books.append(book)
            
            return books
        except Exception as e:
            print(f"Error in realtime collaborative: {e}")
            return []
    
    def get_trending_from_user_activity(self, limit=20):
        """Get recommendations based on what ALL users have been interacting with.
        This finds SIMILAR books to what users have favorited/reviewed, not just the same books."""
        if self.mongo is None:
            return []
        
        try:
            # Collect all ISBNs users have interacted with and their weights
            seed_books = {}
            
            # Get from reviews (weighted by rating)
            reviews = list(self.mongo.db.reviews.find({}))
            for r in reviews:
                isbn = r.get('isbn', '')
                rating = r.get('rating', 0)
                if isbn and rating >= 3:  # Only use positive reviews as seeds
                    seed_books[isbn] = seed_books.get(isbn, 0) + rating
            
            # Get from favorites (high weight)
            users = list(self.mongo.db.users.find({'favorites': {'$exists': True, '$ne': []}}))
            for user in users:
                for isbn in user.get('favorites', []):
                    seed_books[isbn] = seed_books.get(isbn, 0) + 5
            
            # Get from to-read
            users_to_read = list(self.mongo.db.users.find({'to_read': {'$exists': True, '$ne': []}}))
            for user in users_to_read:
                for isbn in user.get('to_read', []):
                    seed_books[isbn] = seed_books.get(isbn, 0) + 3
            
            if not seed_books:
                return []
            
            # Now find SIMILAR books to these seed books
            # Sort seeds by weight and use top ones
            sorted_seeds = sorted(seed_books.items(), key=lambda x: x[1], reverse=True)[:10]
            
            recommendation_scores = {}
            seed_isbns = set(seed_books.keys())
            
            for seed_isbn, weight in sorted_seeds:
                # Get content-based similar books for each seed
                similar = self.get_content_based_recommendations(seed_isbn, limit=5)
                for book in similar:
                    rec_isbn = book['isbn']
                    # Don't recommend books that are already in the seed set
                    if rec_isbn not in seed_isbns:
                        if rec_isbn not in recommendation_scores:
                            recommendation_scores[rec_isbn] = {'book': book, 'score': 0}
                        # Add weighted score based on seed weight and similarity
                        recommendation_scores[rec_isbn]['score'] += weight * book.get('similarity_score', 0.5)
            
            # Sort by combined score
            sorted_recs = sorted(recommendation_scores.values(), key=lambda x: x['score'], reverse=True)
            
            results = []
            for item in sorted_recs[:limit]:
                book = item['book']
                book['method'] = 'community-recommended'
                book['community_score'] = item['score']
                results.append(book)
            
            return results
        except Exception as e:
            print(f"Error getting trending books: {e}")
            return []
    
    def get_recommendations_for_community(self, limit=20):
        """Get books that are similar to what the community has been engaging with.
        Uses collaborative signals to find books users with similar taste enjoyed."""
        if self.mongo is None:
            return []
        
        try:
            # Get all user interactions
            interactions_df = self.get_all_user_interactions()
            if interactions_df.empty:
                return []
            
            # Find the most interacted-with books as seeds
            book_popularity = interactions_df.groupby('isbn').agg({
                'rating': ['sum', 'count']
            }).reset_index()
            book_popularity.columns = ['isbn', 'total_score', 'interaction_count']
            book_popularity = book_popularity.sort_values('total_score', ascending=False)
            
            seed_isbns = book_popularity.head(10)['isbn'].tolist()
            seed_set = set(seed_isbns)
            
            # For each seed, find similar books using hybrid recommendations
            recommendation_scores = {}
            
            for seed_isbn in seed_isbns:
                seed_score = book_popularity[book_popularity['isbn'] == seed_isbn]['total_score'].values[0]
                
                # Get hybrid recommendations (content + collaborative)
                similar = self.get_hybrid_recommendations(seed_isbn, limit=5)
                
                for book in similar:
                    rec_isbn = book['isbn']
                    if rec_isbn not in seed_set:
                        if rec_isbn not in recommendation_scores:
                            recommendation_scores[rec_isbn] = {'book': book, 'score': 0}
                        recommendation_scores[rec_isbn]['score'] += seed_score * book.get('hybrid_score', 0.5)
            
            # Sort and return
            sorted_recs = sorted(recommendation_scores.values(), key=lambda x: x['score'], reverse=True)
            
            results = []
            for item in sorted_recs[:limit]:
                book = item['book']
                book['method'] = 'community-collaborative'
                results.append(book)
            
            return results
        except Exception as e:
            print(f"Error getting community recommendations: {e}")
            return []

    def get_user_review_ratings(self):
        """Get aggregated ratings from user reviews to boost collaborative filtering"""
        if self.mongo is None:
            return {}
        
        try:
            # Aggregate reviews by ISBN
            pipeline = [
                {'$group': {
                    '_id': '$isbn',
                    'avg_rating': {'$avg': '$rating'},
                    'count': {'$sum': 1}
                }},
                {'$match': {'count': {'$gte': 1}}}  # At least 1 review
            ]
            results = list(self.mongo.db.reviews.aggregate(pipeline))
            return {r['_id']: {'avg_rating': r['avg_rating'], 'count': r['count']} for r in results}
        except Exception as e:
            print(f"Error fetching review ratings: {e}")
            return {}
    
    def get_highly_reviewed_books(self, min_rating=4, limit=20):
        """Get books with high user review ratings"""
        if self.mongo is None:
            return []
        
        try:
            pipeline = [
                {'$group': {
                    '_id': '$isbn',
                    'avg_rating': {'$avg': '$rating'},
                    'count': {'$sum': 1}
                }},
                {'$match': {'avg_rating': {'$gte': min_rating}, 'count': {'$gte': 1}}},
                {'$sort': {'avg_rating': -1, 'count': -1}},
                {'$limit': limit}
            ]
            results = list(self.mongo.db.reviews.aggregate(pipeline))
            
            books = []
            for r in results:
                book = self.get_book(r['_id'])
                if book:
                    book['user_review_rating'] = r['avg_rating']
                    book['user_review_count'] = r['count']
                    books.append(book)
            return books
        except Exception:
            return []
    
    def get_recommendations_from_similar_users(self, user_id, limit=20):
        """Find books liked by users with similar taste based on their reviews"""
        if self.mongo is None:
            return []
        
        try:
            # Get current user's highly rated books
            user_reviews = list(self.mongo.db.reviews.find({'user_id': user_id, 'rating': {'$gte': 4}}))
            if not user_reviews:
                return []
            
            user_liked_isbns = [r['isbn'] for r in user_reviews]
            
            # Find users who also liked those books
            similar_user_reviews = list(self.mongo.db.reviews.find({
                'isbn': {'$in': user_liked_isbns},
                'rating': {'$gte': 4},
                'user_id': {'$ne': user_id}
            }))
            
            similar_users = list(set(r['user_id'] for r in similar_user_reviews))
            
            if not similar_users:
                return []
            
            # Get books those similar users liked that current user hasn't reviewed
            recs = list(self.mongo.db.reviews.find({
                'user_id': {'$in': similar_users},
                'rating': {'$gte': 4},
                'isbn': {'$nin': user_liked_isbns}
            }))
            
            # Count how many similar users liked each book
            isbn_scores = {}
            for r in recs:
                isbn = r['isbn']
                if isbn not in isbn_scores:
                    isbn_scores[isbn] = 0
                isbn_scores[isbn] += r['rating']
            
            # Sort by score and return
            sorted_isbns = sorted(isbn_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            books = []
            for isbn, score in sorted_isbns:
                book = self.get_book(isbn)
                if book:
                    book['method'] = 'user-collaborative'
                    book['similarity_score'] = score / 5.0
                    books.append(book)
            
            return books
        except Exception as e:
            print(f"Error getting user-based recommendations: {e}")
            return []
    
    def get_enhanced_personalized_recommendations(self, user_id=None, favorites=None, to_read=None, 
                                                   history=None, searches=None, limit=20):
        """Enhanced personalized recommendations that prioritize user's own activity.
        Returns SIMILAR books based on user's favorites, to_read, and history."""
        
        # Collect user's seed books
        user_seeds = []
        if favorites:
            user_seeds.extend([(isbn, 1.0) for isbn in favorites])
        if to_read:
            user_seeds.extend([(isbn, 0.8) for isbn in to_read])
        if history:
            user_seeds.extend([(isbn, 0.6) for isbn in history])
        
        # Exclude the user's own books from recommendations
        excluded_isbns = set(favorites or []) | set(to_read or []) | set(history or [])
        
        if not user_seeds:
            # If user has no activity, return popular books
            return self.get_popular_books(limit)
        
        # Find SIMILAR books to user's seeds using content-based and collaborative filtering
        recommendation_scores = {}
        
        for seed_isbn, weight in user_seeds:
            # Get content-based similar books
            content_similar = self.get_content_based_recommendations(seed_isbn, limit=5)
            for book in content_similar:
                rec_isbn = book['isbn']
                if rec_isbn not in excluded_isbns:
                    if rec_isbn not in recommendation_scores:
                        recommendation_scores[rec_isbn] = {'book': book, 'score': 0, 'sources': []}
                    recommendation_scores[rec_isbn]['score'] += weight * book.get('similarity_score', 0.5) * 1.5
                    recommendation_scores[rec_isbn]['sources'].append('content')
            
            # Get collaborative filtering similar books
            collab_similar = self.get_collaborative_recommendations(seed_isbn, limit=5)
            for book in collab_similar:
                rec_isbn = book['isbn']
                if rec_isbn not in excluded_isbns:
                    if rec_isbn not in recommendation_scores:
                        recommendation_scores[rec_isbn] = {'book': book, 'score': 0, 'sources': []}
                    recommendation_scores[rec_isbn]['score'] += weight * book.get('similarity_score', 0.5)
                    recommendation_scores[rec_isbn]['sources'].append('collaborative')
        
        # Add recommendations from similar users (if logged in)
        if user_id is not None and self.mongo is not None:
            user_recs = self.get_recommendations_from_similar_users(user_id, limit=10)
            for book in user_recs:
                rec_isbn = book['isbn']
                if rec_isbn not in excluded_isbns:
                    if rec_isbn not in recommendation_scores:
                        recommendation_scores[rec_isbn] = {'book': book, 'score': 0, 'sources': []}
                    recommendation_scores[rec_isbn]['score'] += book.get('similarity_score', 0.5) * 1.2
                    recommendation_scores[rec_isbn]['sources'].append('similar-users')
        
        # Sort by score
        sorted_recs = sorted(recommendation_scores.values(), key=lambda x: x['score'], reverse=True)
        
        final_recs = []
        for item in sorted_recs[:limit]:
            book = item['book']
            book['personalized_score'] = item['score']
            book['rec_sources'] = list(set(item['sources']))
            final_recs.append(book)
        
        # If not enough recommendations, fill with popular books
        if len(final_recs) < limit:
            existing = {r['isbn'] for r in final_recs}
            for book in self.get_popular_books(limit - len(final_recs)):
                if book['isbn'] not in existing and book['isbn'] not in excluded_isbns:
                    final_recs.append(book)
        
        return final_recs[:limit]
