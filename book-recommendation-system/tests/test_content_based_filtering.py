from app.services.content_based_filtering import ContentBasedFiltering
import pandas as pd
import pytest

@pytest.fixture
def setup_data():
    books_data = {
        'book_id': [1, 2, 3],
        'title': ['Book One', 'Book Two', 'Book Three'],
        'author': ['Author A', 'Author B', 'Author C'],
        'genre': ['Fiction', 'Fiction', 'Non-Fiction'],
        'description': ['A great book about life.', 'An interesting tale.', 'A deep dive into history.']
    }
    ratings_data = {
        'user_id': [1, 1, 2],
        'book_id': [1, 2, 1],
        'rating': [5, 4, 3]
    }
    
    books_df = pd.DataFrame(books_data)
    ratings_df = pd.DataFrame(ratings_data)
    
    return books_df, ratings_df

def test_content_based_filtering(setup_data):
    books_df, _ = setup_data
    cbf = ContentBasedFiltering(books_df)
    
    # Test for recommendations based on a specific book
    recommendations = cbf.recommend(book_id=1)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(rec, dict) for rec in recommendations)

def test_no_recommendations_for_unknown_book(setup_data):
    books_df, _ = setup_data
    cbf = ContentBasedFiltering(books_df)
    
    # Test for recommendations for a book that does not exist
    recommendations = cbf.recommend(book_id=999)
    
    assert recommendations == []  # Expecting no recommendations for unknown book

def test_recommendations_based_on_genre(setup_data):
    books_df, _ = setup_data
    cbf = ContentBasedFiltering(books_df)
    
    # Test for recommendations based on genre
    recommendations = cbf.recommend(book_id=2)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(rec['genre'] == 'Fiction' for rec in recommendations)  # Assuming book_id=2 is Fiction