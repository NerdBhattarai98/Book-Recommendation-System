from app.services.collaborative_filtering import CollaborativeFiltering
from app.models import User, Book, Rating
from app.utils.data_loader import load_data
import pytest

@pytest.fixture
def setup_data():
    users, books, ratings = load_data()
    return users, books, ratings

def test_collaborative_filtering(setup_data):
    users, books, ratings = setup_data
    cf = CollaborativeFiltering(users, books, ratings)

    # Test for a specific user recommendation
    user_id = 1  # Example user ID
    recommendations = cf.get_recommendations(user_id)

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0  # Ensure recommendations are returned

    # Check if recommended books are in the books dataset
    for book_id in recommendations:
        assert book_id in books['book_id'].values

def test_collaborative_filtering_no_ratings(setup_data):
    users, books, ratings = setup_data
    cf = CollaborativeFiltering(users, books, ratings)

    # Test for a user with no ratings
    user_id = 999  # Example user ID with no ratings
    recommendations = cf.get_recommendations(user_id)

    assert recommendations == []  # Ensure no recommendations are returned for users with no ratings

def test_collaborative_filtering_edge_cases(setup_data):
    users, books, ratings = setup_data
    cf = CollaborativeFiltering(users, books, ratings)

    # Test for invalid user ID
    user_id = -1  # Invalid user ID
    recommendations = cf.get_recommendations(user_id)

    assert recommendations == []  # Ensure no recommendations are returned for invalid user ID

    # Test for empty dataset
    empty_users = {}
    empty_books = {}
    empty_ratings = {}
    cf_empty = CollaborativeFiltering(empty_users, empty_books, empty_ratings)
    recommendations = cf_empty.get_recommendations(1)

    assert recommendations == []  # Ensure no recommendations are returned for empty datasets