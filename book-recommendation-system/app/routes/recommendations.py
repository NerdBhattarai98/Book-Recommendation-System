from flask import Blueprint, render_template, request, session
from flask_login import current_user
from app.services.recommendation_engine import RecommendationEngine
from app import mongo

recommendations_bp = Blueprint('recommendations', __name__, url_prefix='/recommendations')

engine = None

def get_engine():
    global engine
    if engine is None:
        from app.config import Config
        engine = RecommendationEngine(Config.DATA_DIR, mongo)
    elif engine.mongo is None:
        engine.set_mongo(mongo)
    return engine


@recommendations_bp.route('/')
def recommendations():
    """Get unified personalized recommendations"""
    rec_engine = get_engine()
    
    # Get user data from database if logged in, else from session
    user_id = None
    if current_user.is_authenticated:
        user_id = current_user._id
        favorites = current_user.get_favorite_isbns()
        to_read = current_user.get_to_read_isbns()
        recent_views = current_user.get_history_isbns()
        searches = current_user.get_recent_searches()
        # include high-rated user reviews as additional strong signals
        try:
            user_reviews = list(mongo.db.reviews.find({'user_id': current_user._id, 'rating': {'$gte': 4}}))
            review_isbns = [r['isbn'] for r in user_reviews]
            for isbn in review_isbns:
                if isbn not in favorites:
                    favorites.append(isbn)
        except Exception:
            pass
    else:
        favorites = session.get('favorites', [])
        to_read = session.get('to_read', [])
        recent_views = session.get('recent_views', [])
        searches = session.get('search_history', [])
    
    # Get unified recommendations based on ALL activity
    recommended_books = []
    if favorites or to_read or recent_views or searches:
        recommended_books = rec_engine.get_unified_recommendations(
            user_id=user_id,
            favorites=favorites,
            to_read=to_read,
            history=recent_views,
            searches=searches,
            limit=20
        )
    
    # Get popular books as fallback
    popular_books = rec_engine.get_popular_books(limit=12)
    
    return render_template('recommendations/recommendations.html', 
                           recommended_books=recommended_books,
                           popular_books=popular_books,
                           has_activity=bool(favorites or to_read or recent_views or searches))

@recommendations_bp.route('/for/<isbn>')
def recommendations_for_book(isbn):
    """Get recommendations based on a specific book"""
    rec_engine = get_engine()
    book = rec_engine.get_book(isbn)
    
    if not book:
        return render_template('recommendations/recommendations.html', 
                               recommendations={}, book=None)
    
    content_based = rec_engine.get_content_based_recommendations(isbn, limit=6)
    collaborative = rec_engine.get_collaborative_recommendations(isbn, limit=6)
    
    recommendations = {
        'content_based': content_based,
        'collaborative': collaborative
    }
    
    return render_template('recommendations/recommendations.html',
                           recommendations=recommendations,
                           book=book)
