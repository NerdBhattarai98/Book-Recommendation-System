from flask import Blueprint, request, jsonify, session
from flask_login import current_user
from app import mongo
from app.services.recommendation_engine import RecommendationEngine
import os

user_bp = Blueprint('user', __name__, url_prefix='/user')

# Get the recommendation engine to validate ISBNs against current dataset
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        _engine = RecommendationEngine(data_path, mongo)
    return _engine


def _filter_valid_isbns(isbns):
    """Filter ISBNs to only include those that exist in the current dataset."""
    engine = _get_engine()
    valid_isbns = []
    for isbn in isbns:
        book = engine.get_book(isbn)
        if book is not None:
            valid_isbns.append(isbn)
    return valid_isbns


def _trigger_recommendation_refresh():
    """Clear some old recommendations to force fresh ones on next page load.
    Keeps only the first half of previous recommendations."""
    if 'previous_recommendations' in session:
        prev_recs = session['previous_recommendations']
        # Keep only half, forcing rotation of the rest
        session['previous_recommendations'] = prev_recs[:len(prev_recs) // 2]
        session.modified = True


@user_bp.route('/favorites', methods=['GET'])
def get_favorites():
    if current_user.is_authenticated:
        favorites = current_user.get_favorite_isbns()
    else:
        favorites = session.get('favorites', [])
    # Filter to only include ISBNs that exist in current dataset
    valid_favorites = _filter_valid_isbns(favorites)
    return jsonify({'favorites': valid_favorites})


@user_bp.route('/favorites/add', methods=['POST'])
def add_favorite():
    data = request.get_json()
    isbn = data.get('isbn')
    
    if not isbn:
        return jsonify({'error': 'ISBN required'}), 400
    
    if current_user.is_authenticated:
        current_user.add_favorite(mongo, isbn)
        favorites = current_user.get_favorite_isbns()
    else:
        if 'favorites' not in session:
            session['favorites'] = []
        if isbn not in session['favorites']:
            session['favorites'] = session['favorites'] + [isbn]
            session.modified = True
        favorites = session['favorites']
    
    # Trigger recommendation refresh by clearing some old recommendations
    _trigger_recommendation_refresh()
    
    # Filter to only include valid ISBNs
    valid_favorites = _filter_valid_isbns(favorites)
    return jsonify({'success': True, 'favorites': valid_favorites})


@user_bp.route('/favorites/remove', methods=['POST'])
def remove_favorite():
    data = request.get_json()
    isbn = data.get('isbn')
    
    if current_user.is_authenticated:
        current_user.remove_favorite(mongo, isbn)
        favorites = current_user.get_favorite_isbns()
    else:
        if 'favorites' in session and isbn in session['favorites']:
            session['favorites'] = [f for f in session['favorites'] if f != isbn]
            session.modified = True
        favorites = session.get('favorites', [])
    
    # Trigger recommendation refresh
    _trigger_recommendation_refresh()
    
    # Filter to only include valid ISBNs
    valid_favorites = _filter_valid_isbns(favorites)
    return jsonify({'success': True, 'favorites': valid_favorites})


@user_bp.route('/history', methods=['GET'])
def get_history():
    if current_user.is_authenticated:
        return jsonify({
            'recent_views': current_user.get_history_isbns(),
            'search_history': current_user.get_recent_searches()
        })
    return jsonify({
        'recent_views': session.get('recent_views', []),
        'search_history': session.get('search_history', [])
    })


@user_bp.route('/history/add', methods=['POST'])
def add_to_history():
    data = request.get_json()
    isbn = data.get('isbn')
    
    if not isbn:
        return jsonify({'error': 'ISBN required'}), 400
    
    if current_user.is_authenticated:
        current_user.add_to_history(mongo, isbn)
    else:
        if 'recent_views' not in session:
            session['recent_views'] = []
        if isbn not in session['recent_views']:
            session['recent_views'] = [isbn] + session['recent_views'][:19]
            session.modified = True
    
    return jsonify({'success': True})


@user_bp.route('/search/add', methods=['POST'])
def add_search():
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    if current_user.is_authenticated:
        current_user.add_search(mongo, query)
    else:
        if 'search_history' not in session:
            session['search_history'] = []
        if query not in session['search_history']:
            session['search_history'] = [query] + session['search_history'][:9]
            session.modified = True
    
    # Trigger recommendation refresh for new search
    _trigger_recommendation_refresh()
    
    return jsonify({'success': True})


@user_bp.route('/to_read', methods=['GET'])
def get_to_read():
    if current_user.is_authenticated:
        to_read = current_user.get_to_read_isbns()
    else:
        to_read = session.get('to_read', [])
    # Filter to only include ISBNs that exist in current dataset
    valid_to_read = _filter_valid_isbns(to_read)
    return jsonify({'to_read': valid_to_read})


@user_bp.route('/to_read/add', methods=['POST'])
def add_to_read():
    data = request.get_json()
    isbn = data.get('isbn')
    if not isbn:
        return jsonify({'error': 'ISBN required'}), 400

    if current_user.is_authenticated:
        current_user.add_to_read(mongo, isbn)
        to_read = current_user.get_to_read_isbns()
    else:
        if 'to_read' not in session:
            session['to_read'] = []
        if isbn not in session['to_read']:
            session['to_read'] = session['to_read'] + [isbn]
            session.modified = True
        to_read = session['to_read']

    # Trigger recommendation refresh
    _trigger_recommendation_refresh()
    
    return jsonify({'success': True, 'to_read': to_read})


@user_bp.route('/to_read/remove', methods=['POST'])
def remove_from_to_read():
    data = request.get_json()
    isbn = data.get('isbn')
    if not isbn:
        return jsonify({'error': 'ISBN required'}), 400

    if current_user.is_authenticated:
        current_user.remove_from_to_read(mongo, isbn)
        to_read = current_user.get_to_read_isbns()
    else:
        if 'to_read' in session and isbn in session['to_read']:
            session['to_read'] = [f for f in session['to_read'] if f != isbn]
            session.modified = True
        to_read = session.get('to_read', [])

    return jsonify({'success': True, 'to_read': to_read})


@user_bp.route('/reviews/<isbn>', methods=['GET'])
def get_reviews(isbn):
    reviews = list(mongo.db.reviews.find({'isbn': isbn}).sort('created_at', -1).limit(50))
    for r in reviews:
        r['_id'] = str(r.get('_id'))
        r['user_id'] = str(r.get('user_id')) if r.get('user_id') else None
        if 'created_at' in r:
            r['created_at'] = r['created_at'].isoformat()
    return jsonify({'reviews': reviews})


@user_bp.route('/reviews/add', methods=['POST'])
def add_review():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Authentication required'}), 401

    data = request.get_json()
    isbn = data.get('isbn')
    rating = data.get('rating')
    text = data.get('text', '').strip()

    if not isbn or rating is None:
        return jsonify({'error': 'ISBN and rating required'}), 400

    review_doc = {
        'user_id': current_user._id,
        'username': current_user.username,
        'isbn': isbn,
        'rating': int(rating),
        'text': text,
        'created_at': __import__('datetime').datetime.utcnow()
    }

    mongo.db.reviews.insert_one(review_doc)
    return jsonify({'success': True})


@user_bp.route('/history/clear', methods=['POST'])
def clear_history():
    if current_user.is_authenticated:
        mongo.db.users.update_one(
            {'_id': current_user._id},
            {'$set': {'history': [], 'searches': []}}
        )
    else:
        session['recent_views'] = []
        session['search_history'] = []
        session.modified = True
    return jsonify({'success': True})


@user_bp.route('/reset', methods=['POST'])
def reset_user_data():
    """Reset all user data (favorites, to-read, history, searches) for a fresh start with new dataset"""
    if current_user.is_authenticated:
        current_user.reset_all_data(mongo)
    else:
        session['favorites'] = []
        session['to_read'] = []
        session['recent_views'] = []
        session['search_history'] = []
        session['previous_recommendations'] = []
        session.modified = True
    return jsonify({'success': True, 'message': 'All data reset successfully'})


@user_bp.route('/favorites/clear', methods=['POST'])
def clear_favorites():
    """Clear all favorites"""
    if current_user.is_authenticated:
        current_user.clear_favorites(mongo)
    else:
        session['favorites'] = []
        session.modified = True
    return jsonify({'success': True, 'favorites': []})


@user_bp.route('/to-read/clear', methods=['POST'])
def clear_to_read():
    """Clear all to-read items"""
    if current_user.is_authenticated:
        current_user.clear_to_read(mongo)
    else:
        session['to_read'] = []
        session.modified = True
    return jsonify({'success': True, 'to_read': []})
