from flask import Blueprint, render_template, request, jsonify, session
from flask import redirect, url_for
from flask_login import current_user
from app.services.recommendation_engine import RecommendationEngine
from app import mongo

books_bp = Blueprint('books', __name__)

# Initialize recommendation engine
engine = None

def get_engine():
    global engine
    if engine is None:
        from app.config import Config
        engine = RecommendationEngine(Config.DATA_DIR, mongo)
    elif engine.mongo is None:
        engine.set_mongo(mongo)
    return engine

@books_bp.route('/')
def index():
    """Homepage with popular books and personalized recommendations"""
    rec_engine = get_engine()
    popular_books = rec_engine.get_popular_books(limit=12)
    
    # Get unified personalized recommendations based on ALL user activity
    recommended_books = []
    
    # Get previously shown recommendations to rotate some out
    previous_recs = session.get('previous_recommendations', [])
    
    if current_user.is_authenticated:
        favorites = current_user.get_favorite_isbns()
        to_read = current_user.get_to_read_isbns()
        history = current_user.get_history_isbns() if hasattr(current_user, 'get_history_isbns') else []
        searches = current_user.get_recent_searches() if hasattr(current_user, 'get_recent_searches') else []
        
        if favorites or to_read or history or searches:
            recommended_books = rec_engine.get_unified_recommendations(
                user_id=current_user._id,
                favorites=favorites,
                to_read=to_read,
                history=history,
                searches=searches,
                limit=12,
                previous_recommendations=previous_recs,
                rotation_ratio=0.4  # Rotate out 40% of old recommendations
            )
    else:
        # For non-authenticated users, use session data
        favorites = session.get('favorites', [])
        to_read = session.get('to_read', [])
        history = session.get('recent_views', [])
        searches = session.get('search_history', [])
        
        if favorites or to_read or history or searches:
            recommended_books = rec_engine.get_unified_recommendations(
                favorites=favorites,
                to_read=to_read,
                history=history,
                searches=searches,
                limit=12,
                previous_recommendations=previous_recs,
                rotation_ratio=0.4
            )
    
    # Store current recommendations for next rotation
    if recommended_books:
        session['previous_recommendations'] = [b['isbn'] for b in recommended_books]
        session.modified = True
    
    return render_template('index.html', 
                           popular_books=popular_books,
                           recommended_books=recommended_books)

@books_bp.route('/book/<isbn>')
def book_detail(isbn):
    """Book detail page"""
    rec_engine = get_engine()
    book = rec_engine.get_book(isbn)
    
    if not book:
        return render_template('books/detail.html', book=None)
    
    # Get similar books
    similar_books = rec_engine.get_hybrid_recommendations(isbn=isbn, limit=6)
    
    # Check if book is in favorites / to-read
    if current_user.is_authenticated:
        is_favorite = isbn in current_user.get_favorite_isbns()
        is_to_read = isbn in current_user.get_to_read_isbns()
    else:
        is_favorite = isbn in session.get('favorites', [])
        is_to_read = isbn in session.get('to_read', [])
    
    # Add to recent views
    if 'recent_views' not in session:
        session['recent_views'] = []
    if isbn not in session['recent_views']:
        session['recent_views'] = [isbn] + session['recent_views'][:9]
        session.modified = True
    
    return render_template('books/detail.html', 
                           book=book, 
                           similar_books=similar_books,
                           is_favorite=is_favorite,
                           is_to_read=is_to_read)

@books_bp.route('/search')
def search():
    """Search books"""
    query = request.args.get('q', '').strip()
    rec_engine = get_engine()
    
    # Save search query to history
    if query:
        if 'search_history' not in session:
            session['search_history'] = []
        if query not in session['search_history']:
            session['search_history'] = [query] + session['search_history'][:9]
            session.modified = True
        
        # Also save for logged-in users
        if current_user.is_authenticated:
            try:
                mongo.db.users.update_one(
                    {'_id': current_user._id},
                    {'$addToSet': {'search_history': query}}
                )
            except:
                pass
    
    if request.headers.get('Accept') == 'application/json':
        # AJAX request for autocomplete
        results = rec_engine.search_books(query, limit=8)
        return jsonify(results)
    
    # Regular search page
    results = rec_engine.search_books(query, limit=20)
    return render_template('books/search.html', query=query, results=results)

@books_bp.route('/browse')
def browse():
    """Browse all books with recommendations"""
    rec_engine = get_engine()
    page = request.args.get('page', 1, type=int)
    books, total = rec_engine.get_all_books(page=page, per_page=24)
    total_pages = (total + 23) // 24
    
    # Get recommendations based on user activity (only on first page)
    recommended_books = []
    previous_recs = session.get('previous_recommendations', [])
    
    if page == 1:
        if current_user.is_authenticated:
            favorites = current_user.get_favorite_isbns()
            to_read = current_user.get_to_read_isbns()
            history = current_user.get_history_isbns() if hasattr(current_user, 'get_history_isbns') else []
            searches = current_user.get_recent_searches() if hasattr(current_user, 'get_recent_searches') else []
            
            if favorites or to_read or history or searches:
                recommended_books = rec_engine.get_unified_recommendations(
                    user_id=current_user._id,
                    favorites=favorites,
                    to_read=to_read,
                    history=history,
                    searches=searches,
                    limit=8,
                    previous_recommendations=previous_recs,
                    rotation_ratio=0.4
                )
        else:
            favorites = session.get('favorites', [])
            to_read = session.get('to_read', [])
            history = session.get('recent_views', [])
            searches = session.get('search_history', [])
            
            if favorites or to_read or history or searches:
                recommended_books = rec_engine.get_unified_recommendations(
                    favorites=favorites,
                    to_read=to_read,
                    history=history,
                    searches=searches,
                    limit=8,
                    previous_recommendations=previous_recs,
                    rotation_ratio=0.4
                )
        
        # Store current recommendations for next rotation
        if recommended_books:
            session['previous_recommendations'] = [b['isbn'] for b in recommended_books]
            session.modified = True
    
    return render_template('books/list.html', 
                           books=books, 
                           page=page, 
                           total_pages=total_pages,
                           recommended_books=recommended_books)


@books_bp.route('/favorites')
def favorites_page():
    # Require login for viewing favorites
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login', next=url_for('books.favorites_page')))

    rec_engine = get_engine()
    fav_isbns = current_user.get_favorite_isbns()
    books = [rec_engine.get_book(isbn) for isbn in fav_isbns if rec_engine.get_book(isbn)]
    return render_template('books/favorites.html', books=books)


@books_bp.route('/to-read')
def to_read_page():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login', next=url_for('books.to_read_page')))

    rec_engine = get_engine()
    to_read_isbns = current_user.get_to_read_isbns()
    books = [rec_engine.get_book(isbn) for isbn in to_read_isbns if rec_engine.get_book(isbn)]
    return render_template('books/to_read.html', books=books)
