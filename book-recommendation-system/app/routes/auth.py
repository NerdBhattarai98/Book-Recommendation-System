from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User
from app import mongo
from app.services.recommendation_engine import RecommendationEngine
import os

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# Cache for the recommendation engine
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        _engine = RecommendationEngine(data_path, mongo)
    return _engine


def _count_valid_isbns(isbns):
    """Count ISBNs that exist in the current dataset."""
    engine = _get_engine()
    count = 0
    for isbn in isbns:
        book = engine.get_book(isbn)
        if book is not None:
            count += 1
    return count


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('books.index'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        user = User.find_by_email(mongo, email)
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash('Welcome back!', 'success')
            return redirect(next_page or url_for('books.index'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('auth/login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('books.index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        errors = []
        
        if len(username) < 3:
            errors.append('Username must be at least 3 characters')
        
        if '@' not in email:
            errors.append('Invalid email address')
        
        if len(password) < 6:
            errors.append('Password must be at least 6 characters')
        
        if password != confirm_password:
            errors.append('Passwords do not match')
        
        if User.find_by_username(mongo, username):
            errors.append('Username already taken')
        
        if User.find_by_email(mongo, email):
            errors.append('Email already registered')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('auth/register.html')
        
        user = User.create_user(mongo, username, email, password)
        login_user(user)
        flash('Account created successfully!', 'success')
        return redirect(url_for('books.index'))
    
    return render_template('auth/register.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('books.index'))


@auth_bp.route('/profile')
@login_required
def profile():
    # Count only favorites/to-read that exist in current dataset
    favorites_count = _count_valid_isbns(current_user.favorites)
    to_read_count = _count_valid_isbns(current_user.to_read)
    history_count = current_user.get_history_count()
    
    return render_template('auth/profile.html', 
                           favorites_count=favorites_count,
                           to_read_count=to_read_count,
                           history_count=history_count)
