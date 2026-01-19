from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from bson import ObjectId


class User(UserMixin):
    def __init__(self, user_data):
        self.user_data = user_data
        self._id = user_data.get('_id')
        self.username = user_data.get('username')
        self.email = user_data.get('email')
        self.password_hash = user_data.get('password_hash')
        self.created_at = user_data.get('created_at', datetime.utcnow())
        self.favorites = user_data.get('favorites', [])
        self.to_read = user_data.get('to_read', [])
        self.history = user_data.get('history', [])
        self.searches = user_data.get('searches', [])
    
    def get_id(self):
        return str(self._id)
    
    @staticmethod
    def create_user(mongo, username, email, password):
        user_data = {
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password, method='pbkdf2:sha256'),
            'created_at': datetime.utcnow(),
            'favorites': [],
            'to_read': [],
            'history': [],
            'searches': []
        }
        result = mongo.db.users.insert_one(user_data)
        user_data['_id'] = result.inserted_id
        return User(user_data)
    
    @staticmethod
    def find_by_email(mongo, email):
        user_data = mongo.db.users.find_one({'email': email})
        return User(user_data) if user_data else None
    
    @staticmethod
    def find_by_username(mongo, username):
        user_data = mongo.db.users.find_one({'username': username})
        return User(user_data) if user_data else None
    
    @staticmethod
    def find_by_id(mongo, user_id):
        try:
            user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
            return User(user_data) if user_data else None
        except:
            return None
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def add_favorite(self, mongo, isbn):
        if isbn not in self.favorites:
            mongo.db.users.update_one(
                {'_id': self._id},
                {'$addToSet': {'favorites': isbn}}
            )
            self.favorites.append(isbn)
    
    def remove_favorite(self, mongo, isbn):
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$pull': {'favorites': isbn}}
        )
        if isbn in self.favorites:
            self.favorites.remove(isbn)

    def add_to_read(self, mongo, isbn):
        if isbn not in self.to_read:
            mongo.db.users.update_one(
                {'_id': self._id},
                {'$addToSet': {'to_read': isbn}}
            )
            self.to_read.append(isbn)

    def remove_from_to_read(self, mongo, isbn):
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$pull': {'to_read': isbn}}
        )
        if isbn in self.to_read:
            self.to_read.remove(isbn)
    
    def add_to_history(self, mongo, isbn):
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$push': {'history': {'$each': [{'isbn': isbn, 'viewed_at': datetime.utcnow()}], '$position': 0, '$slice': 50}}}
        )
    
    def add_search(self, mongo, query):
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$push': {'searches': {'$each': [{'query': query, 'searched_at': datetime.utcnow()}], '$position': 0, '$slice': 20}}}
        )
    
    def get_favorite_isbns(self):
        return self.favorites

    def get_to_read_isbns(self):
        return self.to_read
    
    def get_history_isbns(self, limit=20):
        return [h['isbn'] for h in self.history[:limit]]
    
    def get_recent_searches(self, limit=10):
        return [s['query'] for s in self.searches[:limit]]
    
    def get_favorites_count(self):
        return len(self.favorites)
    
    def get_history_count(self):
        return len(self.history)
    
    def clear_favorites(self, mongo):
        """Clear all favorites for this user"""
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$set': {'favorites': []}}
        )
        self.favorites = []
    
    def clear_to_read(self, mongo):
        """Clear all to-read items for this user"""
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$set': {'to_read': []}}
        )
        self.to_read = []
    
    def clear_history(self, mongo):
        """Clear browsing history for this user"""
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$set': {'history': []}}
        )
        self.history = []
    
    def clear_searches(self, mongo):
        """Clear search history for this user"""
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$set': {'searches': []}}
        )
        self.searches = []
    
    def reset_all_data(self, mongo):
        """Reset all user data (favorites, to-read, history, searches)"""
        mongo.db.users.update_one(
            {'_id': self._id},
            {'$set': {
                'favorites': [],
                'to_read': [],
                'history': [],
                'searches': []
            }}
        )
        self.favorites = []
        self.to_read = []
        self.history = []
        self.searches = []
