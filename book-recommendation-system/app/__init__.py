from flask import Flask
from flask_login import LoginManager
from flask_pymongo import PyMongo
import os

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

mongo = PyMongo()


def create_app():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, 'app', 'templates'),
        static_folder=os.path.join(base_dir, 'app', 'static')
    )
    
    from app.config import Config
    app.config.from_object(Config)
    
    # MongoDB configuration
    app.config['MONGO_URI'] = 'mongodb://localhost:27017/bookrec'
    
    # Initialize extensions
    mongo.init_app(app)
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        return User.find_by_id(mongo, user_id)
    
    # Register blueprints
    from app.routes.books import books_bp
    from app.routes.recommendations import recommendations_bp
    from app.routes.user import user_bp
    from app.routes.auth import auth_bp
    
    app.register_blueprint(books_bp)
    app.register_blueprint(recommendations_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(auth_bp)
    
    return app
