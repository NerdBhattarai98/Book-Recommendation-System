# Book Recommendation System

This project is a Book Recommendation System built using Flask for the backend and HTML, CSS, and JavaScript for the frontend. The application recommends books based on users' previous searches, read lists, or favorites, utilizing both collaborative filtering and content-based filtering methods.

## Features

- User authentication (login and registration)
- Browse and search books
- View book details
- Personalized book recommendations based on user preferences
- Collaborative filtering for recommendations based on user ratings
- Content-based filtering for recommendations based on book features

## Project Structure

```
book-recommendation-system
├── app
│   ├── __init__.py              # Initializes the Flask application
│   ├── config.py                 # Configuration settings
│   ├── routes                    # API routes
│   │   ├── __init__.py
│   │   ├── auth.py               # User authentication routes
│   │   ├── books.py              # Book-related routes
│   │   ├── recommendations.py     # Recommendation routes
│   │   └── user.py               # User-related routes
│   ├── models                    # Database models
│   │   ├── __init__.py
│   │   ├── book.py               # Book model
│   │   ├── user.py               # User model
│   │   ├── rating.py             # Rating model
│   │   └── reading_list.py       # Reading list model
│   ├── services                  # Recommendation services
│   │   ├── __init__.py
│   │   ├── recommendation_engine.py # Main recommendation logic
│   │   ├── collaborative_filtering.py # Collaborative filtering algorithms
│   │   └── content_based_filtering.py # Content-based filtering algorithms
│   ├── utils                     # Utility functions
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Data loading functions
│   │   ├── similarity.py         # Similarity calculations
│   │   └── helpers.py            # Helper functions
│   ├── static                    # Static files (CSS, JS)
│   │   ├── css
│   │   │   ├── main.css
│   │   │   ├── books.css
│   │   │   └── auth.css
│   │   ├── js
│   │   │   ├── main.js
│   │   │   ├── recommendations.js
│   │   │   ├── search.js
│   │   │   └── favorites.js
│   │   └── assets
│   │       └── .gitkeep
│   └── templates                 # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── login.html
│       ├── register.html
│       ├── profile.html
│       ├── books
│       │   ├── list.html
│       │   ├── detail.html
│       │   └── search.html
│       └── recommendations
│           └── recommendations.html
├── data                          # Datasets
│   ├── books.csv
│   ├── users.csv
│   └── ratings.csv
├── tests                         # Unit tests
│   ├── __init__.py
│   ├── test_auth.py
│   ├── test_books.py
│   ├── test_collaborative_filtering.py
│   └── test_content_based_filtering.py
├── run.py                       # Entry point to run the application
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd book-recommendation-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables in a `.env` file based on `.env.example`.

5. Run the application:
   ```
   python run.py
   ```

## Usage

- Navigate to `http://localhost:5000` in your web browser.
- Register a new account or log in to an existing account.
- Browse books, search for specific titles, and view recommendations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.