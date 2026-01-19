from app import create_app
import unittest

class TestBooks(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_get_books(self):
        response = self.client.get('/api/books')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json, list)

    def test_get_book_details(self):
        response = self.client.get('/api/books/1')
        self.assertEqual(response.status_code, 200)
        self.assertIn('title', response.json)
        self.assertIn('author', response.json)

    def test_search_books(self):
        response = self.client.get('/api/books/search?q=some_book_title')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json, list)

    def test_add_book_rating(self):
        response = self.client.post('/api/books/1/rate', json={'rating': 5})
        self.assertEqual(response.status_code, 201)
        self.assertIn('message', response.json)

    def test_get_book_recommendations(self):
        response = self.client.get('/api/recommendations')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json, list)

if __name__ == '__main__':
    unittest.main()