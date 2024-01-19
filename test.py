import unittest

from main import app


class TestSearchMovies(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_search_movies(self):
        query = "A movie about the Earth, Mars and an invasion."

        response = self.app.post("/movies/search", json={"query": query})

        self.assertEqual(response.status_code, 200)

        results = response.get_json()
        self.assertIsNotNone(results)
        self.assertTrue("results" in results)

        movies = results["results"]
        self.assertIsNotNone(movies)
        self.assertGreater(len(movies), 0)

        first_movie = movies[0]
        self.assertIsNotNone(first_movie)
        self.assertTrue("title" in first_movie)
        self.assertEqual(first_movie["title"], "The War of the Worlds")


if __name__ == "__main__":
    unittest.main()
