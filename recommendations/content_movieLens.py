# recommendations/content_movieLens.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommendations.models import Movie, Rating

# Global variables to cache the TF-IDF matrix, movie IDs, and vectorizer
tfidf_matrix = None
movie_ids = None
tfidf_vectorizer = None


def build_movie_tfidf_matrix():
    global tfidf_matrix, movie_ids, tfidf_vectorizer
    # Fetch movies from the database
    movies = Movie.objects.all().values('movieId', 'title', 'genres')
    df = pd.DataFrame(list(movies))
    if df.empty:
        return None, None, None

    df['title'] = df['title'].fillna('').astype(str)
    df['genres'] = df['genres'].fillna('').astype(str)
    df['text'] = df['title'] + " " + df['genres']

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    # print("Sample TF-IDF matrix for first movie:", tfidf_matrix[0].toarray())  # Check the first movie's vector
    movie_ids = df['movieId'].tolist()

    return tfidf_matrix, movie_ids, tfidf_vectorizer


MIN_RATED_MOVIES = 3  # Minimum number of movies a user should rate to get recommendations
def content_based_recommendation(user_id, top_n=10):
    global tfidf_matrix, movie_ids, tfidf_vectorizer
    if tfidf_matrix is None or movie_ids is None:
        tfidf_matrix, movie_ids, tfidf_vectorizer = build_movie_tfidf_matrix()

    reviewed_movie_ids = list(Rating.objects.filter(user_id=user_id).values_list('movie__movieId', flat=True))
    if not reviewed_movie_ids:
        return []

    movieid_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    indices = [movieid_to_index[movie_id] for movie_id in reviewed_movie_ids if movie_id in movieid_to_index]
    if len(reviewed_movie_ids) < MIN_RATED_MOVIES:
        # print(f"User {user_id} has not rated enough movies.")
        return []  # Return empty if the user hasn't rated enough movies

    # Print the individual vectors for the rated movies
    movie_vectors = tfidf_matrix[indices].toarray()
    # print(f"Movie vectors for user {user_id}: {movie_vectors[:5]}")  # Show first 5 vectors

    # Compute the average vector for the user
    user_vector = np.asarray(tfidf_matrix[indices].mean(axis=0))
    if user_vector.ndim == 1:
        user_vector = user_vector.reshape(1, -1)

    # Print the user vector to debug
    # print(f"User vector for user {user_id}: {user_vector}")

    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    sim_df = pd.DataFrame({'movieId': movie_ids, 'score': sim_scores})
    sim_df = sim_df[~sim_df['movieId'].isin(reviewed_movie_ids)]
    sim_df = sim_df.sort_values(by='score', ascending=False)

    recommended = sim_df.head(top_n)['movieId'].tolist()
    return recommended


def fallback_recommendation(user_id, candidate_movies, top_n=10):
    """
    Fallback recommendation using popular movies.
    Returns the top_n popular movies (from candidate_movies) that the user hasn't rated.
    """
    # Get movies that the user has already rated
    user_rated = set(
        Rating.objects.filter(user_id=user_id).values_list('movie__movieId', flat=True)
    )
    # Recommend popular movies the user hasn't seen
    recommended = [movieId for movieId in candidate_movies if movieId not in user_rated]
    return recommended[:top_n]

