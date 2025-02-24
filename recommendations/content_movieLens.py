import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Movie, Rating

# Global variables to cache the TF-IDF matrix, movie IDs, and vectorizer
tfidf_matrix = None
movie_ids = None
tfidf_vectorizer = None

def build_movie_tfidf_matrix():
    """
    Build and cache a TF-IDF matrix for all movies using their title and genres.
    If the 'genres' field is missing or empty, it falls back to using the title only.
    """
    global tfidf_matrix, movie_ids, tfidf_vectorizer

    # Fetch movies from the database
    movies = Movie.objects.all().values('movieId', 'title', 'genres')
    df = pd.DataFrame(list(movies))
    if df.empty:
        return None, None, None

    # Fill missing values and ensure text fields are strings
    df['title'] = df['title'].fillna('').astype(str)
    df['genres'] = df['genres'].fillna('').astype(str)

    # Combine title and genres into one text field
    df['text'] = df['title'] + " " + df['genres']
    movie_ids = df['movieId'].tolist()

    # Initialize and fit the TF-IDF vectorizer on the combined text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    return tfidf_matrix, movie_ids, tfidf_vectorizer

def content_based_recommendation(user_id, top_n=10):
    """
    Generate content-based recommendations for a given user.
    Computes an average TF-IDF vector for the movies the user has rated,
    then computes cosine similarity between this vector and all movie vectors.
    """
    global tfidf_matrix, movie_ids, tfidf_vectorizer
    if tfidf_matrix is None or movie_ids is None:
        tfidf_matrix, movie_ids, tfidf_vectorizer = build_movie_tfidf_matrix()
        if tfidf_matrix is None:
            return []

    # Get the list of movie IDs the user has rated from the Rating model
    reviewed_movie_ids = list(Rating.objects.filter(user_id=user_id)
                              .values_list('movie__movieId', flat=True))
    if not reviewed_movie_ids:
        return []

    # Map each movieId to its index in the TF-IDF matrix
    movieid_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    indices = [movieid_to_index[movie_id] for movie_id in reviewed_movie_ids if movie_id in movieid_to_index]
    if not indices:
        return []

    # Compute the average TF-IDF vector for the user’s rated movies
    user_vector = np.asarray(tfidf_matrix[indices].mean(axis=0))
    # Compute cosine similarity between the user vector and all movie vectors
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Create a DataFrame of movie IDs and similarity scores
    sim_df = pd.DataFrame({'movieId': movie_ids, 'score': sim_scores})
    # Exclude movies the user has already rated
    sim_df = sim_df[~sim_df['movieId'].isin(reviewed_movie_ids)]
    sim_df = sim_df.sort_values(by='score', ascending=False)

    recommended = sim_df.head(top_n)['movieId'].tolist()
    return recommended
