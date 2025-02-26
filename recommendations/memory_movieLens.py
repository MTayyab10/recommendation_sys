# recommendations/memory_movieLens.py
from recommendations.models import Rating
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def normalize_ratings(memory_matrix):
    """
    Normalize ratings in the memory matrix by subtracting the user's mean rating
    """
    # Normalize user ratings by subtracting the user's average rating
    normalized_matrix = {}
    for user, item_ratings in memory_matrix.items():
        user_mean = np.mean(list(item_ratings.values()))
        normalized_matrix[user] = {movie: rating - user_mean for movie, rating in item_ratings.items()}
    return normalized_matrix

def create_interaction_matrix():
    """
    Build a user–item interaction matrix from the MovieLens ratings.
    """
    ratings = Rating.objects.all()
    user_item_matrix = {}
    all_items = set()

    for rating in ratings:
        user = rating.user_id
        movie_id = rating.movie.movieId  # or simply rating.movie_id
        if user not in user_item_matrix:
            user_item_matrix[user] = {}
        user_item_matrix[user][movie_id] = rating.rating
        all_items.add(movie_id)

    # Ensure every user has a rating (or zero) for every movie in all_items
    for user in user_item_matrix:
        for item in all_items:
            if item not in user_item_matrix[user]:
                user_item_matrix[user][item] = 0

    # Normalize the ratings
    normalized_matrix = normalize_ratings(user_item_matrix)

    return normalized_matrix, list(all_items)

def calculate_similarity(matrix, all_items):
    """
    Convert the normalized user–item matrix into a NumPy array and compute cosine similarity.
    """
    user_ids = list(matrix.keys())
    ratings_matrix = np.array([[matrix[user].get(item, 0) for item in all_items] for user in user_ids])
    similarity_matrix = cosine_similarity(ratings_matrix)
    return similarity_matrix, user_ids

def recommend_for_user(user_id, matrix, similarity_matrix, user_ids, n_recommendations=10):
    """
    Generate top-n movie recommendations for a given user based on weighted ratings.
    """
    user_idx = user_ids.index(user_id)
    user_similarity = similarity_matrix[user_idx]
    all_items = list(matrix[next(iter(matrix))].keys())
    ratings_matrix = np.array([[matrix[user].get(item, 0) for item in all_items] for user in user_ids])
    weighted_ratings = np.dot(user_similarity, ratings_matrix)
    recommended_indices = weighted_ratings.argsort()[-n_recommendations:][::-1]
    recommended_movie_ids = [all_items[i] for i in recommended_indices]
    return recommended_movie_ids
