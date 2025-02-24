import numpy as np
from recommendations.memory_movieLens import create_interaction_matrix, calculate_similarity
from recommendations.methods import load_svd_model  # Ensure load_svd_model is defined in your methods module
import math


def get_memory_based_scores(user_id, memory_matrix, similarity_matrix, user_ids):
    """
    Compute memory-based recommendation scores for a given user.
    Returns a dictionary mapping each movie (movieId) to a computed score.
    """
    try:
        user_idx = user_ids.index(user_id)
    except ValueError:
        return {}

    user_similarity = similarity_matrix[user_idx]
    # Assuming every user has the same set of movies
    all_movieIds = list(memory_matrix[next(iter(memory_matrix))].keys())
    scores = {movieId: 0 for movieId in all_movieIds}

    # Aggregate weighted ratings from all users based on similarity
    for idx, other_user in enumerate(user_ids):
        sim = user_similarity[idx]
        for movieId, rating in memory_matrix[other_user].items():
            scores[movieId] += sim * rating
    return scores


def get_svd_based_scores(user_id, mf_model, candidate_movies):
    """
    Predict ratings for a given user for the candidate movies using the SVD model.
    Returns a dictionary mapping each candidate movie (movieId) to the predicted rating.
    """
    scores = {}
    for movieId in candidate_movies:
        pred = mf_model.predict(user_id, movieId)
        scores[movieId] = pred.est
    return scores


def compute_dynamic_weights(user_id, memory_matrix, threshold=10, alpha=0.5):
    """
    Compute dynamic weights for the hybrid model based on the user's number of interactions.

    Uses a logistic function:
      w_memory = 1 / (1 + exp(-alpha * (interaction_count - threshold)))
      w_svd = 1 - w_memory
    """
    user_data = memory_matrix.get(user_id, {})
    interaction_count = sum(1 for rating in user_data.values() if rating > 0)
    w_memory = 1 / (1 + math.exp(-alpha * (interaction_count - threshold)))
    w_svd = 1 - w_memory
    return w_memory, w_svd


def hybrid_recommendation(user_id, memory_matrix, similarity_matrix, user_ids, mf_model, candidate_movies,
                          dynamic=True, n_recommendations=10):
    """
    Generate top-N hybrid recommendations by combining memory-based and SVD-based scores.

    Parameters:
      - user_id: The target user's ID.
      - memory_matrix: The user–movie interaction matrix from memory-based CF.
      - similarity_matrix: The cosine similarity matrix for users.
      - user_ids: List of user IDs corresponding to the rows in memory_matrix.
      - mf_model: Pre-trained SVD model loaded from disk.
      - candidate_movies: List of candidate movie IDs to consider.
      - dynamic: If True, compute dynamic weights based on user interaction count.
      - n_recommendations: Number of recommendations to return.

    Returns:
      A list of recommended movie IDs.
    """
    # Obtain memory-based scores
    mem_scores = get_memory_based_scores(user_id, memory_matrix, similarity_matrix, user_ids)
    # Obtain SVD-based scores for candidate movies
    svd_scores = get_svd_based_scores(user_id, mf_model, candidate_movies)
    # Determine weights dynamically if enabled
    if dynamic:
        w_memory, w_svd = compute_dynamic_weights(user_id, memory_matrix)
    else:
        w_memory, w_svd = 0.5, 0.5  # Default static weights
    # Combine scores (defaulting missing scores to 0)
    hybrid_scores = {}
    all_movieIds = set(mem_scores.keys()) | set(svd_scores.keys())
    for movieId in all_movieIds:
        score_mem = mem_scores.get(movieId, 0)
        score_svd = svd_scores.get(movieId, 0)
        hybrid_scores[movieId] = w_memory * score_mem + w_svd * score_svd
    # Return top-N recommendations by combined score
    sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [movieId for movieId, score in sorted_items[:n_recommendations]]
    return top_recommendations
