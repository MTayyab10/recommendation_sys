import numpy as np
from recommendations.memory_based import create_interaction_matrix, calculate_similarity
from recommendations.methods import load_svd_model  # Ensure load_svd_model is defined in your methods module


def get_memory_based_scores(user_id, memory_matrix, similarity_matrix, user_ids):
    """
    Compute memory-based recommendation scores for a given user.
    Returns a dictionary mapping each product (asin) to a computed score.
    """
    try:
        user_idx = user_ids.index(user_id)
    except ValueError:
        return {}

    user_similarity = similarity_matrix[user_idx]
    # Assuming every user has the same set of items
    all_items = list(memory_matrix[next(iter(memory_matrix))].keys())
    scores = {asin: 0 for asin in all_items}

    # Aggregate weighted ratings from all users based on similarity
    for idx, other_user in enumerate(user_ids):
        sim = user_similarity[idx]
        for asin, rating in memory_matrix[other_user].items():
            scores[asin] += sim * rating
    return scores


def get_svd_based_scores(user_id, mf_model, candidate_asins):
    """
    Predict ratings for a given user for the candidate items using the SVD model.
    Returns a dictionary mapping each candidate product (asin) to the predicted rating.
    """
    scores = {}
    for asin in candidate_asins:
        pred = mf_model.predict(user_id, asin)
        scores[asin] = pred.est
    return scores


def compute_dynamic_weights(user_id, memory_matrix, threshold=10):
    """
    Compute dynamic weights based on the number of interactions for a given user.
    If the user has fewer than 'threshold' interactions, assign a higher weight to the SVD model;
    otherwise, assign a higher weight to the memory-based model.
    Returns (w_memory, w_svd).
    """
    user_data = memory_matrix.get(user_id, {})
    interaction_count = sum(1 for rating in user_data.values() if rating > 0)
    if interaction_count < threshold:
        return 0.3, 0.7  # More weight to SVD-based scores when interactions are few
    else:
        return 0.7, 0.3  # More weight to memory-based scores when sufficient interactions exist


def hybrid_recommendation(user_id, memory_matrix, similarity_matrix, user_ids, mf_model, candidate_asins,
                          dynamic=True, n_recommendations=10):
    """
    Generate top-N hybrid recommendations by combining memory-based and SVD-based scores.

    Parameters:
      - user_id: The target user's ID.
      - memory_matrix: The user-item interaction matrix from memory-based CF.
      - similarity_matrix: The cosine similarity matrix for users.
      - user_ids: List of user IDs corresponding to the rows in memory_matrix.
      - mf_model: Pre-trained SVD model loaded from disk.
      - candidate_asins: List of candidate product IDs to consider.
      - dynamic: If True, compute dynamic weights based on user interaction count.
      - n_recommendations: Number of recommendations to return.

    Returns:
      A list of recommended product ASINs.
    """
    # Obtain memory-based scores
    mem_scores = get_memory_based_scores(user_id, memory_matrix, similarity_matrix, user_ids)

    # Obtain SVD-based scores for candidate items
    svd_scores = get_svd_based_scores(user_id, mf_model, candidate_asins)

    # Determine weights dynamically if enabled
    if dynamic:
        w_memory, w_svd = compute_dynamic_weights(user_id, memory_matrix)
    else:
        w_memory, w_svd = 0.5, 0.5  # Default static weights

    # Combine scores: if an item is missing from one method, its score defaults to 0
    hybrid_scores = {}
    all_asins = set(mem_scores.keys()) | set(svd_scores.keys())
    for asin in all_asins:
        score_mem = mem_scores.get(asin, 0)
        score_svd = svd_scores.get(asin, 0)
        hybrid_scores[asin] = w_memory * score_mem + w_svd * score_svd

    # Sort items by combined score and return top-N recommendations
    sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [asin for asin, score in sorted_items[:n_recommendations]]
    return top_recommendations
