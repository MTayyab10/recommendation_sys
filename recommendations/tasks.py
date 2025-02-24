# recommendations/tasks.py
from celery import shared_task
from recommendations.memory_amazonReviews import create_interaction_matrix, calculate_similarity
from recommendations.content_amazonReviews import content_based_recommendation
from recommendations.hybrid_amazonReviews import hybrid_recommendation
from recommendations.methods import load_svd_model, matrix_factorization_recommendation, memory_based_recommendation
from django.core.cache import cache


@shared_task
def async_memory_based_recommendations(user_id):
    # Compute recommendations using the memory based method
    recs = memory_based_recommendation(user_id)
    # Cache the result for 5 minutes
    cache.set(f"recs_mf_{user_id}", recs, 600)
    return recs


@shared_task
def async_content_based_recommendations(user_id):
    # Compute recommendations using the memory based method
    recs = content_based_recommendation(user_id)
    # Cache the result for 5 minutes
    cache.set(f"recs_mf_{user_id}", recs, 600)
    return recs


@shared_task
def async_mf_recommendations(user_id):
    # Compute recommendations using the matrix factorization method
    recs = matrix_factorization_recommendation(user_id)
    # Cache the result for 5 minutes
    cache.set(f"recs_mf_{user_id}", recs, 600)
    return recs


@shared_task
def async_hybrid_recommendations(user_id):
    """
    Asynchronously generate hybrid recommendations for a given user.
    This task builds the interaction matrix, computes similarity, loads the SVD model,
    and then uses the hybrid recommendation function.
    """
    # Build the interaction matrix and similarity matrix
    memory_matrix, all_items = create_interaction_matrix()
    similarity_matrix, user_ids = calculate_similarity(memory_matrix, all_items)

    # Define candidate items (e.g., top 100 popular items)
    candidate_asins = list(all_items)[:100]

    # Load the pre-trained SVD model
    mf_model = load_svd_model()

    # Generate hybrid recommendations with dynamic weighting
    recommended_ids = hybrid_recommendation(user_id, memory_matrix, similarity_matrix, user_ids,
                                            mf_model, candidate_asins, dynamic=True, n_recommendations=10)

    # Optionally, you can cache the result here for future requests
    cache_key = f"hybrid_recs_{user_id}"
    cache.set(cache_key, recommended_ids, 600)  # Cache for 10 minutes

    return recommended_ids
