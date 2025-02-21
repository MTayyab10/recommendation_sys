import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recommendations.models import Review, Product
from sklearn.metrics import mean_squared_error

def create_interaction_matrix():
    """
    Build a user-item interaction matrix using reviews.
    For each review, use the product's parent_asin (if available)
    as the product identifier; otherwise, use asin.
    """
    reviews = Review.objects.all()
    user_item_matrix = {}
    all_items = set()  # To store all unique product identifiers

    # Step 1: Collect all items (using parent_asin when available) and user ratings
    for review in reviews:
        product_id = review.product.parent_asin if review.product.parent_asin else review.product.asin
        if review.user_id not in user_item_matrix:
            user_item_matrix[review.user_id] = {}
        user_item_matrix[review.user_id][product_id] = review.rating
        all_items.add(product_id)

    # Step 2: Ensure all users have ratings for the same set of items (fill missing values with 0)
    for user in user_item_matrix:
        for item in all_items:
            if item not in user_item_matrix[user]:
                user_item_matrix[user][item] = 0  # Fill missing with 0

    return user_item_matrix, list(all_items)

def calculate_similarity(matrix, all_items):
    """
    Convert the user-item interaction matrix to a NumPy array and compute
    cosine similarity between users.
    """
    user_ids = list(matrix.keys())
    if not user_ids or not all_items:
        raise ValueError("The interaction matrix is empty. Ensure that review data is loaded properly.")

    ratings_matrix = np.array([
        [matrix[user].get(item, 0) for item in all_items]
        for user in user_ids
    ])

    if ratings_matrix.ndim != 2 or ratings_matrix.size == 0:
        raise ValueError("The ratings matrix is empty or not two-dimensional. Check the data preprocessing steps.")

    similarity_matrix = cosine_similarity(ratings_matrix)
    return similarity_matrix, user_ids



def recommend_for_user(user_id, matrix, similarity_matrix, user_ids, n_recommendations=5):
    """
    Generate top-N product recommendations for a given user based on
    weighted ratings using user-based cosine similarity.
    """
    # Find index of the target user
    user_idx = user_ids.index(user_id)
    user_similarity = similarity_matrix[user_idx]

    # Re-create the ratings matrix (consistent with calculate_similarity)
    all_items = list(matrix[next(iter(matrix))].keys())
    ratings_matrix = np.array([
        [matrix[user].get(item, 0) for item in all_items]
        for user in user_ids
    ])

    # Compute the weighted sum of ratings
    weighted_ratings = np.dot(user_similarity, ratings_matrix)

    # Sort the products by predicted rating (highest first)
    recommended_indices = weighted_ratings.argsort()[-n_recommendations:][::-1]
    recommended_product_ids = [all_items[i] for i in recommended_indices]

    return recommended_product_ids

def calculate_rmse(predictions, actual_ratings):
    """
    Calculate the Root Mean Squared Error (RMSE) between predictions and actual ratings.
    """
    mse = mean_squared_error(actual_ratings, predictions)
    rmse = np.sqrt(mse)
    return rmse
