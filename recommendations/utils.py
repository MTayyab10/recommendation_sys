import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recommendations.models import Review, Product


def create_interaction_matrix():
    # Fetch all reviews and create a user-item interaction matrix
    reviews = Review.objects.all()
    user_item_matrix = {}
    for review in reviews:
        if review.user_id not in user_item_matrix:
            user_item_matrix[review.user_id] = {}
        user_item_matrix[review.user_id][review.product.asin] = review.rating
    return user_item_matrix


def calculate_similarity(matrix):
    # Calculate the similarity between users using cosine similarity
    user_ids = list(matrix.keys())
    ratings_matrix = np.array([list(matrix[user].values()) for user in user_ids])
    similarity_matrix = cosine_similarity(ratings_matrix)
    return similarity_matrix, user_ids


def recommend_for_user(user_id, matrix, similarity_matrix, user_ids, n_recommendations=5):
    user_idx = user_ids.index(user_id)
    user_similarity = similarity_matrix[user_idx]

    # Get the weighted sum of ratings based on user similarity
    weighted_ratings = np.dot(user_similarity, matrix)

    # Sort products by predicted rating and get the top-N recommendations
    recommended_products_idx = weighted_ratings.argsort()[-n_recommendations:][::-1]
    recommended_products = [list(matrix[user_ids[idx]].keys())[recommended_products_idx[i]] for i in
                            range(n_recommendations)]

    return recommended_products
