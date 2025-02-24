import pickle
from .memory_movieLens import create_interaction_matrix, calculate_similarity, recommend_for_user
from recommendations.models import Product


def memory_based_recommendation(user_id):
    """
    Generate product recommendations for a given user using memory-based collaborative filtering.
    """
    # Step 1: Create the interaction matrix and get all product IDs
    user_item_matrix, all_items = create_interaction_matrix()

    # Step 2: Compute cosine similarity between users
    similarity_matrix, user_ids = calculate_similarity(user_item_matrix, all_items)

    # Step 3: Generate recommendations for the given user_id
    recommended_ids = recommend_for_user(user_id, user_item_matrix, similarity_matrix, user_ids, n_recommendations=5)
    return recommended_ids

def matrix_factorization_recommendation(user_id):
    """
    Generate product recommendations for a given user using matrix factorization.
    """
    # Load the pre-trained SVD model
    model = load_svd_model()
    # Fetch all products from the enriched Product model
    products = Product.objects.all()
    predictions = []

    # For each product, use parent_asin if available for consistency
    for product in products:
        prod_id = product.parent_asin if product.parent_asin else product.asin
        # Predict the rating for this user-product pair using the SVD model
        pred = model.predict(user_id, prod_id)
        predictions.append((prod_id, pred.est))

    # Sort by predicted rating (highest first) and take top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_ids = [prod_id for prod_id, _ in predictions[:5]]
    return top_ids

def load_svd_model():
    """
    Load the pre-trained SVD model from a pickle file.
    """
    with open('svd_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
