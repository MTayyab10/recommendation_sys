from django.http import JsonResponse
from .utils import create_interaction_matrix, calculate_similarity, recommend_for_user
from recommendations.models import Product

def recommend_products(request, user_id):
    # Step 1: Create the interaction matrix
    interaction_matrix = create_interaction_matrix()

    # Step 2: Calculate similarity matrix
    similarity_matrix, user_ids = calculate_similarity(interaction_matrix)

    # Step 3: Get recommendations for the user
    recommended_product_ids = recommend_for_user(user_id, interaction_matrix, similarity_matrix, user_ids)

    # Step 4: Fetch product details for recommended product ASINs
    recommended_products = Product.objects.filter(asin__in=recommended_product_ids)
    product_data = [
        {
            'asin': product.asin,
            'title': product.title,
            'price': product.price
        }
        for product in recommended_products
    ]

    # Return the product recommendations as JSON
    return JsonResponse({'recommendations': product_data})
