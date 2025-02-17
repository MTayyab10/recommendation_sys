from django.http import JsonResponse
from .utils import create_interaction_matrix, calculate_similarity, recommend_for_user
from recommendations.models import Product, Review
from rest_framework import viewsets
from .serializers import ProductSerializer, ReviewSerializer


def recommend_products(request, user_id):
    """
    API endpoint to get product recommendations for a given user.
    It builds the user-item interaction matrix from reviews,
    computes user similarities, and then returns top-N product recommendations.
    """
    # Step 1: Create the interaction matrix and get all product IDs (using enriched meta info)
    user_item_matrix, all_items = create_interaction_matrix()

    # Step 2: Compute cosine similarity between users
    similarity_matrix, user_ids = calculate_similarity(user_item_matrix, all_items)

    # Step 3: Generate recommendations for the given user_id
    recommended_ids = recommend_for_user(user_id, user_item_matrix, similarity_matrix, user_ids, n_recommendations=5)

    # Step 4: Query enriched Product details for each recommended product
    products = Product.objects.filter(asin__in=recommended_ids)
    product_list = []
    for product in products:
        product_list.append({
            'asin': product.asin,
            'title': product.title,
            'price': product.price,
            'features': product.features,
            'images': product.images
        })

    return JsonResponse({'recommendations': product_list})


class ProductViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows products to be viewed or edited.
    """
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class ReviewViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows reviews to be viewed or edited.
    """
    queryset = Review.objects.all()
    serializer_class = ReviewSerializer

