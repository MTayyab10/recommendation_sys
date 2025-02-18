from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Product
from .serializers import ProductSerializer
from .recommendation_methods import memory_based_recommendation, matrix_factorization_recommendation

class ProductViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows products to be viewed or edited.
    """
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

    @action(detail=False, methods=['get'], url_path='recommendations/memory')
    def memory_based(self, request):
        """
        Retrieve product recommendations using memory-based collaborative filtering.
        """
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id parameter is required'}, status=400)
        recommended_ids = memory_based_recommendation(user_id)
        products = Product.objects.filter(asin__in=recommended_ids)
        serializer = self.get_serializer(products, many=True)
        return Response({'recommendations': serializer.data})

    @action(detail=False, methods=['get'], url_path='recommendations/mf')
    def matrix_factorization(self, request):
        """
        Retrieve product recommendations using matrix factorization.
        """
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id parameter is required'}, status=400)
        recommended_ids = matrix_factorization_recommendation(user_id)
        products = Product.objects.filter(asin__in=recommended_ids)
        serializer = self.get_serializer(products, many=True)
        return Response({'recommendations': serializer.data})


# import pickle
# from django.http import JsonResponse
# from recommendations.models import Product, Review
# from rest_framework import viewsets
# from .serializers import ProductSerializer, ReviewSerializer
#
#
# # Function to load the pre-trained SVD model
# def load_svd_model():
#     with open('svd_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     return model
#
#
# # Function to generate recommendations using the SVD model
# def get_recommendations(user_id, model, n_recommendations=5):
#     # Fetch all products from the enriched Product model
#     products = Product.objects.all()
#     predictions = []
#
#     # For each product, use parent_asin if available for consistency
#     for product in products:
#         prod_id = product.parent_asin if product.parent_asin else product.asin
#         # Predict the rating for this user-product pair using the SVD model
#         pred = model.predict(user_id, prod_id)
#         predictions.append((prod_id, pred.est))
#
#     # Sort by predicted rating (highest first) and take top-N
#     predictions.sort(key=lambda x: x[1], reverse=True)
#     top_ids = [prod_id for prod_id, _ in predictions[:n_recommendations]]
#     return top_ids
#
#
# def recommend_products(request, user_id):
#     """
#     API endpoint to get product recommendations for a given user using the SVD-based matrix factorization model.
#     """
#     # Load the pre-trained model
#     model = load_svd_model()
#     # Get top recommendations for the user
#     recommended_ids = get_recommendations(user_id, model, n_recommendations=5)
#
#     # Retrieve enriched product details for recommended product IDs
#     products = Product.objects.filter(asin__in=recommended_ids)
#     product_list = []
#     for product in products:
#         product_list.append({
#             'asin': product.asin,
#             'title': product.title,
#             'price': product.price,
#             'features': product.features,
#             'images': product.images,
#         })
#
#     return JsonResponse({'recommendations': product_list})
#
#
# class ProductViewSet(viewsets.ModelViewSet):
#     """
#     API endpoint that allows products to be viewed or edited.
#     """
#     queryset = Product.objects.all()
#     serializer_class = ProductSerializer
#
#
# class ReviewViewSet(viewsets.ModelViewSet):
#     """
#     API endpoint that allows reviews to be viewed or edited.
#     """
#     queryset = Review.objects.all()  # Or use Review.objects.all() if you want reviews endpoint
#     serializer_class = ReviewSerializer
