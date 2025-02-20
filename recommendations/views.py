from django.core.cache import cache
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Product
from .serializers import ProductSerializer
from .methods import memory_based_recommendation, matrix_factorization_recommendation
from recommendations.methods import load_svd_model
from recommendations.hybrid_model import create_interaction_matrix, calculate_similarity, hybrid_recommendation
from django.db.models import Count

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
        Caches recommendations for a given user for 5 minutes.
        """
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id parameter is required'}, status=400)

        cache_key = f"recs_memory_{user_id}"
        recs = cache.get(cache_key)
        if recs is None:
            recs = memory_based_recommendation(user_id)  # This function computes recommendations
            cache.set(cache_key, recs, timeout=500)  # Cache for 300 seconds (5 minutes)

        products = Product.objects.filter(asin__in=recs)
        serializer = self.get_serializer(products, many=True)
        return Response({'recommendations': serializer.data})

    @action(detail=False, methods=['get'], url_path='recommendations/mf')
    def matrix_factorization(self, request):
        """
        Retrieve product recommendations using matrix factorization.
        Caches recommendations for a given user for 5 minutes.
        """
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id parameter is required'}, status=400)

        cache_key = f"recs_mf_{user_id}"
        recs = cache.get(cache_key)
        if recs is None:
            recs = matrix_factorization_recommendation(user_id)  # This function computes recommendations
            cache.set(cache_key, recs, timeout=300)

        products = Product.objects.filter(asin__in=recs)
        serializer = self.get_serializer(products, many=True)
        return Response({'recommendations': serializer.data})

    @action(detail=False, methods=['get'], url_path='recommendations/hybrid')
    def hybrid(self, request):
        """
        Retrieve product recommendations using the hybrid approach.
        """
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id parameter is required'}, status=400)

        # Build the memory-based structures
        memory_matrix, all_items = create_interaction_matrix()
        similarity_matrix, user_ids = calculate_similarity(memory_matrix, all_items)

        # Define candidate items - for example, top 100 popular items
        # Get the top 100 popular items by counting occurrences of each asin
        popular_items_qs = Product.objects.values('asin').annotate(count=Count('asin')).order_by('-count')
        candidate_items = [item['asin'] for item in popular_items_qs[:100]]

        # Load the SVD model (trained previously)
        mf_model = load_svd_model()

        # Generate hybrid recommendations
        recommended_ids = hybrid_recommendation(user_id, memory_matrix, similarity_matrix, user_ids, mf_model,
                                                candidate_items)
        products = Product.objects.filter(asin__in=recommended_ids)
        serializer = self.get_serializer(products, many=True)
        return Response({'recommendations': serializer.data})

