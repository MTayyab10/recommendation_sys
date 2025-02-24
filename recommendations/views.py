from django.core.cache import cache
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Product
from .serializers import ProductSerializer
from .methods import memory_based_recommendation, matrix_factorization_recommendation, load_svd_model
from recommendations.hybrid_amazonReviews import create_interaction_matrix, calculate_similarity, hybrid_recommendation
from django.db.models import Count
from recommendations.tasks import async_memory_based_recommendations, async_mf_recommendations, \
    async_hybrid_recommendations, async_content_based_recommendations
from celery.result import AsyncResult
from rest_framework.response import Response
from rest_framework.views import APIView

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
            task = async_memory_based_recommendations(user_id)  # This function computes recommendations
            return Response({'task_id': task.id, 'status': 'Processing'})

        products = Product.objects.filter(asin__in=recs)
        serializer = self.get_serializer(products, many=True)
        return Response({'recommendations': serializer.data})

    @action(detail=False, methods=['get'], url_path='recommendations/content')
    def content_based(self, request):
        """
        Retrieve product recommendations using content-based filtering.
        This method recommends products similar to those the user has previously interacted with.
        """
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id parameter is required'}, status=400)
        cache_key = f"recs_content_{user_id}"
        recs = cache.get(cache_key)
        if recs is None:
            task = async_content_based_recommendations(user_id)  # This function computes recommendations
            return Response({'task_id': task.id, 'status': 'Processing'})

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
            # Trigger asynchronous task for MF-based recommendations
            task = async_mf_recommendations.delay(user_id)
            return Response({'task_id': task.id, 'status': 'Processing'})

        products = Product.objects.filter(asin__in=recs)
        serializer = self.get_serializer(products, many=True)
        return Response({'recommendations': serializer.data})


    @action(detail=False, methods=['get'], url_path='recommendations/hybrid')
    def hybrid(self, request):
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id parameter is required'}, status=400)

        # Check if result is cached
        cache_key = f"hybrid_recs_{user_id}"
        cached_result = cache.get(cache_key)
        if cached_result:
            products = Product.objects.filter(asin__in=cached_result)
            serializer = self.get_serializer(products, many=True)
            return Response({'recommendations': serializer.data})

        # Trigger the asynchronous task
        task = async_hybrid_recommendations.delay(user_id)
        return Response({'task_id': task.id, 'status': 'Processing'})



class TaskStatusView(APIView):
    """
    API endpoint to retrieve the status of a Celery task.
    """
    def get(self, request, task_id):
        # Retrieve the task result using the task_id
        result = AsyncResult(task_id)
        # Prepare the response data
        data = {
            'task_id': task_id,
            'status': result.status,
            'result': result.result if result.ready() else None
        }
        # If the task is complete and successful, fetch detailed product info
        if result.ready() and result.status == 'SUCCESS':
            recs = result.result
            products = Product.objects.filter(asin__in=recs)
            serializer = ProductSerializer(products, many=True)
            data['recommendations'] = serializer.data
        return Response(data)


