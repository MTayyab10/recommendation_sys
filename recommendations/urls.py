from django.urls import path, include
from rest_framework import routers
from .views import ProductViewSet, ReviewViewSet, recommend_products

# Create a router and register your viewsets
router = routers.DefaultRouter()
router.register(r'products', ProductViewSet)
router.register(r'reviews', ReviewViewSet)

urlpatterns = [
    # Your custom view endpoint for recommendations
    path('recommendations/<str:user_id>/', recommend_products, name='recommendations'),

    # Include the router URLs under an API prefix
    path('api/', include(router.urls)),
]
