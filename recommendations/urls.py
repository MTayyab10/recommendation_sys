from django.urls import path
from .views import recommend_products

urlpatterns = [
    path('recommendations/<str:user_id>/', recommend_products, name='recommendations'),
]
