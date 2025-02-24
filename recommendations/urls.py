from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ProductViewSet, TaskStatusView

router = DefaultRouter()
router.register(r'products', ProductViewSet, basename='product')
urlpatterns = [
    path('', include(router.urls)),

    # sample req: http://127.0.0.1:8000/task/1497289d-079d-4f73-83b7-7094e3ebf355
    path('task-status/<str:task_id>/', TaskStatusView.as_view(), name='task'),

]
