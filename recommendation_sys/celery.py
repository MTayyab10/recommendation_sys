# celery.py
import os
from celery import Celery

# Set default Django settings module for 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendation_sys.settings')

app = Celery('recommendation_sys')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
