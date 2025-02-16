import datetime
from django.db import models


class Product(models.Model):
    # The unique product identifier (ASIN)
    asin = models.CharField(max_length=50, primary_key=True)
    # Title of the product
    title = models.CharField(max_length=255)
    # Price of the product (optional)
    price = models.FloatField(null=True, blank=True)
    # Features of the product stored as a JSON field (can store an array)
    features = models.JSONField(null=True, blank=True)
    # Parent ASIN for grouping similar products
    parent_asin = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.title


class Review(models.Model):
    user_id = models.CharField(max_length=50)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="reviews")
    rating = models.FloatField()
    title = models.CharField(max_length=255, default='Review Title')
    text = models.TextField(default='some text here')
    helpful_vote = models.IntegerField(default=0)
    verified_purchase = models.BooleanField(default=False)
    images = models.JSONField(null=True, blank=True)
    timestamp = models.DateTimeField(default=datetime.datetime.now)

    def save(self, *args, **kwargs):
        # Convert timestamp (milliseconds) to DateTime before saving
        if isinstance(self.timestamp, int):  # If it's in milliseconds, convert it
            self.timestamp = datetime.datetime.fromtimestamp(self.timestamp / 1000)
        elif isinstance(self.timestamp, str):  # If it's a string, try to parse it
            try:
                self.timestamp = datetime.datetime.fromisoformat(self.timestamp)
            except ValueError:
                self.timestamp = datetime.datetime.now()  # Use current time if invalid string
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Review by {self.user_id} on {self.product.title}"

