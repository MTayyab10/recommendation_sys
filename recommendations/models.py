import datetime
from django.db import models

class Product(models.Model):
    # Unique identifier for the product (from reviews and meta)
    asin = models.CharField(max_length=50, primary_key=True)

    # Product metadata fields (from meta file)
    title = models.CharField(max_length=255, null=True, blank=True)  # From meta file, more reliable than review title
    main_category = models.CharField(max_length=255, null=True, blank=True)
    average_rating = models.FloatField(null=True, blank=True)
    rating_number = models.IntegerField(null=True, blank=True)
    features = models.JSONField(null=True, blank=True)
    description = models.JSONField(null=True,
                                   blank=True)  # Stored as JSON (list) or you can convert to TextField if needed
    price = models.FloatField(null=True, blank=True)
    images = models.JSONField(null=True, blank=True)  # Detailed images from meta file
    videos = models.JSONField(null=True, blank=True)
    store = models.CharField(max_length=255, null=True, blank=True)
    categories = models.JSONField(null=True, blank=True)
    details = models.JSONField(null=True, blank=True)

    # Some items may have a parent asin (for variants)
    parent_asin = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        # Prefer meta title if available; otherwise fallback to asin
        return self.title if self.title else self.asin


class Review(models.Model):
    user_id = models.CharField(max_length=50, db_index=True)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="reviews")
    rating = models.FloatField()
    # The review title is specific to the review; note that the product title comes from meta
    title = models.CharField(max_length=255, default='Review Title')
    text = models.TextField(default='Some review text here')
    helpful_vote = models.IntegerField(default=0)
    verified_purchase = models.BooleanField(default=False)
    # Review-specific images (often empty) – meta images will be in Product.images
    images = models.JSONField(null=True, blank=True)
    # Timestamp is stored as a DateTimeField; we convert from milliseconds if necessary
    timestamp = models.DateTimeField(default=datetime.datetime.now)

    def save(self, *args, **kwargs):
        # If timestamp is an int (milliseconds), convert it to a datetime object.
        if isinstance(self.timestamp, int):
            self.timestamp = datetime.datetime.fromtimestamp(self.timestamp / 1000)
        elif isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.datetime.fromisoformat(self.timestamp)
            except ValueError:
                self.timestamp = datetime.datetime.now()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Review by {self.user_id} on {self.product.title}"
