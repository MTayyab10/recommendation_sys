from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Product, Review

class ProductAdmin(admin.ModelAdmin):
    list_display = ('asin', 'title', 'price', 'features', 'parent_asin')

class ReviewAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'product', 'rating', 'title', 'text', 'helpful_vote', 'verified_purchase', 'images', 'timestamp')

admin.site.register(Product, ProductAdmin)
admin.site.register(Review, ReviewAdmin)