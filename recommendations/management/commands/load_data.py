import json
from django.core.management.base import BaseCommand
from recommendations.models import Product, Review
from django.db import transaction
from time import time
import os
import datetime


class Command(BaseCommand):
    help = 'Load product meta data and review data from JSONL files into the database'

    def handle(self, *args, **kwargs):
        # File paths (adjust as needed)
        meta_file = 'data/meta_All_Beauty.jsonl'
        review_file = 'data/All_Beauty.jsonl'

        review_batch_size = 1000  # Number of reviews to bulk insert at once
        review_batch = []
        processed_reviews = 0
        review_start_time = time()

        # Delete existing data (optional; adjust if you want to merge instead)
        Product.objects.all().delete()
        Review.objects.all().delete()

        # ----- Load Meta Data -----
        if not os.path.exists(meta_file):
            self.stdout.write(self.style.ERROR(f'Meta file {meta_file} does not exist'))
            return

        self.stdout.write(self.style.SUCCESS(f'Start loading meta data from {meta_file}...'))
        meta_lines = 0
        product_limit = 100000  # Adjust the limit as needed (currently processing first 100 products)
        current_product_line = 0

        with open(meta_file, 'r') as fp:
            for line in fp:
                if current_product_line >= product_limit:
                    break

                meta_lines += 1
                try:
                    meta_data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    self.stdout.write(self.style.ERROR(f'Error decoding meta data JSON: {e}'))
                    continue

                # Use parent_asin if available; otherwise, fall back to asin
                product_asin = meta_data.get('parent_asin') or meta_data.get('asin')

                # Create or update the Product with rich meta fields
                Product.objects.update_or_create(
                    asin=product_asin,
                    defaults={
                        'title': meta_data.get('title', '')[:255],
                        'price': float(meta_data.get('price')) if meta_data.get('price') and meta_data.get(
                            'price') != 'None' else None,
                        'features': meta_data.get('features'),
                        'description': meta_data.get('description'),
                        'images': meta_data.get('images'),
                        'parent_asin': meta_data.get('parent_asin'),
                    }
                )
                current_product_line += 1

        self.stdout.write(self.style.SUCCESS(f'Finished loading meta data, processed {meta_lines} lines.'))

        # ----- Load Review Data -----
        if not os.path.exists(review_file):
            self.stdout.write(self.style.ERROR(f'Review file {review_file} does not exist'))
            return

        self.stdout.write(self.style.SUCCESS(f'Start loading review data from {review_file}...'))
        limit = 100000  # Adjust the limit as needed (currently processing first 100 reviews)
        current_review_line = 0

        with open(review_file, 'r') as fp:
            for line in fp:
                if current_review_line >= limit:
                    break

                try:
                    review_data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    self.stdout.write(self.style.ERROR(f'Error decoding review JSON: {e}'))
                    continue

                # Process timestamp: if string, try ISO; else assume milliseconds
                ts = review_data.get('timestamp')
                if isinstance(ts, str):
                    try:
                        timestamp = datetime.datetime.fromisoformat(ts)
                    except ValueError:
                        self.stdout.write(self.style.ERROR(f"Invalid timestamp format: {ts}"))
                        timestamp = datetime.datetime.now()
                else:
                    timestamp = datetime.datetime.fromtimestamp(ts / 1000)

                # Lookup product: try using parent_asin first, then asin
                asin = review_data.get('asin')
                parent_asin = review_data.get('parent_asin')
                product = None
                try:
                    product = Product.objects.get(asin=parent_asin)
                except Product.DoesNotExist:
                    try:
                        product = Product.objects.get(asin=asin)
                    except Product.DoesNotExist:
                        # If not found, create a fallback product (will be less rich)
                        product, created = Product.objects.get_or_create(
                            asin=asin,
                            defaults={
                                'title': review_data.get('title', ''),
                                'price': None,
                                'features': None,
                                'parent_asin': parent_asin,
                            }
                        )

                # Prepare review instance
                review = Review(
                    user_id=review_data.get('user_id'),
                    product=product,
                    rating=review_data.get('rating'),
                    title=review_data.get('title', 'Review Title'),
                    text=review_data.get('text', ''),
                    helpful_vote=review_data.get('helpful_vote', 0),
                    verified_purchase=review_data.get('verified_purchase', False),
                    images=review_data.get('images'),
                    timestamp=timestamp
                )
                review_batch.append(review)
                processed_reviews += 1
                current_review_line += 1

                # Bulk insert review batch if batch size is reached
                if len(review_batch) >= review_batch_size:
                    with transaction.atomic():
                        Review.objects.bulk_create(review_batch)
                    review_batch = []
                    self.stdout.write(self.style.SUCCESS(
                        f'{processed_reviews} reviews processed so far...'
                    ))

        # Insert any remaining reviews in batch
        if review_batch:
            with transaction.atomic():
                Review.objects.bulk_create(review_batch)
        elapsed_time = time() - review_start_time
        self.stdout.write(self.style.SUCCESS(
            f'Finished loading {processed_reviews} reviews in {elapsed_time:.2f} seconds.'
        ))
