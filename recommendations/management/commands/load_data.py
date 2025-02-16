# load_data.py

import json
from django.core.management.base import BaseCommand
from recommendations.models import Product, Review
from django.db import transaction
from time import time
import os
import datetime

class Command(BaseCommand):
    help = 'Load product and review data from JSONL files into the database'

    def handle(self, *args, **kwargs):
        review_file = 'data/Gift_Cards.jsonl'  # Path to the review JSONL file
        batch_size = 1000  # Batch size for bulk inserts
        batch = []
        processed_lines = 0
        start_time = time()

        # Delete existing products and reviews
        Product.objects.all().delete()
        Review.objects.all().delete()

        # Limit the number of entries to process
        limit = 1000  # Limit to the first 100 products/reviews
        current_line = 0

        # Ensure the file exists
        if not os.path.exists(review_file):
            self.stdout.write(self.style.ERROR('Data file does not exist'))
            return

        total_lines = sum(1 for line in open(review_file))  # Calculate total number of lines

        with open(review_file, 'r') as fp:
            for line in fp:
                if current_line >= limit:
                    break  # Stop after processing the first 100 entries

                review_data = json.loads(line.strip())

                # Validate timestamp format
                if isinstance(review_data['timestamp'], str):
                    try:
                        timestamp = datetime.datetime.fromisoformat(review_data['timestamp'])
                    except ValueError:
                        self.stdout.write(self.style.ERROR(f"Invalid timestamp format: {review_data['timestamp']}"))
                        continue  # Skip this review if the timestamp is invalid
                else:
                    timestamp = datetime.datetime.fromtimestamp(review_data['timestamp'] / 1000)  # If it's in milliseconds

                # Check if product exists, create it if not
                product, created = Product.objects.get_or_create(
                    asin=review_data['asin'],
                    defaults={
                        'title': review_data['title'],
                        'price': review_data.get('price', None),
                        'features': review_data.get('features', None),
                        'parent_asin': review_data.get('parent_asin', None)
                    }
                )

                # Prepare review data and add to batch
                batch.append(
                    Review(
                        user_id=review_data['user_id'],
                        product=product,
                        rating=review_data['rating'],
                        title=review_data['title'],
                        text=review_data['text'],
                        helpful_vote=review_data['helpful_vote'],
                        verified_purchase=review_data['verified_purchase'],
                        images=review_data['images'],
                        timestamp=timestamp  # Set the validated timestamp
                    )
                )

                processed_lines += 1
                current_line += 1

                # Insert reviews in batches
                if len(batch) >= batch_size:
                    with transaction.atomic():
                        Review.objects.bulk_create(batch)
                    batch = []  # Reset batch after insert

                # Provide progress feedback every 1000 lines
                if processed_lines % 1000 == 0:
                    elapsed_time = time() - start_time
                    self.stdout.write(self.style.SUCCESS(
                        f'{processed_lines} lines processed in {elapsed_time:.2f} seconds.'
                    ))

        # Insert any remaining reviews in batch
        if batch:
            with transaction.atomic():
                Review.objects.bulk_create(batch)
            self.stdout.write(self.style.SUCCESS(f'Finished loading {processed_lines} reviews.'))
