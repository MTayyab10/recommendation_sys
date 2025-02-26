import csv
import time
from django.core.management.base import BaseCommand
from django.db import transaction
from recommendations.models import Movie, Rating


class Command(BaseCommand):
    help = "Load MovieLens data (movies.csv and ratings.csv) into the database with limit and batch processing"

    def handle(self, *args, **options):
        # small development dataset
        # movies_file = 'data/ml-latest-small/movies.csv'
        # ratings_file = 'data/ml-latest-small/ratings.csv'

        # large dataset
        movies_file = 'data/ml-32m/movies.csv'
        ratings_file = 'data/ml-32m/ratings.csv'

        # Control parameters
        movie_limit = 100000  # Limit number of movies to process
        rating_limit = 100000  # Limit number of ratings to process
        batch_size = 1000  # Number of ratings to bulk insert at once

        # Delete existing data (optional)
        Movie.objects.all().delete()
        Rating.objects.all().delete()
        self.stdout.write(self.style.WARNING('Deleted existing movie and rating data...'))

        # Load movies
        self.stdout.write(self.style.SUCCESS(f"Loading up to {movie_limit} movies from {movies_file}..."))
        with open(movies_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            movie_count = 0
            for row in reader:
                if movie_count >= movie_limit:
                    break
                Movie.objects.update_or_create(
                    movieId=int(row['movieId']),
                    defaults={
                        'title': row['title'],
                        'genres': row['genres']
                    }
                )
                movie_count += 1
        self.stdout.write(self.style.SUCCESS(f"Loaded {movie_count} movies successfully."))

        # Load ratings
        self.stdout.write(self.style.SUCCESS(f"Loading up to {rating_limit} ratings from {ratings_file}..."))
        rating_objects = []
        rating_count = 0
        start_time = time.time()

        with open(ratings_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if rating_count >= rating_limit:
                    break
                try:
                    movie = Movie.objects.get(movieId=int(row['movieId']))
                except Movie.DoesNotExist:
                    continue  # Skip ratings for movies not in the database

                rating_objects.append(Rating(
                    user_id=int(row['userId']),
                    movie=movie,
                    rating=float(row['rating']),
                    timestamp=int(row['timestamp'])
                ))
                rating_count += 1

                # Bulk insert in batches
                if len(rating_objects) >= batch_size:
                    with transaction.atomic():
                        Rating.objects.bulk_create(rating_objects)
                    rating_objects = []
                    self.stdout.write(self.style.SUCCESS(f"Processed {rating_count} ratings so far..."))

        # Insert remaining ratings
        if rating_objects:
            with transaction.atomic():
                Rating.objects.bulk_create(rating_objects)

        elapsed_time = time.time() - start_time
        self.stdout.write(self.style.SUCCESS(
            f"Finished loading {rating_count} ratings in {elapsed_time:.2f} seconds."
        ))