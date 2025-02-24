# recommendations/management/commands/load_movielens.py

import csv
from django.core.management.base import BaseCommand
from recommendations.models import Movie, Rating

class Command(BaseCommand):
    help = "Load MovieLens data (movies.csv and ratings.csv) into the database"

    def handle(self, *args, **options):
        # Adjust the file paths as needed
        movies_file = 'data/ml-latest-small/movies.csv'
        ratings_file = 'data/ml-latest-small/ratings.csv'

        # Delete existing data (optional; adjust if you want to merge instead)
        Movie.objects.all().delete()
        Rating.objects.all().delete()
        print('Deleted existing data...')

        # Load movies
        self.stdout.write("Loading movies...")
        with open(movies_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie, created = Movie.objects.update_or_create(
                    movieId=int(row['movieId']),
                    defaults={
                        'title': row['title'],
                        'genres': row['genres']
                    }
                )
        self.stdout.write(self.style.SUCCESS("Movies loaded successfully."))

        # Load ratings
        self.stdout.write("Loading ratings...")
        rating_objects = []
        with open(ratings_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
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
        Rating.objects.bulk_create(rating_objects, batch_size=1000)
        self.stdout.write(self.style.SUCCESS("Ratings loaded successfully."))
