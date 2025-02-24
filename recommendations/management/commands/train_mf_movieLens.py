import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from django.core.management.base import BaseCommand
from recommendations.models import Rating  # Using the new Rating model
import pickle

class Command(BaseCommand):
    help = 'Train a matrix factorization model using SVD on MovieLens rating data and save the model to disk'

    def handle(self, *args, **options):
        self.stdout.write("Fetching rating data from the database...")
        # Fetch rating data from the database; note we use 'movie__movieId' for the MovieLens movie identifier
        ratings_qs = Rating.objects.all().values('user_id', 'movie__movieId', 'rating')
        df = pd.DataFrame(list(ratings_qs))

        # Rename the column to match Surprise's expected format
        df.rename(columns={'movie__movieId': 'movieId'}, inplace=True)

        # Ensure that there is data to train on
        if df.empty:
            self.stdout.write(self.style.ERROR("No rating data found."))
            return

        self.stdout.write("Preparing data for training...")
        # MovieLens ratings typically range from 0.5 to 5.0
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(df[['user_id', 'movieId', 'rating']], reader)

        # Split the dataset into training and testing sets
        trainset, testset = train_test_split(data, test_size=0.2)

        self.stdout.write("Training SVD model...")
        algo = SVD()
        algo.fit(trainset)

        # Evaluate the model using RMSE on the test set
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions)
        self.stdout.write(self.style.SUCCESS(f"Model trained with RMSE: {rmse:.4f}"))

        # Save the trained model to a pickle file
        model_path = 'svd_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(algo, f)
        self.stdout.write(self.style.SUCCESS(f"Trained model saved to {model_path}"))
