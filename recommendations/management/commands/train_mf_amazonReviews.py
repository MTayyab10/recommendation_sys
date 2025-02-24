"""
Matrix factorization is indeed a very effective approach for handling large-scale datasets.
By decomposing the large and sparse user–item interaction matrix into lower-dimensional latent factors, you not only reduce the computational complexity
but also uncover hidden patterns in user preferences and item characteristics.

Why Matrix Factorization?
Scalability: Instead of computing pairwise similarities over a huge, dense matrix, you work with much smaller latent factor matrices.
Handling Sparsity: It works well with sparse data—a common characteristic in recommender systems.
Latent Factors: These factors capture underlying patterns that might not be apparent from raw ratings alone.
Integration with Hybrid Models: You can later combine these latent factors with content-based signals (like product meta data) for a more robust system.
Implementing Matrix Factorization with SVD (Using Surprise Library)
Below is a sample implementation using the Surprise library. This code demonstrates how to train an SVD model on your review data, evaluate it,
 and then save it for later use in your API.
"""

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from django.core.management.base import BaseCommand
from recommendations.models import Review
import pickle


class Command(BaseCommand):
    help = 'Train a matrix factorization model using SVD on review data and save the model to disk'

    def handle(self, *args, **options):
        self.stdout.write("Fetching review data from the database...")
        # Fetch review data from the database
        reviews_qs = Review.objects.all().values('user_id', 'product__asin', 'rating')
        df = pd.DataFrame(list(reviews_qs))

        # Rename the column to match Surprise's expected format
        df.rename(columns={'product__asin': 'asin'}, inplace=True)

        # Ensure that there is data to train on
        if df.empty:
            self.stdout.write(self.style.ERROR("No review data found."))
            return

        self.stdout.write("Preparing data for training...")
        # Set up the reader with the known rating scale
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'asin', 'rating']], reader)

        # Split the dataset into training and testing sets
        trainset, testset = train_test_split(data, test_size=0.2)

        self.stdout.write("Training SVD model...")
        # Initialize and train the SVD model
        algo = SVD()
        algo.fit(trainset)

        # Evaluate the model using RMSE
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions)
        self.stdout.write(self.style.SUCCESS(f"Model trained with RMSE: {rmse:.4f}"))

        # Save the trained model to a pickle file
        model_path = 'svd_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(algo, f)
        self.stdout.write(self.style.SUCCESS(f"Trained model saved to {model_path}"))
