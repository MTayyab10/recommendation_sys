""" Data Fetching and Sampling:
The command fetches review data from your database and converts it into a Pandas DataFrame. To speed up the grid search,
it randomly samples a fraction (10% in this example) of the data.

Parameter Grid:
We define a grid over the number of latent factors (n_factors), learning rates (lr_all), and regularization parameters (reg_all).

GridSearchCV:
Surprise’s GridSearchCV runs 3-fold cross-validation over the sample, evaluates RMSE for each combination, and returns the best configuration.

Output:
The best parameters and corresponding RMSE are printed and saved to a file (best_svd_params.txt).
"""
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
from django.core.management.base import BaseCommand
from recommendations.models import Review


class Command(BaseCommand):
    help = 'Tune hyperparameters for the SVD model using GridSearchCV on a sample of review data.'

    def handle(self, *args, **kwargs):
        self.stdout.write("Fetching review data from the database for tuning...")
        # Fetch review data from the database
        reviews_qs = Review.objects.all().values('user_id', 'product__asin', 'rating')
        df = pd.DataFrame(list(reviews_qs))

        if df.empty:
            self.stdout.write(self.style.ERROR("No review data found."))
            return

        # Rename the column for Surprise
        df.rename(columns={'product__asin': 'asin'}, inplace=True)
        df['rating'] = df['rating'].astype(float)

        # Use a smaller sample if necessary to speed up tuning
        sample_fraction = 0.1  # For example, use 10% of the data
        df_sample = df.sample(frac=sample_fraction, random_state=42)

        self.stdout.write("Preparing data for grid search...")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_sample[['user_id', 'asin', 'rating']], reader)

        # Define parameter grid for SVD
        param_grid = {
            'n_factors': [20, 50, 100],
            'lr_all': [0.002, 0.005, 0.01],
            'reg_all': [0.02, 0.05, 0.1]
        }

        self.stdout.write("Starting grid search for hyperparameter tuning...")
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, joblib_verbose=1)
        gs.fit(data)

        best_params = gs.best_params['rmse']
        best_score = gs.best_score['rmse']
        self.stdout.write(self.style.SUCCESS(f"Best parameters: {best_params}"))
        self.stdout.write(self.style.SUCCESS(f"Best RMSE: {best_score:.4f}"))

        # Optionally, you could save the best parameters to a file for later use
        with open('best_svd_params.txt', 'w') as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best RMSE: {best_score:.4f}\n")

