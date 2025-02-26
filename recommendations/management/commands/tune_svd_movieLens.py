import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
from django.core.management.base import BaseCommand
from recommendations.models import Rating  # Using the new Rating model

class Command(BaseCommand):
    help = 'Tune hyperparameters for the SVD model using GridSearchCV on a sample of MovieLens rating data.'

    def handle(self, *args, **kwargs):
        self.stdout.write("Fetching rating data from the database for tuning...")
        # Fetch rating data from the database
        ratings_qs = Rating.objects.all().values('user_id', 'movie__movieId', 'rating')
        df = pd.DataFrame(list(ratings_qs))

        if df.empty:
            self.stdout.write(self.style.ERROR("No rating data found."))
            return

        # Rename the column for Surprise's format
        df.rename(columns={'movie__movieId': 'movieId'}, inplace=True)
        df['rating'] = df['rating'].astype(float)

        # Use a smaller sample to speed up tuning
        sample_fraction = 0.1  # Use 10% of the data for tuning
        df_sample = df.sample(frac=sample_fraction, random_state=42)

        self.stdout.write("Preparing data for grid search...")
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(df_sample[['user_id', 'movieId', 'rating']], reader)

        # Define parameter grid for SVD
        param_grid = {
            'n_factors': [50, 100, 300],
            'lr_all': [0.005, 0.01, 0.02],
            'reg_all': [0.02, 0.05, 0.1]
        }

        self.stdout.write("Starting grid search for hyperparameter tuning...")
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, joblib_verbose=1)
        gs.fit(data)

        best_params = gs.best_params['rmse']
        best_score = gs.best_score['rmse']
        self.stdout.write(self.style.SUCCESS(f"Best parameters: {best_params}"))
        self.stdout.write(self.style.SUCCESS(f"Best RMSE: {best_score:.4f}"))

        # Save the best parameters to a file
        with open('best_svd_params.txt', 'w') as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best RMSE: {best_score:.4f}\n")
