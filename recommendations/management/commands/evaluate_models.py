""" Explanation  Data Preparation:

The command fetches review data from the database via the Django ORM.
The data is converted to a Pandas DataFrame and split into training (80%) and test (20%) sets.
A ground truth mapping (per user) is built from the test set (i.e., the items each user interacted with in the test set).
Memory-Based Evaluation:

We build an interaction matrix from the training data using a memory-based approach.
We compute cosine similarities and then generate recommendations for users present in the test set.
We compute Precision@10 and Recall@10 for each user, and then average these metrics.
Matrix Factorization Evaluation:

Using the Surprise library, we train an SVD model on the training set.
The model is evaluated on the test set using RMSE.
For each user, we predict ratings for all products, sort by predicted rating, and extract the top 10 recommendations.
We compute Precision@10 and Recall@10 for these recommendations.
Diversity and Novelty (Optional):

A dummy item feature matrix is created (assigning random vectors to each item) to compute diversity. In practice, you’d use actual item features (e.g., textual embeddings).
Novelty is calculated based on the inverse popularity (number of reviews) of items.
"""
import random
import numpy as np
import pickle
import datetime
from django.core.management.base import BaseCommand
from django.db.models import Count
from recommendations.models import Review, Product
from recommendations.memory_based import create_interaction_matrix, calculate_similarity, recommend_for_user
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise import SVD
import pandas as pd


def precision_recall_at_k(recommended, ground_truth, k):
    """
    Compute precision and recall at k for a single user.
    recommended: list of recommended item ids
    ground_truth: set of item ids in the test set for the user
    """
    recommended = recommended[:k]
    relevant = set(recommended) & ground_truth
    precision = len(relevant) / k if k > 0 else 0
    recall = len(relevant) / len(ground_truth) if ground_truth else 0
    return precision, recall


def compute_diversity(recommended, item_feature_matrix):
    """
    Compute diversity as the average pairwise dissimilarity.
    item_feature_matrix: a dictionary mapping item id to its feature vector (or any numeric representation)
    For simplicity, we use cosine similarity between feature vectors.
    If features are missing, we assume maximum diversity (1.0).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    if len(recommended) < 2:
        return 0
    vectors = []
    for item in recommended:
        vec = item_feature_matrix.get(item)
        if vec is None:
            # If no feature vector, assume a default (zero vector)
            vec = np.zeros(10)  # assume dimension 10; adjust as needed
        vectors.append(vec)
    vectors = np.array(vectors)
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(vectors)
    # We want diversity = 1 - similarity; compute average pairwise (excluding self-similarity)
    n = len(recommended)
    diversity = (np.sum(sim_matrix) - n) / (n * (n - 1))  # average similarity
    return 1 - diversity


def compute_novelty(recommended, item_popularity):
    """
    Compute novelty as the average inverse popularity of the recommended items.
    item_popularity: dictionary mapping item id to popularity (e.g., number of reviews)
    """
    novelty_scores = []
    for item in recommended:
        pop = item_popularity.get(item, 1)  # avoid division by zero
        novelty_scores.append(1 / pop)
    return np.mean(novelty_scores) if novelty_scores else 0


class Command(BaseCommand):
    help = 'Evaluate both memory-based and matrix factorization recommendation models'

    def handle(self, *args, **options):
        # -------------------------
        # Step 1: Prepare Data for Evaluation
        # -------------------------
        self.stdout.write("Fetching review data from the database...")
        reviews_qs = Review.objects.all().values('user_id', 'product__asin', 'rating')
        df = pd.DataFrame(list(reviews_qs))
        if df.empty:
            self.stdout.write(self.style.ERROR("No review data found."))
            return

        df.rename(columns={'product__asin': 'asin'}, inplace=True)
        # Convert rating to float if necessary
        df['rating'] = df['rating'].astype(float)

        # For evaluation purposes, perform a simple train-test split (80% train, 20% test)
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        # Build a ground truth mapping per user from test set
        ground_truth = {}
        for _, row in test_df.iterrows():
            uid = row['user_id']
            asin = row['asin']
            if uid not in ground_truth:
                ground_truth[uid] = set()
            ground_truth[uid].add(asin)

        # -------------------------
        # Step 2: Memory-Based Evaluation
        # -------------------------
        self.stdout.write("Evaluating Memory-Based Collaborative Filtering...")
        # Build the interaction matrix using training data only:
        # We simulate this by filtering reviews to only those in train_df.
        # For simplicity, we override Review.objects.all() by using our df
        memory_matrix = {}
        all_items = set()
        for _, row in train_df.iterrows():
            uid = row['user_id']
            asin = row['asin']
            if uid not in memory_matrix:
                memory_matrix[uid] = {}
            memory_matrix[uid][asin] = row['rating']
            all_items.add(asin)
        # Fill missing values with 0
        for uid in memory_matrix:
            for asin in all_items:
                if asin not in memory_matrix[uid]:
                    memory_matrix[uid][asin] = 0

        # Compute similarity matrix for memory-based approach
        similarity_matrix, user_ids = calculate_similarity(memory_matrix, list(all_items))

        # Evaluate for each user in test set that exists in training data:
        precisions_memory = []
        recalls_memory = []
        for uid in ground_truth:
            if uid not in memory_matrix:
                continue
            recs = recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10)
            precision, recall = precision_recall_at_k(recs, ground_truth[uid], k=10)
            precisions_memory.append(precision)
            recalls_memory.append(recall)
        avg_precision_memory = np.mean(precisions_memory) if precisions_memory else 0
        avg_recall_memory = np.mean(recalls_memory) if recalls_memory else 0

        self.stdout.write(self.style.SUCCESS(
            f"Memory-Based CF - Precision@10: {avg_precision_memory:.4f}, Recall@10: {avg_recall_memory:.4f}"
        ))

        # -------------------------
        # Step 3: Matrix Factorization Evaluation
        # -------------------------
        self.stdout.write("Evaluating Matrix Factorization (SVD) approach...")
        # Use Surprise library for SVD evaluation on training data
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_df[['user_id', 'asin', 'rating']], reader)
        trainset = data.build_full_trainset()
        # Train SVD model on training set
        mf_model = SVD()
        mf_model.fit(trainset)

        # Evaluate using RMSE on the test set (using Surprise's data format)
        test_data = Dataset.load_from_df(test_df[['user_id', 'asin', 'rating']], reader)
        testset = test_data.build_full_trainset().build_testset()
        predictions = mf_model.test(testset)
        rmse_value = accuracy.rmse(predictions, verbose=False)
        self.stdout.write(self.style.SUCCESS(f"SVD Model RMSE on test set: {rmse_value:.4f}"))

        # Evaluate top-10 recommendations for each user using the SVD model
        precisions_mf = []
        recalls_mf = []
        # Build a popularity mapping for novelty: count reviews per item in training data
        item_popularity = train_df.groupby('asin').size().to_dict()

        # For each user in ground_truth
        for uid in ground_truth:
            # For every product in training data, predict rating for this user
            all_products = list(train_df['asin'].unique())
            user_preds = []
            for asin in all_products:
                pred = mf_model.predict(uid, asin)
                user_preds.append((asin, pred.est))
            user_preds.sort(key=lambda x: x[1], reverse=True)
            recs = [asin for asin, _ in user_preds[:10]]
            precision, recall = precision_recall_at_k(recs, ground_truth[uid], k=10)
            precisions_mf.append(precision)
            recalls_mf.append(recall)
        avg_precision_mf = np.mean(precisions_mf) if precisions_mf else 0
        avg_recall_mf = np.mean(recalls_mf) if recalls_mf else 0

        self.stdout.write(self.style.SUCCESS(
            f"SVD Model - Precision@10: {avg_precision_mf:.4f}, Recall@10: {avg_recall_mf:.4f}"
        ))

        # -------------------------
        # Step 4: Diversity and Novelty (Optional)
        # -------------------------
        # For demonstration, assume a dummy item feature matrix: here, we assign a random vector to each item.
        # In practice, you might use item embeddings from textual features or other metadata.
        all_products = list(all_items)
        item_feature_matrix = {asin: np.random.rand(10) for asin in all_products}
        diversity_memory = np.mean([compute_diversity(
            recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10),
            item_feature_matrix)
                                    for uid in ground_truth if uid in memory_matrix])
        novelty_memory = np.mean(
            [compute_novelty(recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10),
                             item_popularity)
             for uid in ground_truth if uid in memory_matrix])

        self.stdout.write(self.style.SUCCESS(
            f"Memory-Based CF - Average Diversity: {diversity_memory:.4f}, Average Novelty: {novelty_memory:.4f}"
        ))

        # Similar diversity/novelty could be computed for SVD recommendations using predicted top-N lists.
        # (You would need to generate item_feature_matrix and item_popularity similarly for SVD recommendations.)

        self.stdout.write(self.style.SUCCESS("Evaluation complete."))
