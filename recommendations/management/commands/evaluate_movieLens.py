"""
Evaluation Command for Recommendation Models (Scaled for MovieLens Dataset)

Steps:
1. Data Preparation:
   - Fetch rating data from the database via Django ORM.
   - Convert the data to a Pandas DataFrame and sample a fraction (e.g., 10%).
   - Split the data using a leave-one-out strategy per user (each user’s most recent rating is held out as test).
   - Build a ground truth mapping per user from the test set (considering ratings >= 4 as relevant).

2. Memory-Based Evaluation:
   - Build a user–item interaction matrix from the training data.
   - Compute cosine similarities and generate top-10 recommendations for a sampled subset of users.
   - Compute Precision@10, Recall@10, Hit Rate, and NDCG@10 using a candidate set (top 200 popular movies).

3. Matrix Factorization Evaluation (SVD):
   - Train an SVD model on the training data using the Surprise library.
   - Evaluate the model using RMSE on the test set.
   - For each sampled user, predict ratings over a candidate set, extract the top-10 recommendations, and compute evaluation metrics.

4. Content-Based Evaluation:
   - Generate recommendations using TF-IDF based content filtering.
   - Compute top-N metrics (Precision@10, Recall@10, Hit Rate, NDCG@10).

5. Hybrid Model Evaluation:
   - Combine memory-based and SVD-based scores using dynamic weighting.
   - Compute the same ranking metrics.

6. Optional: Diversity and Novelty:
   - Compute diversity using a dummy item feature matrix.
   - Compute novelty based on item popularity.
"""

import random
import numpy as np
import pandas as pd
import datetime
import math

from django.core.management.base import BaseCommand
from django.db.models import Count
from recommendations.models import Rating, Movie
from recommendations.memory_movieLens import create_interaction_matrix, calculate_similarity, recommend_for_user
from recommendations.hybrid_movieLens import hybrid_recommendation
from recommendations.content_movieLens import content_based_recommendation
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from joblib import Parallel, delayed

# ---------------------- Evaluation Metrics ----------------------
def precision_recall_at_k(recommended, ground_truth, k):
    recommended = recommended[:k]
    relevant = set(recommended) & ground_truth
    precision = len(relevant) / k if k > 0 else 0
    recall = len(relevant) / len(ground_truth) if ground_truth else 0
    return precision, recall

def hit_rate(recommended, ground_truth):
    return 1 if set(recommended) & ground_truth else 0

def ndcg_at_k(recommended, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    ideal_rels = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def compute_diversity(recommended, item_feature_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    if len(recommended) < 2:
        return 0
    vectors = []
    for item in recommended:
        vec = item_feature_matrix.get(item)
        if vec is None:
            vec = np.zeros(10)
        vectors.append(vec)
    vectors = np.array(vectors)
    sim_matrix = cosine_similarity(vectors)
    n = len(recommended)
    diversity = (np.sum(sim_matrix) - n) / (n * (n - 1))
    return 1 - diversity

def compute_novelty(recommended, item_popularity):
    novelty_scores = []
    for item in recommended:
        pop = item_popularity.get(item, 1)
        novelty_scores.append(1 / pop)
    return np.mean(novelty_scores) if novelty_scores else 0

def leave_one_out_split(df):
    train_list = []
    test_list = []
    for uid, group in df.groupby('user_id'):
        group = group.sort_values('timestamp')
        if len(group) < 2:
            train_list.append(group)
        else:
            train_list.append(group.iloc[:-1])
            test_list.append(group.iloc[-1:])
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame()
    return train_df, test_df

# ---------------------- Per-User Evaluation Functions ----------------------
def evaluate_memory_user(uid, memory_matrix, similarity_matrix, user_ids, ground_truth, candidate_movies, k=10):
    if uid not in memory_matrix:
        return (0, 0, 0, 0)
    recs = recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=k)
    # Expand candidate set: include both candidate movies and ground truth movies
    user_candidates = set(candidate_movies).union(ground_truth.get(uid, set()))
    recs = [r for r in recs if r in user_candidates]
    # print("Memory-Based -> User:", uid)
    # print("Ground Truth:", ground_truth.get(uid, set()))
    # print("Candidate Set:", user_candidates)
    # print("Recommendations:", recs)
    precision, recall = precision_recall_at_k(recs, ground_truth.get(uid, set()), k)
    hr = hit_rate(recs, ground_truth.get(uid, set()))
    ndcg = ndcg_at_k(recs, ground_truth.get(uid, set()), k)
    return (precision, recall, hr, ndcg)

def evaluate_svd_user(uid, mf_model, candidate_movies, ground_truth, k=10):
    user_preds = []
    for movieId in candidate_movies:
        pred = mf_model.predict(uid, movieId)
        user_preds.append((movieId, pred.est))
    user_preds.sort(key=lambda x: x[1], reverse=True)
    recs = [movieId for movieId, _ in user_preds[:k]]
    # print("SVD-Based -> User:", uid)
    # print("Ground Truth:", ground_truth.get(uid, set()))
    # print("Recommendations:", recs)
    precision, recall = precision_recall_at_k(recs, ground_truth.get(uid, set()), k)
    hr = hit_rate(recs, ground_truth.get(uid, set()))
    ndcg = ndcg_at_k(recs, ground_truth.get(uid, set()), k)
    return (precision, recall, hr, ndcg)

def evaluate_content_user(uid, ground_truth, k=10):
    recs = content_based_recommendation(uid, top_n=k)
    # print("Content-Based -> User:", uid)
    # print("Ground Truth:", ground_truth.get(uid, set()))
    # print("Recommendations:", recs)
    precision, recall = precision_recall_at_k(recs, ground_truth.get(uid, set()), k)
    hr = hit_rate(recs, ground_truth.get(uid, set()))
    ndcg = ndcg_at_k(recs, ground_truth.get(uid, set()), k)
    return (precision, recall, hr, ndcg)

def evaluate_hybrid_user(uid, memory_matrix, similarity_matrix, user_ids, mf_model, candidate_movies, ground_truth, k=10):
    if uid not in memory_matrix:
        return (0, 0, 0, 0)
    recs = hybrid_recommendation(uid, memory_matrix, similarity_matrix, user_ids, mf_model, candidate_movies,
                                 dynamic=True, n_recommendations=k)
    # print("Hybrid -> User:", uid)
    # print("Ground Truth:", ground_truth.get(uid, set()))
    # print("Recommendations:", recs)
    precision, recall = precision_recall_at_k(recs, ground_truth.get(uid, set()), k)
    hr = hit_rate(recs, ground_truth.get(uid, set()))
    ndcg = ndcg_at_k(recs, ground_truth.get(uid, set()), k)
    return (precision, recall, hr, ndcg)

# ---------------------- Main Evaluation Command ----------------------
class Command(BaseCommand):
    help = 'Evaluate recommendation models (memory-based, SVD, content-based, and hybrid) on the MovieLens dataset using parallel processing.'

    def handle(self, *args, **options):
        self.stdout.write("Fetching rating data from the database...")
        # Use Rating and related Movie (via movie__movieId)
        ratings_qs = Rating.objects.all().values('user_id', 'movie__movieId', 'rating', 'timestamp')
        df = pd.DataFrame(list(ratings_qs))
        if df.empty:
            self.stdout.write(self.style.ERROR("No rating data found."))
            return

        # Rename for Surprise: 'movie__movieId' to 'movieId'
        df.rename(columns={'movie__movieId': 'movieId'}, inplace=True)
        df['rating'] = df['rating'].astype(float)
        try:
            df['timestamp'] = pd.to_numeric(df['timestamp'])
        except Exception as e:
            self.stdout.write(self.style.WARNING("Timestamp conversion issue: " + str(e)))

        self.stdout.write("Performing leave-one-out split...")
        train_df, test_df = leave_one_out_split(df)

        # Build ground truth: consider ratings >= 4 as relevant
        ground_truth = {}
        for _, row in test_df.iterrows():
            if row['rating'] >= 4:
                uid = row['user_id']
                movieId = row['movieId']
                ground_truth.setdefault(uid, set()).add(movieId)

        # Filter out users with fewer than 3 interactions in training
        user_interaction_counts = train_df.groupby('user_id').size().to_dict()
        eligible_users = [uid for uid in ground_truth if user_interaction_counts.get(uid, 0) >= 3]
        if not eligible_users:
            self.stdout.write(self.style.ERROR("No eligible users for evaluation (minimum interactions not met)."))
            return

        # -------------------------
        # Memory-Based Evaluation
        # -------------------------
        self.stdout.write("Evaluating Memory-Based Collaborative Filtering...")
        memory_matrix = {}
        all_movies = set()
        for _, row in train_df.iterrows():
            uid = row['user_id']
            movieId = row['movieId']
            memory_matrix.setdefault(uid, {})[movieId] = row['rating']
            all_movies.add(movieId)
        for uid in memory_matrix:
            for movieId in all_movies:
                if movieId not in memory_matrix[uid]:
                    memory_matrix[uid][movieId] = 0
        similarity_matrix, user_ids = calculate_similarity(memory_matrix, list(all_movies))

        # Expand candidate set: use top 200 popular movies
        popular_movies = sorted(train_df.groupby('movieId').size().to_dict().items(), key=lambda x: x[1], reverse=True)
        candidate_movies = [movieId for movieId, _ in popular_movies][:200]

        sampled_users = eligible_users if len(eligible_users) <= 50 else random.sample(eligible_users, 50)

        # Debug print for one sample user
        sample_uid = sampled_users[0]
        print("Sample User (Memory-Based):", sample_uid)
        print("Ground Truth:", ground_truth.get(sample_uid, set()))
        print("Memory-Based Recommendations:",
              recommend_for_user(sample_uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10))

        results_memory = Parallel(n_jobs=-1, backend="threading", verbose=10)(
            delayed(evaluate_memory_user)(uid, memory_matrix, similarity_matrix, user_ids, ground_truth, candidate_movies, k=10)
            for uid in sampled_users
        )
        precisions_memory, recalls_memory, hit_rates_memory, ndcg_scores_memory = zip(*results_memory)
        avg_precision_memory = np.mean(precisions_memory)
        avg_recall_memory = np.mean(recalls_memory)
        avg_hit_rate_memory = np.mean(hit_rates_memory)
        avg_ndcg_memory = np.mean(ndcg_scores_memory)
        self.stdout.write(self.style.SUCCESS(
            f"Memory-Based CF - Precision@10: {avg_precision_memory:.4f}, Recall@10: {avg_recall_memory:.4f}, Hit Rate: {avg_hit_rate_memory:.4f}, NDCG@10: {avg_ndcg_memory:.4f}"
        ))

        # -------------------------
        # Matrix Factorization (SVD) Evaluation
        # -------------------------
        self.stdout.write("Evaluating Matrix Factorization (SVD) approach...")
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(train_df[['user_id', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        mf_model = SVD()
        mf_model.fit(trainset)
        test_data = Dataset.load_from_df(test_df[['user_id', 'movieId', 'rating']], reader)
        testset = test_data.build_full_trainset().build_testset()
        predictions = mf_model.test(testset)
        rmse_value = accuracy.rmse(predictions, verbose=False)
        self.stdout.write(self.style.SUCCESS(f"SVD Model RMSE on test set: {rmse_value:.4f}"))

        results_svd = Parallel(n_jobs=-1, backend="threading", verbose=10)(
            delayed(evaluate_svd_user)(uid, mf_model, candidate_movies, ground_truth, k=10)
            for uid in sampled_users
        )
        precisions_svd, recalls_svd, hit_rates_svd, ndcg_scores_svd = zip(*results_svd)
        avg_precision_svd = np.mean(precisions_svd)
        avg_recall_svd = np.mean(recalls_svd)
        avg_hit_rate_svd = np.mean(hit_rates_svd)
        avg_ndcg_svd = np.mean(ndcg_scores_svd)
        self.stdout.write(self.style.SUCCESS(
            f"SVD Model - Precision@10: {avg_precision_svd:.4f}, Recall@10: {avg_recall_svd:.4f}, Hit Rate: {avg_hit_rate_svd:.4f}, NDCG@10: {avg_ndcg_svd:.4f}"
        ))

        # -------------------------
        # Content-Based Evaluation
        # -------------------------
        self.stdout.write("Evaluating Content-Based Filtering...")
        results_cb = Parallel(n_jobs=-1, backend="threading", verbose=10)(
            delayed(evaluate_content_user)(uid, ground_truth, k=10)
            for uid in sampled_users
        )
        precisions_cb, recalls_cb, hit_rates_cb, ndcg_scores_cb = zip(*results_cb)
        avg_precision_cb = np.mean(precisions_cb)
        avg_recall_cb = np.mean(recalls_cb)
        avg_hit_rate_cb = np.mean(hit_rates_cb)
        avg_ndcg_cb = np.mean(ndcg_scores_cb)
        self.stdout.write(self.style.SUCCESS(
            f"Content-Based CF - Precision@10: {avg_precision_cb:.4f}, Recall@10: {avg_recall_cb:.4f}, Hit Rate: {avg_hit_rate_cb:.4f}, NDCG@10: {avg_ndcg_cb:.4f}"
        ))

        # -------------------------
        # Hybrid Model Evaluation
        # -------------------------
        self.stdout.write("Evaluating Hybrid Recommendation Model...")
        results_hybrid = Parallel(n_jobs=-1, backend="threading", verbose=10)(
            delayed(evaluate_hybrid_user)(uid, memory_matrix, similarity_matrix, user_ids, mf_model, candidate_movies, ground_truth, k=10)
            for uid in sampled_users
        )
        precisions_hybrid, recalls_hybrid, hit_rates_hybrid, ndcg_scores_hybrid = zip(*results_hybrid)
        avg_precision_hybrid = np.mean(precisions_hybrid)
        avg_recall_hybrid = np.mean(recalls_hybrid)
        avg_hit_rate_hybrid = np.mean(hit_rates_hybrid)
        avg_ndcg_hybrid = np.mean(ndcg_scores_hybrid)
        self.stdout.write(self.style.SUCCESS(
            f"Hybrid Model - Precision@10: {avg_precision_hybrid:.4f}, Recall@10: {avg_recall_hybrid:.4f}, Hit Rate: {avg_hit_rate_hybrid:.4f}, NDCG@10: {avg_ndcg_hybrid:.4f}"
        ))

        # -------------------------
        # Optional: Diversity & Novelty for Memory-Based Recommendations
        # -------------------------
        # item_feature_matrix = {movieId: np.random.rand(10) for movieId in list(all_movies)}
        # diversity_memory = np.mean([
        #     # Use raw memory-based recommendations (without candidate filtering) for diversity
        #     compute_diversity(recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10),
        #                       item_feature_matrix)
        #     for uid in ground_truth if uid in memory_matrix
        # ])
        # novelty_memory = np.mean([
        #     compute_novelty(recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10),
        #                     {movieId: count for movieId, count in train_df.groupby('movieId').size().to_dict().items()})
        #     for uid in ground_truth if uid in memory_matrix
        # ])
        # self.stdout.write(self.style.SUCCESS(
        #     f"Memory-Based CF - Average Diversity: {diversity_memory:.4f}, Average Novelty: {novelty_memory:.4f}"
        # ))
        self.stdout.write(self.style.SUCCESS("Evaluation complete."))
