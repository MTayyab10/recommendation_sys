"""
Evaluation Command for Recommendation Models (Scaled for Large Datasets)

This command performs the following steps:

1. Data Preparation:
   - Fetch review data from the database via Django ORM.
   - Convert the data to a Pandas DataFrame and sample a fraction (e.g., 10%).
   - Split the data using a leave-one-out strategy per user (each user’s most recent interaction is held out as test).
   - Build a ground truth mapping per user from the test set (considering ratings >= 4 as relevant).

2. Memory-Based Evaluation:
   - Build a user–item interaction matrix from the training data.
   - Compute cosine similarities and generate top-10 recommendations for a sampled subset of users.
   - Compute Precision@10, Recall@10, Hit Rate, and NDCG@10 using a candidate set (top 100 popular items).

3. Matrix Factorization Evaluation (SVD):
   - Train an SVD model on the training data using the Surprise library.
   - Evaluate the model using RMSE on the test set.
   - For each sampled user, predict ratings over a candidate set of top 100 popular items, extract the top-10 recommendations, and compute evaluation metrics.

4. Diversity and Novelty (Optional):
   - Compute diversity using a dummy item feature matrix.
   - Compute novelty based on item popularity (number of reviews).
"""

import random
import numpy as np
import datetime
import pandas as pd

from django.core.management.base import BaseCommand
from django.db.models import Count
from recommendations.models import Review, Product
from recommendations.memory_based import create_interaction_matrix, calculate_similarity, recommend_for_user

from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise import SVD

# Evaluation Metrics
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

class Command(BaseCommand):
    help = 'Evaluate both memory-based and matrix factorization recommendation models (scaled for large datasets)'

    def handle(self, *args, **options):
        self.stdout.write("Fetching review data from the database...")
        # Sample a fraction of reviews (e.g., 10%) to reduce load during evaluation
        reviews_qs = Review.objects.all().values('user_id', 'product__asin', 'rating', 'timestamp')
        df = pd.DataFrame(list(reviews_qs)).sample(frac=0.1, random_state=42)
        if df.empty:
            self.stdout.write(self.style.ERROR("No review data found."))
            return
        df.rename(columns={'product__asin': 'asin'}, inplace=True)
        df['rating'] = df['rating'].astype(float)
        # Ensure timestamp is numeric (if already datetime, this may be skipped)
        try:
            df['timestamp'] = pd.to_numeric(df['timestamp'])
        except Exception as e:
            self.stdout.write(self.style.WARNING("Timestamp conversion issue: " + str(e)))

        self.stdout.write("Performing leave-one-out split...")
        train_df, test_df = leave_one_out_split(df)

        # Build ground truth: only interactions with rating >= 4 are considered relevant
        ground_truth = {}
        for _, row in test_df.iterrows():
            if row['rating'] >= 4:
                uid = row['user_id']
                asin = row['asin']
                if uid not in ground_truth:
                    ground_truth[uid] = set()
                ground_truth[uid].add(asin)

        # -------------------------
        # Memory-Based Evaluation
        # -------------------------
        self.stdout.write("Evaluating Memory-Based Collaborative Filtering...")
        memory_matrix = {}
        all_items = set()
        for _, row in train_df.iterrows():
            uid = row['user_id']
            asin = row['asin']
            if uid not in memory_matrix:
                memory_matrix[uid] = {}
            memory_matrix[uid][asin] = row['rating']
            all_items.add(asin)
        for uid in memory_matrix:
            for asin in all_items:
                if asin not in memory_matrix[uid]:
                    memory_matrix[uid][asin] = 0
        similarity_matrix, user_ids = calculate_similarity(memory_matrix, list(all_items))

        precisions_memory, recalls_memory, hit_rates_memory, ndcg_scores_memory = [], [], [], []
        # Sample up to 100 users for evaluation
        sampled_users = list(ground_truth.keys())
        if len(sampled_users) > 50:
            sampled_users = random.sample(sampled_users, 50)

        # For memory-based, we can also limit candidate items to top 100 popular items
        popular_items = sorted(train_df.groupby('asin').size().to_dict().items(), key=lambda x: x[1], reverse=True)
        candidate_items = [asin for asin, _ in popular_items][:50]

        for uid in sampled_users:
            if uid not in memory_matrix:
                continue
            # Instead of computing recommendations over all items, restrict to candidate_items
            recs = recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10)
            # Filter recommendations to candidate items (for fair evaluation)
            recs = [r for r in recs if r in candidate_items]
            prec, rec = precision_recall_at_k(recs, ground_truth[uid], k=10)
            hr = hit_rate(recs, ground_truth[uid])
            ndcg = ndcg_at_k(recs, ground_truth[uid], k=10)
            precisions_memory.append(prec)
            recalls_memory.append(rec)
            hit_rates_memory.append(hr)
            ndcg_scores_memory.append(ndcg)
        avg_precision_memory = np.mean(precisions_memory) if precisions_memory else 0
        avg_recall_memory = np.mean(recalls_memory) if recalls_memory else 0
        avg_hit_rate_memory = np.mean(hit_rates_memory) if hit_rates_memory else 0
        avg_ndcg_memory = np.mean(ndcg_scores_memory) if ndcg_scores_memory else 0

        self.stdout.write(self.style.SUCCESS(
            f"Memory-Based CF - Precision@10: {avg_precision_memory:.4f}, Recall@10: {avg_recall_memory:.4f}, Hit Rate: {avg_hit_rate_memory:.4f}, NDCG@10: {avg_ndcg_memory:.4f}"
        ))

        # -------------------------
        # Matrix Factorization Evaluation (SVD)
        # -------------------------
        self.stdout.write("Evaluating Matrix Factorization (SVD) approach...")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_df[['user_id', 'asin', 'rating']], reader)
        trainset = data.build_full_trainset()
        mf_model = SVD()
        mf_model.fit(trainset)

        test_data = Dataset.load_from_df(test_df[['user_id', 'asin', 'rating']], reader)
        testset = test_data.build_full_trainset().build_testset()
        predictions = mf_model.test(testset)
        rmse_value = accuracy.rmse(predictions, verbose=False)
        self.stdout.write(self.style.SUCCESS(f"SVD Model RMSE on test set: {rmse_value:.4f}"))

        precisions_mf, recalls_mf, hit_rates_mf, ndcg_scores_mf = [], [], [], []
        # Instead of predicting over all products, use candidate_items (top 100 popular)
        for uid in sampled_users:
            user_preds = []
            for asin in candidate_items:
                pred = mf_model.predict(uid, asin)
                user_preds.append((asin, pred.est))
            user_preds.sort(key=lambda x: x[1], reverse=True)
            recs = [asin for asin, _ in user_preds[:10]]
            prec, rec = precision_recall_at_k(recs, ground_truth.get(uid, set()), k=10)
            hr = hit_rate(recs, ground_truth.get(uid, set()))
            ndcg = ndcg_at_k(recs, ground_truth.get(uid, set()), k=10)
            precisions_mf.append(prec)
            recalls_mf.append(rec)
            hit_rates_mf.append(hr)
            ndcg_scores_mf.append(ndcg)
        avg_precision_mf = np.mean(precisions_mf) if precisions_mf else 0
        avg_recall_mf = np.mean(recalls_mf) if recalls_mf else 0
        avg_hit_rate_mf = np.mean(hit_rates_mf) if hit_rates_mf else 0
        avg_ndcg_mf = np.mean(ndcg_scores_mf) if ndcg_scores_mf else 0

        self.stdout.write(self.style.SUCCESS(
            f"SVD Model - Precision@10: {avg_precision_mf:.4f}, Recall@10: {avg_recall_mf:.4f}, Hit Rate: {avg_hit_rate_mf:.4f}, NDCG@10: {avg_ndcg_mf:.4f}"
        ))

        # -------------------------
        # Optional: Compute Diversity and Novelty for Memory-Based Recommendations
        # -------------------------
        item_feature_matrix = {asin: np.random.rand(10) for asin in list(all_items)}
        diversity_memory = np.mean([compute_diversity(recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10), item_feature_matrix)
                                     for uid in ground_truth if uid in memory_matrix])
        novelty_memory = np.mean([compute_novelty(recommend_for_user(uid, memory_matrix, similarity_matrix, user_ids, n_recommendations=10),
                                                  {asin: count for asin, count in train_df.groupby('asin').size().to_dict().items()})
                                   for uid in ground_truth if uid in memory_matrix])
        self.stdout.write(self.style.SUCCESS(
            f"Memory-Based CF - Average Diversity: {diversity_memory:.4f}, Average Novelty: {novelty_memory:.4f}"
        ))
        self.stdout.write(self.style.SUCCESS("Evaluation complete."))
