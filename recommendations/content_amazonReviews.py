# recommendations/content_amazonReviews.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Product, Review

# Global variables to cache the TF-IDF matrix, product IDs, and vectorizer
tfidf_matrix = None
product_ids = None
tfidf_vectorizer = None


def build_product_tfidf_matrix():
    """
    Build and cache a TF-IDF matrix for all products using their title and description.
    If the 'description' field is missing or empty, it falls back to using the title.
    """
    global tfidf_matrix, product_ids, tfidf_vectorizer

    # Fetch products from the database
    products = Product.objects.all().values('asin', 'title', 'description')
    df = pd.DataFrame(list(products))
    if df.empty:
        return None, None, None

    # Fill missing values for 'title' and 'description'
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')

    # If 'description' is a list, join its elements into a single string
    df['description'] = df['description'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)

    # Use 'description' if available, otherwise fall back to 'title'
    df['description'] = df.apply(lambda row: row['description'] if row['description'].strip() != "" else row['title'],
                                 axis=1)

    # Combine title and description to form a single text field
    df['text'] = df['title'] + " " + df['description']
    product_ids = df['asin'].tolist()

    # Initialize and fit the TF-IDF vectorizer on the combined text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    return tfidf_matrix, product_ids, tfidf_vectorizer


def content_based_recommendation(user_id, top_n=10):
    """
    Generate content-based recommendations for a given user.
    This method computes an average TF-IDF vector for the products that the user has reviewed,
    then computes cosine similarity between this average vector and all product vectors.
    """
    global tfidf_matrix, product_ids, tfidf_vectorizer
    if tfidf_matrix is None or product_ids is None:
        tfidf_matrix, product_ids, tfidf_vectorizer = build_product_tfidf_matrix()
        if tfidf_matrix is None:
            return []

    # Get the list of ASINs the user has reviewed
    reviewed_asins = list(Review.objects.filter(user_id=user_id).values_list('product__asin', flat=True))
    if not reviewed_asins:
        return []

    # Map ASIN to index in the TF-IDF matrix
    asin_to_index = {asin: idx for idx, asin in enumerate(product_ids)}
    indices = [asin_to_index[asin] for asin in reviewed_asins if asin in asin_to_index]
    if not indices:
        return []

    # Compute the average TF-IDF vector for the user and convert it to a numpy array
    user_vector = np.asarray(tfidf_matrix[indices].mean(axis=0))

    # Compute cosine similarity between the user vector and all product vectors
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Create a DataFrame to sort the similarity scores
    sim_df = pd.DataFrame({'asin': product_ids, 'score': sim_scores})
    # Exclude products already reviewed by the user
    sim_df = sim_df[~sim_df['asin'].isin(reviewed_asins)]
    sim_df = sim_df.sort_values(by='score', ascending=False)

    recommended = sim_df.head(top_n)['asin'].tolist()
    return recommended

