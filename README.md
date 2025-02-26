## Collaborative Filtering for Intelligent E-commerce Recommendation Systems:  
A Comparative Analysis of Memory-Based, Matrix Factorization, Content-Based, and Hybrid Approaches

### Overview

This project builds an **Intelligent E-commerce Recommendation System** that leverages multiple recommendation techniques to personalize product suggestions. It is designed to work with two types of datasets:
- **Amazon Reviews Dataset:** Uses product reviews and metadata.
- **MovieLens Dataset:** Uses movie ratings and metadata.

The system implements four recommendation approaches:
1. **Memory-Based Collaborative Filtering:** Uses a user–item (or user–movie) interaction matrix and cosine similarity.
2. **Matrix Factorization (SVD):** Decomposes the interaction matrix into latent factors for scalable rating predictions.
3. **Content-Based Filtering:** Uses TF-IDF vectorization on product (or movie) text (e.g., titles, descriptions, genres) and cosine similarity.
4. **Hybrid Filtering:** Dynamically combines scores from memory‑based and SVD‑based models using a logistic weighting function.

The project is implemented in **Python** with **Django** and **Django REST Framework** and uses **PostgreSQL** as the database. Key libraries include **NumPy**, **Pandas**, **scikit-learn**, and **Surprise**. Advanced enhancements such as caching and asynchronous processing (via Celery) are incorporated to efficiently handle large datasets.

---

## Table of Contents

- [Project Setup](#project-setup)
- [Data](#data)
  - [Dataset Description](#dataset-description)
  - [Data Preprocessing](#data-preprocessing)
- [Features](#features)
- [Methodology](#methodology)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Matrix Factorization (SVD)](#matrix-factorization-svd)
  - [Content-Based Filtering](#content-based-filtering)
  - [Hybrid Filtering](#hybrid-filtering)
- [Evaluation and Results](#evaluation-and-results)
- [Advanced Enhancements](#advanced-enhancements)
  - [Caching](#caching)
  - [Asynchronous Processing with Celery](#asynchronous-processing-with-celery)
- [Project Structure](#project-structure)
- [Run the Project](#run-the-project)
- [API Documentation](#api-documentation)
- [Future Improvements](#future-improvements)
- [Next Steps](#next-steps)

---

## Project Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/MTayyab10/recommendation_sys.git
   cd recommendation_sys
   ```

2. **Install Dependencies:**
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate   # On Windows: venv\Scripts\activate
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Database Setup:**
   Set up your PostgreSQL database and run migrations:
   ```bash
   python manage.py migrate
   ```

4. **Run the Project Locally:**
   ```bash
   python manage.py runserver
   ```
   Your project is now accessible at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

---

## Data

### Dataset Description

This project supports two datasets:

1. **Amazon Reviews Dataset:**
   - **User Reviews (All_Beauty.jsonl):**  
     Contains review data with fields such as `rating`, `title`, `text`, `asin`, `parent_asin`, `user_id`, `timestamp`, `helpful_votes`, `verified_purchase`, and `images`.
   - **Product Metadata (meta_All_Beauty.jsonl):**  
     Contains enriched product data with fields such as `main_category`, `title`, `average_rating`, `rating_number`, `features`, `description`, `price`, `images`, and additional attributes.

2. **MovieLens Dataset (ml-latest-small):**
   - **movies.csv:** Contains movie details including `movieId`, `title`, and `genres`.
   - **ratings.csv:** Contains rating data with `user_id`, `movieId`, `rating`, and `timestamp`.
   - (Additional files like `tags.csv` and `links.csv` are available for extended functionality.)

### Data Preprocessing

- **Loading Data:**  
  Custom management commands (`load_amazonReviews.py` and `load_movieLens.py`) read the respective data files, populate the models (e.g., `Product` and `Review` for Amazon; `Movie` and `Rating` for MovieLens), and link reviews/ratings to metadata.
  
- **Interaction Matrix:**  
  An interaction matrix is built from review/rating data for collaborative filtering methods.

interaction_matrix---

## Features

- **Memory-Based Collaborative Filtering:**  
  Uses cosine similarity on the interaction matrix to recommend similar items.
  
- **Matrix Factorization (SVD):**  
  Uses latent factor models (via the Surprise library) for scalable predictions.
  
- **Content-Based Filtering:**  
  Applies TF-IDF vectorization on product/movie text (e.g., title, description, genres) to compute similarities.
  
- **Hybrid Filtering:**  
  Combines memory-based and SVD-based scores dynamically using a logistic weighting function.
  
- **Batch Data Insertion:**  
  Loads large datasets efficiently using bulk inserts.
  
- **Rich API Endpoints:**  
  Provides endpoints for memory-based, matrix factorization, content-based, and hybrid recommendations.
  
- **Advanced Enhancements:**  
  Integrates caching (with Django’s caching framework) and asynchronous processing (with Celery) to improve performance.

---

## Methodology

### Collaborative Filtering

- **Memory-Based CF:**  
  Constructs a user–item (or user–movie) interaction matrix and computes cosine similarity:
  ```python
  from sklearn.metrics.pairwise import cosine_similarity
  user_similarity = cosine_similarity()
  ```

### Matrix Factorization (SVD)

- **SVD:**  
  Decomposes the interaction matrix into latent factors using the Surprise library:
  ```python
  from surprise import SVD, Dataset, Reader
  # Example: Train SVD model and evaluate RMSE.
  ```

### Content-Based Filtering

- **TF-IDF & Cosine Similarity:**  
  Builds a TF-IDF matrix from product/movie text (e.g., title + description or title + genres) and computes similarity scores:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  # Compute cosine similarity between TF-IDF vectors.
  ```

### Hybrid Filtering

- **Dynamic Weighting:**  
  Combines memory-based and SVD-based recommendations using dynamic weights computed via a logistic function:
  ```python
  hybrid_score = w_memory * (memory-based score) + w_svd * (SVD-based score)
  ```
  We adjust weights based on the number of interactions per user.

---

## Evaluation and Results

The evaluation metrics:

1. **Data Preparation:**
   - Fetches data, samples 10% for evaluation, and performs a leave-one-out split.
   - Constructs ground truth where ratings ≥ 4 are considered relevant.

2. **Method Evaluations:**
   - **Memory-Based:**  
     Evaluates ranking metrics (Precision@10, Recall@10, Hit Rate, NDCG@10) on a candidate set (top 200 popular items).
   - **Matrix Factorization (SVD):**  
     Trains the SVD model, computes RMSE, and evaluates ranking metrics.
   - **Content-Based:**  
     Generates recommendations using TF-IDF and computes ranking metrics.
   - **Hybrid:**  
     Combines scores from memory-based and SVD approaches and evaluates the same metrics.

3. **Additional Metrics:**
   - **Diversity:** Average pairwise dissimilarity among recommended items.
   - **Novelty:** Based on the inverse popularity of items.

Results are printed to the console and can be exported for visualization in your thesis.

### Evaluation Metrics – What to Expect
If tuned well, you might see evaluation numbers similar to:

**1. Memory-Based CF:**
   - Precision@10: ~0.01 to 0.05 
   - Recall@10: ~0.05 to 0.15
   - Hit Rate: ~0.05 to 0.2
   - NDCG@10: ~0.1 to 0.25

**2. SVD (MF):** 
   - Precision@10 around 0.10–0.20, Recall@10 around 0.10–0.20, RMSE around 0.90–1.0

**3. Content-Based CF:** 
   - (Depending on text quality) Precision@10 around 0.05–0.15, Recall@10 around 0.05–0.15

**4. Hybrid Model:**
   - Ideally higher than individual baselines; for example, Precision@10 around 0.15–0.25, Recall@10 around 0.15–0.25, and NDCG@10 above 0.15

---

## Advanced Enhancements

### Caching

Django’s caching is configured in `settings.py` to store computed recommendations:
```python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}
```

### Asynchronous Processing with Celery

Celery handles heavy computation asynchronously:
- **Configuration in `settings.py`:**
  ```python
  CELERY_BROKER_URL = 'redis://localhost:6379/0'
  CELERY_RESULT_BACKEND = 'redis://localhost:6379/1'
  ```
- **Celery tasks** are defined in `recommendations/tasks.py`.
- **Start the Celery worker:**
  ```bash
  celery -A recommendation_sys worker --loglevel=info
  ```

---

## Project Structure

```
├── README.md
├── data
│   ├── All_Beauty.jsonl
│   ├── Gift_Cards.jsonl
│   ├── meta_All_Beauty.jsonl
│   ├── ml-latest-small
│   │   ├── README.md
│   │   ├── links.csv
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   └── tags.csv
│   └── review_data.py
├── manage.py
├── recommendation_sys
│   ├── __init__.py
│   ├── asgi.py
│   ├── celery.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── recommendations
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── content_amazonReviews.py
│   ├── content_movieLens.py
│   ├── hybrid_amazonReviews.py
│   ├── hybrid_movieLens.py
│   ├── management
│   │   └── commands
│   │       ├── evaluate_amazonReviews.py
│   │       ├── evaluate_movieLens.py
│   │       ├── load_amazonReviews.py
│   │       ├── load_movieLens.py
│   │       ├── train_mf_amazonReviews.py
│   │       ├── train_mf_movieLens.py
│   │       ├── tune_svd_amazonReviews.py
│   │       └── tune_svd_movieLens.py
│   ├── memory_amazonReviews.py
│   ├── memory_movieLens.py
│   ├── methods.py
│   ├── models.py
│   ├── serializers.py
│   ├── tasks.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── requirements.txt
└── svd_model.pkl
```

---

## Run the Project

### Workflow

1. **Load Data:**
   - For Amazon Reviews:
     ```bash
     python manage.py load_amazonReviews
     ```
   - For MovieLens:
     ```bash
     python manage.py load_movieLens
     ```
2. **Tune the SVD Model:**
   - For Amazon Reviews:
     ```bash
     python manage.py tune_svd_amazonReviews
     ```
   - For MovieLens:
     ```bash
     python manage.py tune_svd_movieLens
     ```
3. **Train the SVD Model:**
   - For Amazon Reviews:
     ```bash
     python manage.py train_mf_amazonReviews
     ```
   - For MovieLens:
     ```bash
     python manage.py train_mf_movieLens
     ```
4. **Evaluate Models:**
   - For Amazon Reviews:
     ```bash
     python manage.py evaluate_amazonReviews
     ```
   - For MovieLens:
     ```bash
     python manage.py evaluate_movieLens
     ```
5. **Run the Server:**
   ```bash
   python manage.py runserver
   ```
6. **Start Celery Worker:**
   ```bash
   celery -A recommendation_sys worker --loglevel=info
   ```

---

## API Documentation

### Endpoints

- **Memory-Based Recommendations:**  
  `GET /products/recommendations/memory/?user_id=<USER_ID>`

- **Matrix Factorization Recommendations:**  
  `GET /products/recommendations/mf/?user_id=<USER_ID>`

- **Content-Based Recommendations:**  
  `GET /products/recommendations/content/?user_id=<USER_ID>`

- **Hybrid Recommendations:**  
  `GET /products/recommendations/hybrid/?user_id=<USER_ID>`

- **Task Status (for asynchronous tasks):**  
  `GET /recommendations/task-status/<task_id>/`

*Example Response for Hybrid Recommendations:*
```json
{
  "task_id": "30d607bf-ff62-4ce0-995b-ce70795c2f3b",
  "status": "SUCCESS",
  "result": [
      "B08RS762Z7",
      "B071NBFSLY",
      "B092QKS5WX",
      "B005FLNLM8",
      "B0785QM77V",
      "B08KHRF9NY",
      "B07SQ8193F",
      "B075MPBJDF",
      "B0943GJDP6",
      "B07JFRNWDY"
  ]
}
```

---

## Evaluation and Results

The evaluation command measures the performance of each recommendation method using:
- **Ranking Metrics:** Precision@10, Recall@10, Hit Rate, NDCG@10.
- **Prediction Accuracy:** RMSE for the SVD model.
- **Additional Metrics:** Diversity and novelty (optional).

The evaluation process involves:
1. Sampling a fraction (e.g., 10%) of the data.
2. Performing a leave-one-out split per user.
3. Building ground truth based on a relevance threshold (e.g., ratings ≥ 4).
4. Evaluating each method on a candidate set (e.g., top 200 popular items).
5. Parallelizing per-user evaluations to handle large datasets.

Results are printed to the console and can be exported for visualization and inclusion in your thesis.

### Evaluation Metrics – What to Expect
If tuned well, you might see evaluation numbers similar to:

**1. Memory-Based CF:**
   - Precision@10 around 0.05–0.10,  Recall@10 around 0.05–0.10 

**2. SVD (MF):** 
   - Precision@10 around 0.10–0.20, Recall@10 around 0.10–0.20, RMSE around 0.90–1.0

**3. Content-Based CF:** 
   - (Depending on text quality) Precision@10 around 0.05–0.15, Recall@10 around 0.05–0.15

**4. Hybrid Model:**
   - Ideally higher than individual baselines; for example, Precision@10 around 0.15–0.25, Recall@10 around 0.15–0.25, and NDCG@10 above 0.15


---

## Future Improvements

- **Refine Dynamic Weighting:** Explore alternative functions to dynamically adjust hybrid model weights.
- **Deep Learning Integration:** Consider implementing neural collaborative filtering or session-based models.
- **Enhanced Evaluation:** Incorporate additional metrics (e.g., serendipity, statistical tests) and perform A/B testing.
- **Explainability:** Implement techniques (e.g., LIME, SHAP) to provide transparent explanations for recommendations.
- **Frontend Integration:** Develop a user-friendly UI to display recommendations and capture user feedback.
- **Scalability:** Utilize distributed processing (e.g., Apache Spark) and advanced caching (e.g., Redis) to handle larger datasets.

---
