## Collaborative Filtering for Intelligent E-commerce Recommendation Systems: A Comparative Analysis of Memory-Based, Matrix Factorization, and Hybrid Approaches

### Overview

This project aims to build an **Intelligent E-commerce Recommendation System** using **Collaborative Filtering (CF)** augmented with rich product metadata. The system recommends products to users based on their historical interactions (ratings, reviews, timestamps, etc.) and leverages separate metadata to enrich product information. The approach employs a **User-Item Interaction Matrix** and cosine similarity for the memory-based method, utilizes matrix factorization (e.g., SVD) to uncover latent factors for scalability and improved accuracy, and integrates these methods into a **Hybrid Recommendation Framework** that combines their strengths.

The system is built using **Django** as the web framework, and a RESTful API is provided to deliver personalized product recommendations. The backend integrates two types of data from the Amazon Reviews'23 dataset:
- **User Reviews** (interaction data)
- **Item Metadata** (rich product details)

---

### Table of Contents

- [Project Setup](#project-setup)
- [Data](#data)
  - [Dataset Description](#dataset-description)
  - [Data Preprocessing](#data-preprocessing)
- [Features](#features)
- [Methodology](#methodology)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Matrix Factorization](#matrix-factorization)
  - [Hybrid Filtering](#hybrid-filtering)
  - [Evaluation](#evaluation)
- [Run the Project](#run-the-project)
- [API Documentation](#api-documentation)
- [Future Improvements](#future-improvements)

---

### Project Setup

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
   - Install project dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Database Setup:**
   This project uses **PostgreSQL** for development. Set up the database and apply migrations:
   ```bash
   python manage.py migrate
   ```

4. **Run the Project Locally:**
   ```bash
   python manage.py runserver
   ```
   The project will be accessible at `http://127.0.0.1:8000/`.

---

### Data

#### Dataset Description

The dataset is based on the [Amazon Reviews'23](https://amazon-reviews-2023.github.io/) collection by McAuley Lab and consists of two parts:

1. **User Reviews (All_Beauty.jsonl):**  
   Contains user-generated review data with fields such as:
   - **rating:** Numeric rating (1–5 scale).
   - **title:** Title of the review.
   - **text:** Full review text.
   - **asin:** Unique product identifier.
   - **parent_asin:** Group identifier for variants.
   - **user_id:** Reviewer ID.
   - **timestamp:** When the review was posted (in milliseconds).
   - **helpful_votes:** Number of helpful votes.
   - **verified_purchase:** Indicates if the review is for a verified purchase.
   - **images:** List of image objects (often empty).

2. **Product Metadata (meta_All_Beauty.jsonl):**  
   Provides enriched product information with fields such as:
   - **main_category:** Primary category of the product.
   - **title:** Official product title.
   - **average_rating:** Aggregated rating.
   - **rating_number:** Total number of ratings.
   - **features:** List of product features.
   - **description:** Product description.
   - **price:** Product price (can be null).
   - **images:** List of image objects (each with `thumb`, `large`, `hi_res`, and `variant`).
   - **videos, store, categories, details, parent_asin, bought_together:** Additional fields.

#### Data Preprocessing

- **Data Loading:**  
  The `load_data` management command reads both the review and metadata files, populates the `Review` and `Product` models, and links reviews to enriched products (using `parent_asin` when available).

- **Interaction Matrix:**  
  For collaborative filtering, a user–item interaction matrix is built from review ratings.

---

### Features

- **Memory-Based Collaborative Filtering:**  
  Uses cosine similarity on the user–item interaction matrix to recommend products based on similar users.
- **Matrix Factorization (SVD):**  
  Decomposes the interaction matrix into latent factors to predict ratings more efficiently, offering improved scalability and accuracy.
- **Hybrid Filtering:**  
  Integrates the memory-based and matrix factorization approaches by combining their recommendation scores using a weighted mechanism. This novel hybrid approach aims to overcome limitations such as data sparsity and cold start issues.
- **Batch Data Insertion:**  
  Efficient data loading with bulk inserts.
- **Rich API:**  
  Endpoints for retrieving personalized product recommendations.

---

### Methodology

#### Collaborative Filtering

- **Memory-Based CF:**  
  Directly computes user similarities using cosine similarity. For example:
  ```python
  from sklearn.metrics.pairwise import cosine_similarity
  user_similarity = cosine_similarity(interaction_matrix)
  ```

#### Matrix Factorization

- **SVD (Singular Value Decomposition):**  
  Decomposes the interaction matrix into latent factors to predict ratings more efficiently. (Implemented in the `train_mf` command.)

#### Hybrid Filtering

The hybrid model combines scores from both memory-based and SVD-based methods:
- **Memory-Based Scores:**  
  Derived from the user–item interaction matrix.
- **SVD-Based Scores:**  
  Predicted using the pre-trained SVD model.
- **Combination Strategy:**  
  A weighted sum of the two scores:
  ```python
  hybrid_score = w_memory * (memory-based score) + w_svd * (SVD-based score)
  ```
  This approach produces a final ranked list of recommendations that leverages the strengths of both methods.

#### Evaluation

Evaluation metrics include:
- **RMSE (Root Mean Squared Error):** Measures prediction error.
- **Precision@K and Recall@K:** Evaluate the quality of top-N recommendations.
- **Diversity and Novelty:** Optional metrics to assess recommendation variety and unexpectedness.

See the `evaluate_models` command for detailed evaluation results.

---

### Run the Project

#### Workflow

1. **Load Data:**  
   ```bash
   python manage.py load_data
   ```
   Loads the review and metadata into the database using chunked processing.

2. **Tune the SVD Model:**  
   ```bash
   python manage.py tune_svd
   ```
   Finds optimal hyperparameters for the matrix factorization (SVD) model.

3. **Train the SVD Model:**  
   ```bash
   python manage.py train_mf
   ```
   Trains the SVD model and saves it as `svd_model.pkl`.

4. **Evaluate Models:**  
   ```bash
   python manage.py evaluate_models
   ```
   Compares memory-based, SVD-based, and (in future) hybrid approaches using metrics such as RMSE, Precision@K, Recall@K, etc.

5. **Run the Server:**  
   ```bash
   python manage.py runserver
   ```
   The API endpoints become accessible.

#### API Endpoints

- **Memory-Based Recommendations:**  
  `http://127.0.0.1:8000/products/recommendations/memory/?user_id=<USER_ID>`
  
- **Matrix Factorization Recommendations:**  
  `http://127.0.0.1:8000/products/recommendations/mf/?user_id=<USER_ID>`
  
- **Hybrid Recommendations:**  
  `http://127.0.0.1:8000/products/recommendations/hybrid/?user_id=<USER_ID>`

*Example response:*
```json
{
  "recommendations": [
    {"asin": "B00YQ6X8EO", "title": "Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)", "price": null, "features": [], "images": [...]},
    {"asin": "B081TJ8YS3", "title": "Yes to Tomatoes Detoxifying Charcoal Cleanser (Pack of 2)", "price": null, "features": [], "images": [...]}
  ]
}
```

---

### Future Improvements

- **Enhance Hybrid Filtering:**  
  Experiment with dynamic weighting strategies that adapt based on user activity and metadata richness.
- **Matrix Factorization Enhancements:**  
  Explore alternative methods (e.g., ALS) and advanced hyperparameter tuning.
- **Temporal Dynamics:**  
  Incorporate time-aware factors to capture evolving user preferences.
- **Explainability:**  
  Develop methods to provide clear explanations for recommendations.
- **Scalability Enhancements:**  
  Implement caching, batch processing, and distributed computing techniques.
- **Frontend Integration:**  
  Build a user interface to display recommendations and collect feedback.
- **Thesis Documentation:**  
  Continuously document methodologies, experiments, and findings for your thesis.

---


