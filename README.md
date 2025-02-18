
## Intelligent E-commerce Recommendation System

### Overview

This project aims to build an **Intelligent E-commerce Recommendation System** using **Collaborative Filtering (CF)** augmented with rich product metadata. The system recommends products to users based on their historical interactions (ratings, reviews, timestamps, etc.) and leverages separate meta data to enrich product information. The approach uses a **User-Item Interaction Matrix** and cosine similarity for the memory-based method, and it is designed to be extended with matrix factorization (e.g., SVD) for scalability and improved accuracy.

The system is built using **Django** as the web framework, and a RESTful API is provided to deliver personalized product recommendations. The backend integrates two types of data from the Amazon Reviews'23 dataset:
- **User Reviews** (interaction data)
- **Item Metadata** (rich product details)

## Table of Contents

- [Project Setup](#project-setup)
- [Data](#data)
  - [Dataset Description](#dataset-description)
  - [Data Preprocessing](#data-preprocessing)
- [Features](#features)
- [Methodology](#methodology)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Matrix Factorization (Future Work)](#matrix-factorization-future-work)
  - [Evaluation](#evaluation)
- [Run the Project](#run-the-project)
- [API Documentation](#api-documentation)
- [Future Improvements](#future-improvements)

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

## Data

### Dataset Description

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

### Data Preprocessing

- **Data Loading:**  
  The `load_data` management command reads both the review and meta data files, populates the `Review` and `Product` models, and links reviews to enriched products (using `parent_asin` when available).

- **Interaction Matrix:**  
  For collaborative filtering, a user–item interaction matrix is built from review ratings.

## Features

- **Memory-Based Collaborative Filtering:**  
  Uses cosine similarity on the user–item interaction matrix to recommend products based on similar users.
- **Matrix Factorization (Model-Based) Collaborative Filtering:**  
  (Future Work) Uses SVD to decompose the interaction matrix into latent factors for scalability and improved accuracy.
- **Hybrid Filtering:**  
  Future integration of content-based signals from product metadata.
- **Batch Data Insertion:**  
  Efficient data loading with bulk inserts.
- **Rich API:**  
  Endpoints for retrieving personalized product recommendations.

## Methodology

### Collaborative Filtering

- **Memory-Based CF:**  
  Directly computes user similarities from the interaction matrix. For example:
  ```python
  from sklearn.metrics.pairwise import cosine_similarity
  user_similarity = cosine_similarity(interaction_matrix)
  ```

### Matrix Factorization (Future Work)

- **SVD (Singular Value Decomposition):**  
  Decomposes the interaction matrix into latent factors to predict ratings more efficiently. (Implemented in the `train_mf` command.)

### Evaluation

Evaluation metrics include:
- **RMSE (Root Mean Squared Error):** Measures prediction error.
- **Precision@K and Recall@K:** Evaluate the quality of the top-N recommendations.
- **Diversity and Novelty:** Optional metrics to assess recommendation variety and unexpectedness.

## Run the Project

### Workflow

1. **Load Data:**  
   Load the review and meta data into your database:
   ```bash
   python manage.py load_data
   ```

2. **Train the Matrix Factorization Model:**  
   Train the SVD model and save it:
   ```bash
   python manage.py train_mf
   ```

3. **Evaluate Models:**  
   Run the evaluation command to compare memory-based and matrix factorization approaches:
   ```bash
   python manage.py evaluate_models
   ```

4. **Run the Server:**  
   Start the Django server to access the API:
   ```bash
   python manage.py runserver
   ```

### API Endpoints

- **Memory-Based Recommendations:**  
  Accessible at:  
  `http://127.0.0.1:8000/products/recommendations/memory/?user_id=<USER_ID>`

- **Matrix Factorization Recommendations:**  
  Accessible at:  
  `http://127.0.0.1:8000/products/recommendations/mf/?user_id=<USER_ID>`


- **Description:** Returns the top product recommendations for the specified user.
- **Parameter:**  
  - `user_id`: The ID of the user.
- **Response:**  
  A JSON array containing enriched product data (e.g., title, price, features, images).

Example response:
```json
{
  "recommendations": [
    {"asin": "B00YQ6X8EO", "title": "Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)", "price": null, "features": [], "images": [...]},
    {"asin": "B081TJ8YS3", "title": "Yes to Tomatoes Detoxifying Charcoal Cleanser (Pack of 2)", "price": null, "features": [], "images": [...]}
  ]
}
```

## Future Improvements

- **Hybrid Filtering:**  
  Integrate content-based filtering with collaborative filtering using product metadata to address cold start problems.
- **Matrix Factorization Enhancements:**  
  Experiment with different latent factors, regularization, and alternative optimization methods (e.g., ALS).
- **Temporal Dynamics:**  
  Incorporate time-aware factors to capture changes in user preferences.
- **Explainability:**  
  Develop methods to provide explanations for recommendations.
- **Scalability Enhancements:**  
  Utilize sparse matrices, caching, or approximate nearest neighbor algorithms for handling very large datasets.
- **Frontend Integration:**  
  Build a user interface to display recommendations and collect feedback.
- **Thesis Documentation:**  
  Document methodologies, experiments, and comparisons in your thesis.

---

This README provides a comprehensive overview of the project, explains how the data files are connected, details the workflow of data loading, model training, evaluation, and API usage, and outlines future directions for further enhancements.
```

