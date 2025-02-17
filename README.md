Below is an updated README that now incorporates detailed information about both data sources—user reviews and product meta data—with sample JSON objects and explanation of each field.

---

```markdown
# Intelligent E-commerce Recommendation System

## Overview

This project aims to build an **Intelligent E-commerce Recommendation System** using **Collaborative Filtering (CF)** augmented with rich product metadata. The system recommends products to users based on their historical interactions (ratings, reviews, timestamps, etc.) and leverages separate meta data to enrich product information. The approach uses a **User-Item Interaction Matrix** and cosine similarity to identify similar users, and it is designed to be later extended with matrix factorization (e.g., SVD) for scalability and improved accuracy.

The system is built using **Django** as the web framework, and a RESTful API is provided to deliver personalized product recommendations. The backend is structured to integrate two types of Amazon Reviews'23 data:
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

## Data

### Dataset Description

The dataset is based on the [Amazon Reviews'23](https://amazon-reviews-2023.github.io/) collection by McAuley Lab and consists of two parts:

1. **User Reviews** (JSONL format):
   - **Sample:**
     ```json
     {
       "sort_timestamp": 1634275259292,
       "rating": 3.0,
       "helpful_votes": 0,
       "title": "Meh",
       "text": "These were lightweight and soft but much too small for my liking. I would have preferred two of these together to make one loc. For that reason I will not be repurchasing.",
       "images": [
         {
           "small_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL256_.jpg",
           "medium_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL800_.jpg",
           "large_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL1600_.jpg",
           "attachment_type": "IMAGE"
         }
       ],
       "asin": "B088SZDGXG",
       "verified_purchase": true,
       "parent_asin": "B08BBQ29N5",
       "user_id": "AEYORY2AVPMCPDV57CE337YU5LXA"
     }
     ```
   - **Field Explanations:**
     - **rating**: Numeric rating (1–5 scale).
     - **title**: Title of the review.
     - **text**: Full review text.
     - **asin**: Unique product ID.
     - **parent_asin**: Group identifier for variants (helps in aggregating products).
     - **user_id**: Identifier of the reviewer.
     - **timestamp**: Unix timestamp (in milliseconds).
     - **helpful_votes**: Count of helpful votes.
     - **verified_purchase**: Indicates if the purchase was verified.
     - **images**: List of image objects (may be empty).

2. **Product Metadata** (JSONL format):
   - **Sample:**
     ```json
     {
       "main_category": "All Beauty",
       "title": "Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)",
       "average_rating": 4.8,
       "rating_number": 10,
       "features": [],
       "description": [],
       "price": null,
       "images": [
         {
           "thumb": "https://m.media-amazon.com/images/I/41qfjSfqNyL._SS40_.jpg",
           "large": "https://m.media-amazon.com/images/I/41qfjSfqNyL.jpg",
           "variant": "MAIN",
           "hi_res": null
         },
         {
           "thumb": "https://m.media-amazon.com/images/I/41w2yznfuZL._SS40_.jpg",
           "large": "https://m.media-amazon.com/images/I/41w2yznfuZL.jpg",
           "variant": "PT01",
           "hi_res": "https://m.media-amazon.com/images/I/71i77AuI9xL._SL1500_.jpg"
         }
       ],
       "videos": [],
       "bought_together": null,
       "store": "Howard Products",
       "categories": [],
       "details": {
         "Package Dimensions": "7.1 x 5.5 x 3 inches; 2.38 Pounds",
         "UPC": "617390882781"
       },
       "parent_asin": "B01CUPMQZE"
     }
     ```
   - **Field Explanations:**
     - **main_category**: The primary domain of the product.
     - **title**: Official product title (more reliable than review titles).
     - **average_rating**: Aggregated product rating.
     - **rating_number**: Number of ratings.
     - **features**: List of product features.
     - **description**: Product description.
     - **price**: Price of the product (can be null).
     - **images**: List of image objects (each with thumb, large, hi_res, and variant).
     - **videos**: List of product videos.
     - **store**: Store name.
     - **categories**: Hierarchical product categories.
     - **details**: Additional product details (dimensions, UPC, etc.).
     - **parent_asin**: Group identifier for product variants.
     - **bought_together**: Recommendations for bundled products.

### Data Preprocessing

- **Data Loading**:  
  Two separate management commands load the data:
  - **Reviews**: Loaded into the `Review` model.
  - **Meta Data**: Loaded into the `Product` model.
- **Product Enrichment**:  
  The meta data updates the `Product` model with fields like `title`, `price`, `features`, and `images`. Reviews reference these enriched Product entries.
- **Interaction Matrix Construction**:  
  A user-item interaction matrix is built from reviews, using the product’s `parent_asin` (if available) to aggregate similar items.

In early testing, a subset (e.g., 10,000 entries) is processed. This can be scaled up as needed.

---

## Features

- **User-Based Collaborative Filtering**: Recommends products by finding users similar to the current user using cosine similarity.
- **Item-Based Collaborative Filtering**: (Future extension) Recommends products similar to those a user has interacted with.
- **Hybrid Filtering**: (Future work) Combine collaborative signals with content-based features from product metadata.
- **Batch Data Insertion**: Efficiently loads large datasets via bulk inserts.
- **Rich API**: Provides endpoints for personalized recommendations enriched with detailed product meta data.

---

## Methodology

### Collaborative Filtering

The system builds a **User-Item Interaction Matrix** from review ratings, using cosine similarity to measure user similarity. The recommendations are generated by computing a weighted sum of ratings from similar users. For instance:
```python
from sklearn.metrics.pairwise import cosine_similarity
user_similarity = cosine_similarity(interaction_matrix)
```

### Matrix Factorization (Future Work)

To further improve accuracy and scalability, matrix factorization (e.g., using SVD) will be explored. This method decomposes the interaction matrix into latent factors representing user preferences and item attributes.

### Evaluation

Model performance will be evaluated using:
- **RMSE (Root Mean Squared Error)**
- **Precision/Recall@K**

Example using the `surprise` library:
```python
from surprise import SVD, Dataset, Reader, accuracy
```

---

## Run the Project

### Steps to Run the Project:

1. **Environment Setup**:  
   Follow the [Project Setup](#project-setup) instructions to install dependencies and configure the database.
   
2. **Load the Data**:  
   Run the management command to load both review and meta data:
   ```bash
   python manage.py load_data
   ```

3. **Run the Server**:  
   Start the Django development server:
   ```bash
   python manage.py runserver
   ```

4. **Access the API**:  
   The API for recommendations is accessible at:
   ```bash
   http://127.0.0.1:8000/recommendations/{user_id}/
   ```

---

## API Documentation

### GET /recommendations/{user_id}/

- **Description**: Returns the top product recommendations for the specified user.
- **Parameter**:  
  - `user_id`: The ID of the user.
- **Response**:  
  A JSON array containing enriched product data (e.g., title, price, features, images). For example:
  ```json
  {
    "recommendations": [
      {"asin": "B00YQ6X8EO", "title": "Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)", "price": null, "features": [], "images": [...]},
      {"asin": "B081TJ8YS3", "title": "Yes to Tomatoes Detoxifying Charcoal Cleanser (Pack of 2)", "price": null, "features": [], "images": [...]}
    ]
  }
  ```

---

## Future Improvements

- **Hybrid Filtering**:  
  Integrate content-based filtering using product metadata to address cold start problems and further enhance recommendation accuracy.
- **Matrix Factorization**:  
  Implement SVD or ALS-based models to reduce dimensionality and improve scalability.
- **Temporal Dynamics**:  
  Incorporate time-aware factors to account for changes in user preferences over time.
- **Explainability**:  
  Develop mechanisms to explain why specific recommendations were made.
- **Scalability Enhancements**:  
  Optimize the system using sparse matrices, caching, or approximate nearest neighbor algorithms.
- **Frontend Integration**:  
  Build a user interface to display recommendations and gather user feedback, further refining the system.

---

This README provides a comprehensive overview of the Intelligent E-commerce Recommendation System, covering data sources, methodology, API endpoints, and future directions. It aligns with the research direction of integrating collaborative filtering with rich product metadata, paving the way for innovative extensions in your master's thesis.

Feel free to modify or extend this document as your project evolves!
```