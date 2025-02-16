

## **Intelligent E-commerce Recommendation System**

### **Overview**

This project aims to build an **Intelligent E-commerce Recommendation System** using **Collaborative Filtering (CF)**. The system recommends products to users based on their historical interactions and ratings. The approach uses a **User-Item Interaction Matrix** to calculate user similarity and make personalized recommendations.

The system is built using **Django** as the web framework and **Collaborative Filtering** as the core recommendation algorithm. The recommendation system is integrated into a web API that can provide personalized product recommendations based on user data.

### **Table of Contents**
- [Project Setup](#project-setup)
- [Data](#data)
  - [Dataset Description](#dataset-description)
  - [Data Preprocessing](#data-preprocessing)
- [Features](#features)
- [Methodology](#methodology)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Evaluation](#evaluation)
- [Run the Project](#run-the-project)
- [API Documentation](#api-documentation)
- [Future Improvements](#future-improvements)

---

## **Project Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MTayyab10/recommendation_sys.git
   cd recommendation_sys
   ```

2. **Install dependencies:**
   - Create a virtual environment and activate it:
     ```bash
     python -m venv venv
     source venv/bin/activate   # On Windows use: venv\Scripts\activate
     ```
   - Install project dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Database Setup**:
   The project uses **PostgreSQL** for development. To set up the database:
   ```bash
   python manage.py migrate
   ```

4. **Run the project locally**:
   ```bash
   python manage.py runserver
   ```
   Now the project will be accessible at `http://127.0.0.1:8000/`.

---

## **Data**

### **Dataset Description**

The dataset used in this project contains [Amazon product reviews](https://amazon-reviews-2023.github.io/). It includes the following data for each review:
1. **Review Data** (in JSONL format):
   - `rating`: The rating given by the user (1-5 scale).
   - `title`: Title of the review.
   - `text`: Detailed review content provided by the user.
   - `asin`: The Amazon Standard Identification Number for the product.
   - `parent_asin`: Parent ASIN for grouped products (like variants).
   - `user_id`: Unique identifier for the user who submitted the review.
   - `timestamp`: The timestamp of when the review was posted (in milliseconds).
   - `helpful_vote`: The number of users who found the review helpful.
   - `verified_purchase`: Boolean indicating whether the product was verified as purchased.

2. **Product Metadata**:
   - `asin`: The product's ASIN (same as in the review data).
   - `title`: Name of the product.
   - `price`: Price of the product (optional).
   - `features`: Key product features (stored as JSON).
   - `parent_asin`: Group identifier for variants of the same product.

### **Data Preprocessing**

- The dataset is loaded into Django models: **Product** and **Review**.
- **Product**: Contains information about the product itself, such as its title, price, and features.
- **Review**: Contains user interactions, including the rating, title, and text of the review.

During preprocessing, the following steps are performed:
1. **Missing Values**: Handle missing values, particularly for `price` and `features` in the product data.
2. **Interaction Matrix**: Construct a **User-Item Interaction Matrix**, where users are represented by rows, products by columns, and the ratings are used as values.
3. **Data Limiting**: In the initial testing phase, data is limited to 100 products/reviews for faster testing. You can adjust this as needed.

---

## **Features**

- **User-Based Collaborative Filtering**: Recommends products by finding users similar to the current user.
- **Item-Based Collaborative Filtering**: Recommends products that are similar to those the user has interacted with.
- **Hybrid Filtering**: Combine user and item-based approaches to improve recommendation quality.
- **Batch Data Insertion**: Efficiently loads the dataset into the database using batch inserts.
- **API**: Provides an API to get product recommendations for a specific user.

---

### **Methodology**

## **Collaborative Filtering**

The recommendation system uses **Collaborative Filtering** to recommend products based on users’ historical interactions (ratings). There are two main approaches:

1. **User-Based Collaborative Filtering**:
   - This method recommends products based on the ratings of similar users. For example, if User A and User B have rated the same products similarly, we can recommend products that User B likes to User A.
   
   We use **cosine similarity** to measure user similarity:
   ```python
   from sklearn.metrics.pairwise import cosine_similarity

   user_similarity = cosine_similarity(interaction_matrix)
   ```

2. **Item-Based Collaborative Filtering**:
   - This method recommends products based on the similarity between products. For example, if User A likes Product X, the system will recommend other products that are similar to Product X.
   
   Item similarity is also measured using **cosine similarity**:
   ```python
   item_similarity = cosine_similarity(interaction_matrix.T)
   ```

3. **Matrix Factorization (SVD)**:
   - For a more advanced approach, we use **Singular Value Decomposition (SVD)** for matrix factorization, which reduces the dimensionality of the interaction matrix and finds hidden patterns (latent factors).
   - Using the **`surprise` library**, SVD is applied to model the data and predict ratings.
   
   ```python
   from surprise import SVD, Dataset, Reader
   from surprise.model_selection import train_test_split

   reader = Reader(rating_scale=(1, 5))
   data = Dataset.load_from_df(df[['user_id', 'asin', 'rating']], reader)
   trainset, testset = train_test_split(data, test_size=0.2)

   model = SVD()
   model.fit(trainset)
   ```

### **Evaluation**

To evaluate the performance of the recommendation system, we use the **Root Mean Squared Error (RMSE)** to measure how well the predicted ratings align with the true ratings from the test set.

```python
from surprise import accuracy

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')
```

Additionally, we can generate **Top-N recommendations** for each user to evaluate the model's ability to recommend products accurately.

---

## **Run the Project**

### **Steps to Run the Project**:
1. **Set Up Your Environment**:
   - Follow the [Project Setup](#project-setup) steps to install dependencies and set up the database.

2. **Load the Data**:
   - Run the custom management command to load the product and review data:
     ```bash
     python manage.py load_data
     ```

3. **Run the Server**:
   - Once the data is loaded, start the Django development server:
     ```bash
     python manage.py runserver
     ```

4. **Access the API**:
   - The API for getting product recommendations is accessible at:
     ```bash
     http://127.0.0.1:8000/recommendations/{user_id}/
     ```

---

## **API Documentation**

### **GET /recommendations/{user_id}/**
- **Description**: Get the top product recommendations for a specific user based on collaborative filtering.
- **Parameters**:
  - `user_id`: The ID of the user for whom recommendations are being fetched.
- **Response**: JSON array of recommended products for the user.

Example:
```json
{
  "recommendations": [
    {"asin": "B00YQ6X8EO", "title": "Product A", "price": 19.99},
    {"asin": "B081TJ8YS3", "title": "Product B", "price": 29.99}
  ]
}
```

---

## **Future Improvements**

- **Implement Content-Based Filtering**: Combine CF with Content-Based Filtering using product metadata (e.g., `title`, `features`) for more accurate recommendations.
- **Address Cold Start Problem**: Implement strategies to recommend products to new users or with minimal interaction history.
- **User Feedback**: Allow users to provide feedback on recommendations to further refine the system.
- **Scalability**: Optimize for large-scale data (e.g., using distributed processing for larger datasets).

---

This **README** should provide a comprehensive overview of your **Intelligent E-commerce Recommendation System**. Feel free to modify and extend it as needed!