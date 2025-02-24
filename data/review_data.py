import json
import pandas as pd

# Load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load product & review data
products_data = load_jsonl("meta_All_Beauty.jsonl")
reviews_data = load_jsonl("All_Beauty.jsonl")

# Print first review sample
print(reviews_data[:3])  # Print first 3 reviews

# Convert list of reviews to DataFrame
df_reviews = pd.DataFrame(reviews_data)

# Display structure of the data
print(df_reviews.info())
print(df_reviews.head())

print(df_reviews["rating"].value_counts())
