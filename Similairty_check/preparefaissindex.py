#make the pickle file and indexing file 
import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

# Set relative paths
data_path = os.path.join("data", "cleaned_posts.json")
model_path = os.path.join("models", "all-MiniLM-L6-v2")

# Load posts
with open(data_path, "r", encoding="utf-8") as f:
    posts = json.load(f)

# Prepare text data
texts = [post["post_title"] + " " + post["post_body"] for post in posts]

# Load local model
model = SentenceTransformer(model_path)

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index.index")

# Save post mapping
with open("post_mapping.pkl", "wb") as f:
    pickle.dump(posts, f)

print("✅ FAISS index and post mapping saved successfully.")
