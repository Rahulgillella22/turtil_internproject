#this script will search the similairty through the terminal (in this script i actually used the sentence transformers and faiss indexing)

import faiss
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import sys  # For flushing output

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)

SIMILARITY_THRESHOLD = config.get("SIMILARITY_THRESHOLD", 0.8)  #if its missing then keeps 0.8 as default
TOP_K = config.get("TOP_K", 10)  # Optional, defaults to 10

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Load the post mapping (list of original questions)
with open("post_mapping.pkl", "rb") as f:
    post_mapping = pickle.load(f)

# Load sentence transformer model
#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("models/all-MiniLM-L6-v2")


# Function to find similar posts
def get_similar_posts(user_query):
    query_embedding = model.encode([user_query]).astype("float32")
    distances, indices = index.search(query_embedding, TOP_K)

    # Filter by similarity threshold
    filtered = [
        (idx, distance)
        for idx, distance in zip(indices[0], distances[0])
        if distance <= SIMILARITY_THRESHOLD
    ]

    if not filtered:
        print("❌ No similar questions found within the threshold.")
    else:
        print("\n✅ Similar Questions Found:\n")
        for idx, distance in filtered:
            print(f"➤ {post_mapping[idx]} (distance: {distance:.4f})")

# Run in a loop
if __name__ == "__main__":
    while True:
        user_query = input("\n🔍 Enter your question (or type 'exit' to quit): ")
        sys.stdout.flush()
        if not user_query.strip():
            continue
        if user_query.lower() == "exit":
            break
        get_similar_posts(user_query)
