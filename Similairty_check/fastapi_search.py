import os                             #fast api search faiss code 
import faiss
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

# ✅ Define request and response schemas
class PostData(BaseModel):
    post_title: str
    post_body: str

class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool
    similarity_score: float
    matched_post: str  # Make sure to return string, not dict

# ✅ Setup FastAPI router
router = APIRouter()

# ✅ Correct path to config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
print("📂 Using config:", CONFIG_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

SIMILARITY_THRESHOLD = config.get("SIMILARITY_THRESHOLD", 0.8)
TOP_K = config.get("TOP_K", 10)

# ✅ Load FAISS index
INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.index")
index = faiss.read_index(INDEX_PATH)

# ✅ Load post mapping
MAPPING_PATH = os.path.join(os.path.dirname(__file__), "post_mapping.pkl")
with open(MAPPING_PATH, "rb") as f:
    post_mapping = pickle.load(f)

# ✅ Load SentenceTransformer model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_PATH)


@router.post("/check-duplicate", response_model=DuplicateCheckResponse)
def check_duplicate(post: PostData):
    try:
        user_query = f"{post.post_title} {post.post_body}"
        query_embedding = model.encode([user_query]).astype("float32")

        distances, indices = index.search(query_embedding, TOP_K)

        closest_idx = indices[0][0]
        closest_distance = distances[0][0]

        is_duplicate = closest_distance <= SIMILARITY_THRESHOLD
        matched_post = str(post_mapping[closest_idx])  # ✅ Converted to string

        return {
            "is_duplicate": is_duplicate,
            "similarity_score": float(closest_distance),
            "matched_post": matched_post
        }

    except Exception as e:
        print("❌ Error in /check-duplicate:", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
