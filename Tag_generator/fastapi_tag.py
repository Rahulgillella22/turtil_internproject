from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import json
import os

# ✅ Create the router instance
router = APIRouter()

# ✅ Define paths to model and tag JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "Similairty_check", "models", "all-MiniLM-L6-v2")
TAGS_PATH = os.path.join(BASE_DIR, "models", "tag_model", "tag_classes.json")

# ✅ Load model
try:
    model = SentenceTransformer(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# ✅ Load tags
try:
    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        tags = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load tags from {TAGS_PATH}: {e}")

# ✅ Precompute tag embeddings
tag_embeddings = model.encode(tags, convert_to_tensor=True)

# ✅ Request body schema
class TagRequest(BaseModel):
    text: str

# ✅ POST endpoint for tag prediction
@router.post("/generate-tags")
async def generate_tags(request: TagRequest):
    try:
        query_embedding = model.encode(request.text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, tag_embeddings)[0]

        threshold = 0.30  # 🔽 Lowered to capture more tags but still relevant
        top_results = torch.topk(cosine_scores, k=10)  # 🔼 Consider more tags

        top_tags = []
        for score, idx in zip(top_results.values, top_results.indices):
            if score >= threshold:
                top_tags.append(tags[idx])

        # 🔄 Ensure minimum 2 relevant tags returned if possible
        if len(top_tags) < 2:
            # Take top 2 tags regardless of threshold
            sorted_indices = torch.argsort(cosine_scores, descending=True)
            for idx in sorted_indices[:2]:
                tag = tags[idx]
                if tag not in top_tags:
                    top_tags.append(tag)

        return {"predicted_tags": top_tags}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
