import json
import torch
from sentence_transformers import SentenceTransformer, util

# ✅ Paths (relative to your current folder: E:/Microservice_turtil/Tag_generator)
MODEL_PATH = "../Similairty_check/models/all-MiniLM-L6-v2"
TAGS_PATH = "models/tag_model/tag_classes.json"

# ✅ Load the SentenceTransformer model
model = SentenceTransformer(MODEL_PATH)

# ✅ Load tag list
with open(TAGS_PATH, "r", encoding="utf-8") as f:
    tags = json.load(f)

# ✅ Encode all tags once
tag_embeddings = model.encode(tags, convert_to_tensor=True)

print("\n📝 Enter your question/body (or type 'exit' to quit):", end=' ')
while True:
    query = input().strip()
    if query.lower() == 'exit':
        break

    # ✅ Encode user query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # ✅ Compute cosine similarities
    cosine_scores = util.cos_sim(query_embedding, tag_embeddings)[0]

    # ✅ Top 5 tags with threshold
    threshold = 0.4
    top_results = torch.topk(cosine_scores, k=5)
    top_tags = []
    
    print("\n📊 Tag Similarities (above threshold):")
    for score, idx in zip(top_results.values, top_results.indices):
        if score >= threshold:
            print(f"{tags[idx]:<20}: {score.item():.4f}")
            top_tags.append(tags[idx])

    print(f"\n🏷️ Predicted tags: {top_tags}\n")
    print("📝 Enter your question/body (or type 'exit' to quit):", end=' ')
