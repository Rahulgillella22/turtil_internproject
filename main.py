from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# ✅ Import routers
from Similairty_check.fastapi_search import router as similarity_router
from Tag_generator.fastapi_tag import router as tag_router

# ✅ Create FastAPI app
app = FastAPI(
    title="Post Similarity & Tagging API",
    description="This API provides endpoints to detect similar posts using FAISS and suggest tags using sentence-transformers.",
    version="1.0.0"
)

# ✅ Include routers
app.include_router(similarity_router, tags=["Similarity"], prefix="/similarity")
app.include_router(tag_router, tags=["Tagging"], prefix="/tags")

# ✅ Mount static folder (frontend)
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(current_dir, "frontend")
app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

# ✅ Serve index.html at root
@app.get("/")
async def serve_home():
    return FileResponse(os.path.join(frontend_path, "indexi.html"))
