from huggingface_hub import snapshot_download

# Download model from Hugging Face to local folder
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/all-MiniLM-L6-v2",
    local_dir_use_symlinks=False  # Important for Windows compatibility
)

print("Model downloaded successfully to models/all-MiniLM-L6-v2/")
