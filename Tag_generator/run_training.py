from src.data_preprocessing import preprocess_data
from src.train import train_model
import os
import torch
import json
import random

# ✅ Optional: Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# ✅ Load and preprocess data
dataset_path = "data/dsa_python_1000.json"
dataset, tag_classes = preprocess_data(dataset_path)
print(f"\n✅ Total Samples: {len(dataset)}")
print(f"✅ Number of Unique Tags: {len(tag_classes)}")

# ✅ Train model
model = train_model(
    dataset=dataset,
    num_labels=len(tag_classes),
    epochs=3,
    batch_size=8,
    lr=2e-5
)

# ✅ Create model directory
model_dir = "models/tag_model"
os.makedirs(model_dir, exist_ok=True)

# ✅ Save model
model_path = os.path.join(model_dir, "pytorch_model.bin")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")

# ✅ Save tag classes
tag_class_path = os.path.join(model_dir, "tag_classes.json")
with open(tag_class_path, "w", encoding="utf-8") as f:
    json.dump(tag_classes, f, indent=2)
print(f"✅ Tag classes saved to: {tag_class_path}")
