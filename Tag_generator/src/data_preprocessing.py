import json
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset

# ✅ Custom Dataset for PyTorch
class TaggingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]  # Already a tensor
        return item

    def __len__(self):
        return len(self.labels)

# ✅ Data preprocessing function
def preprocess_data(json_path, tokenizer_name="bert-base-uncased", max_length=128):
    # Load data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare input text (title + body)
    questions = [item["title"] + " " + item["body"] for item in data]

    # Create sorted list of all unique tags
    all_tags = sorted({tag for item in data for tag in item["tags"]})
    tag2id = {tag: idx for idx, tag in enumerate(all_tags)}

    # Create multi-hot encoded labels
    labels = []
    for item in data:
        label = [0] * len(all_tags)
        for tag in item["tags"]:
            if tag in tag2id:
                label[tag2id[tag]] = 1
        labels.append(label)

    # Tokenize using BERT
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # ✅ Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.float32)

    # Create dataset
    dataset = TaggingDataset(encodings, labels)
    return dataset, all_tags
