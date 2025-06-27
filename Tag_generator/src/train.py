import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from src.model import TagPredictionModel
import numpy as np

# ✅ Compute class-wise positive weights for BCEWithLogitsLoss
def compute_pos_weights(dataset, num_labels):
    counts = np.zeros(num_labels)
    for sample in dataset:
        counts += sample["labels"].numpy()  # Assumes labels are tensors
    pos_weights = (len(dataset) - counts) / (counts + 1e-5)  # Avoid div by zero
    return torch.tensor(pos_weights, dtype=torch.float32)

# ✅ Main training function
def train_model(dataset, num_labels, epochs=3, batch_size=8, lr=2e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TagPredictionModel(num_labels)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    # ✅ Use weighted loss to handle tag imbalance
    pos_weights = compute_pos_weights(dataset, num_labels)
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    # ✅ Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model
