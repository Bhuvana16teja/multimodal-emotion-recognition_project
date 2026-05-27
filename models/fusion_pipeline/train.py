import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import gdown

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==================================================
# PATHS
# ==================================================
checkpoint_path = "checkpoints"
model_save_path = "models/fusion_pipeline"
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)

# ==================================================
# DOWNLOAD FUSION FILES (auto from Google Drive)
# ==================================================
files = {
    "fusion_train_speech_embeddings.pt": "1idCizCcBI7uApPbl4FytoOAFrcr1zEm6",
    "fusion_test_speech_embeddings.pt": "1BrRKgkbFFElAbgV_LfR2_48S0eSqswe1",
    "fusion_train_text_embeddings.pt": "1QLq4IjYt_6dMqW54NY8_ptIeBZvJVXy0",
    "fusion_test_text_embeddings.pt": "1CkooL4yli5r1XGL6O2_vvAjPg4GD2e3e",
    "fusion_train_labels.pt": "1Nf3GZdSHf7x8_cMjz5KOVPxRJrwPywDN",
    "fusion_test_labels.pt": "1czavHL7Flci7MD4IfSrEQOZJQxWcMWYr"
}

for fname, fid in files.items():
    fpath = os.path.join(checkpoint_path, fname)
    if not os.path.exists(fpath):
        gdown.download(f"https://drive.google.com/uc?id={fid}", fpath, quiet=False)

# ==================================================
# LOAD EMBEDDINGS & LABELS
# ==================================================
train_speech_embeddings = torch.load(os.path.join(checkpoint_path, "fusion_train_speech_embeddings.pt"))
test_speech_embeddings  = torch.load(os.path.join(checkpoint_path, "fusion_test_speech_embeddings.pt"))

train_text_embeddings   = torch.load(os.path.join(checkpoint_path, "fusion_train_text_embeddings.pt"))
test_text_embeddings    = torch.load(os.path.join(checkpoint_path, "fusion_test_text_embeddings.pt"))

train_labels = torch.load(os.path.join(checkpoint_path, "fusion_train_labels.pt"))
test_labels  = torch.load(os.path.join(checkpoint_path, "fusion_test_labels.pt"))

# ==================================================
# CONCATENATE SPEECH + TEXT
# ==================================================
fusion_train = torch.cat([train_speech_embeddings, train_text_embeddings], dim=1)
fusion_test  = torch.cat([test_speech_embeddings, test_text_embeddings], dim=1)

print("Fusion train shape:", fusion_train.shape)
print("Fusion test shape:", fusion_test.shape)

# ==================================================
# DATALOADER
# ==================================================
train_dataset = TensorDataset(fusion_train, train_labels)
test_dataset  = TensorDataset(fusion_test, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==================================================
# MODEL
# ==================================================
class FusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 7)
        )
    def forward(self, x):
        return self.network(x)

# ==================================================
# TRAINING
# ==================================================
model = FusionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 25
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # ===============================
    # VALIDATION
    # ===============================
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch_y.numpy())

    acc = accuracy_score(labels, preds)
    print("Accuracy:", acc)

    # ===============================
    # SAVE BEST MODEL
    # ===============================
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(model_save_path, "best_fusion_model.pth"))
        print("Best model saved!")

print("Best Accuracy:", best_acc)
