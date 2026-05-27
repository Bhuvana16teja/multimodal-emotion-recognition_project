import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==================================================
# LOAD EMBEDDINGS & LABELS
# ==================================================
padded_embeddings = torch.load("checkpoints/train_embeddings.pt")
labels_tensor = torch.load("checkpoints/train_labels.pt")

print("Embeddings shape:", padded_embeddings.shape)
print("Labels shape:", labels_tensor.shape)

# ==================================================
# DATASET & DATALOADER
# ==================================================
dataset = TensorDataset(padded_embeddings, labels_tensor)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ==================================================
# MODEL DEFINITIONS
# ==================================================
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled

class EmotionModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=8,
                                                   dim_feedforward=512, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attention_pool = AttentionPooling(hidden_size*2)
        self.fc1 = nn.Linear(hidden_size*2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        trans_out = self.transformer(lstm_out)
        pooled = self.attention_pool(trans_out)
        x = self.fc1(pooled)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ==================================================
# TRAINING LOOP
# ==================================================
model = EmotionModel().to(device)
alpha = torch.tensor([2.5,1.0,1.0,3.0,0.5,1.2,1.0]).to(device)
criterion = FocalLoss(alpha=alpha)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    scheduler.step(epoch)

# ==================================================
# SAVE TRAINED MODEL
# ==================================================
os.makedirs("models/speech_pipeline", exist_ok=True)
torch.save(model.state_dict(), "models/speech_pipeline/best_model.pth")
print("Training completed!")
