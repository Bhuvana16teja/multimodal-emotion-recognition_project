import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import gdown

# =====================================================
# CREATE RESULT FOLDERS
# =====================================================
os.makedirs("Results/plots", exist_ok=True)
os.makedirs("Results/accuracy_tables", exist_ok=True)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# PUBLIC DRIVE FILES
# =====================================================
TEST_EMB_ID   = "1LE9cr4R2gogfh0_5Cja4k08zXLcOGjjX"
TEST_LABEL_ID = "1rSbwxM9L_Dp8LA8X1dvGg_RYeaoHqk8a"
MODEL_ID      = "1i9vO_yxtZUxA57FtrH22c-S3bODWLCDf"

# Download files if not present
os.makedirs("checkpoints", exist_ok=True)
if not os.path.exists("checkpoints/test_embeddings.pt"):
    gdown.download(f"https://drive.google.com/uc?id={TEST_EMB_ID}",
                   "checkpoints/test_embeddings.pt", quiet=False)
if not os.path.exists("checkpoints/test_labels.pt"):
    gdown.download(f"https://drive.google.com/uc?id={TEST_LABEL_ID}",
                   "checkpoints/test_labels.pt", quiet=False)
os.makedirs("models/speech_pipeline", exist_ok=True)
if not os.path.exists("models/speech_pipeline/best_model.pth"):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}",
                   "models/speech_pipeline/best_model.pth", quiet=False)

# =====================================================
# LOAD TEST EMBEDDINGS
# =====================================================
test_padded = torch.load("checkpoints/test_embeddings.pt")
test_labels_tensor = torch.load("checkpoints/test_labels.pt")
print("Test embeddings loaded!")

# =====================================================
# EMOTION LABELS
# =====================================================
emotion_names = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# =====================================================
# ATTENTION POOLING
# =====================================================
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled

# =====================================================
# MODEL
# =====================================================
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

# =====================================================
# LOAD MODEL
# =====================================================
model = EmotionModel().to(device)
model.load_state_dict(torch.load("models/speech_pipeline/best_model.pth"))
model.eval()
print("Model loaded successfully!")

# =====================================================
# PREDICTIONS
# =====================================================
all_preds = []
with torch.no_grad():
    for i in tqdm(range(len(test_padded))):
        sample = test_padded[i].unsqueeze(0).to(device)
        outputs = model(sample)
        pred = torch.argmax(outputs, dim=1).item()
        all_preds.append(pred)

# =====================================================
# ACCURACY & REPORT
# =====================================================
accuracy = accuracy_score(test_labels_tensor.numpy(), all_preds)
print("\nTest Accuracy:", accuracy)

report = classification_report(test_labels_tensor.numpy(), all_preds, target_names=emotion_names)
print(report)

accuracy_path = "Results/accuracy_tables/speech_accuracy_report.txt"
with open(accuracy_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy}\n\n")
    f.write(report)
print(f"Accuracy report saved at:\n{accuracy_path}")

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(test_labels_tensor.numpy(), all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=emotion_names, yticklabels=emotion_names)
plt.title("Speech Emotion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
conf_matrix_path = "Results/plots/speech_confusion_matrix.png"
plt.savefig(conf_matrix_path, dpi=300, bbox_inches="tight")
print(f"Confusion matrix saved at:\n{conf_matrix_path}")
plt.close()

# =====================================================
# TEMPORAL EMBEDDINGS + TSNE
# =====================================================
temporal_embeddings = []
with torch.no_grad():
    for i in tqdm(range(len(test_padded))):
        sample = test_padded[i].unsqueeze(0).to(device)
        lstm_out, _ = model.lstm(sample)
        trans_out = model.transformer(lstm_out)
        pooled = model.attention_pool(trans_out)
        temporal_embeddings.append(pooled.squeeze(0).cpu())
temporal_embeddings = torch.stack(temporal_embeddings)
print("Temporal embeddings extracted!")

tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(temporal_embeddings.numpy())
plt.figure(figsize=(10,8))
scatter = plt.scatter(reduced[:,0], reduced[:,1],
                      c=test_labels_tensor.numpy(), cmap="tab10")
legend1 = plt.legend(*scatter.legend_elements(), title="Emotions")
plt.gca().add_artist(legend1)
plt.title("Emotion Cluster Visualization (Temporal Modelling Block)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
tsne_path = "Results/plots/temporal_tsne.png"
plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
print(f"t-SNE plot saved at:\n{tsne_path}")
plt.close()

print("\nSpeech pipeline testing completed successfully!")
