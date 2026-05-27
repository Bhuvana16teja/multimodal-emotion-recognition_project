import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import gdown

# =====================================================
# RESULT FOLDERS
# =====================================================
os.makedirs("Results/plots", exist_ok=True)
os.makedirs("Results/accuracy_tables", exist_ok=True)

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# PATHS
# =====================================================
checkpoint_path = "checkpoints"
model_path = "models/fusion_pipeline/best_fusion_model.pth"
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs("models/fusion_pipeline", exist_ok=True)

# =====================================================
# DOWNLOAD REQUIRED FILES FROM DRIVE
# =====================================================
files = {
    "fusion_test_speech_embeddings.pt": "1BrRKgkbFFElAbgV_LfR2_48S0eSqswe1",
    "fusion_test_text_embeddings.pt": "1CkooL4yli5r1XGL6O2_vvAjPg4GD2e3e",
    "fusion_test_labels.pt": "1czavHL7Flci7MD4IfSrEQOZJQxWcMWYr",
    "best_fusion_model.pth": "1dg_dr-_ueyDWs_Jm4WeH-CrXdr7OUBLt"
}

for fname, fid in files.items():
    if fname.endswith(".pt"):
        fpath = os.path.join(checkpoint_path, fname)
    else:
        fpath = model_path
    if not os.path.exists(fpath):
        gdown.download(f"https://drive.google.com/uc?id={fid}", fpath, quiet=False)

# =====================================================
# LOAD TEST EMBEDDINGS & LABELS
# =====================================================
test_speech_embeddings = torch.load(os.path.join(checkpoint_path, "fusion_test_speech_embeddings.pt"))
test_text_embeddings   = torch.load(os.path.join(checkpoint_path, "fusion_test_text_embeddings.pt"))
test_labels_tensor     = torch.load(os.path.join(checkpoint_path, "fusion_test_labels.pt"))
labels = test_labels_tensor.tolist()

fusion_test = torch.cat([test_speech_embeddings, test_text_embeddings], dim=1)
print("Fusion test shape:", fusion_test.shape)

# =====================================================
# EMOTION LABELS
# =====================================================
emotion_names = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# =====================================================
# DATASET
# =====================================================
test_dataset = TensorDataset(fusion_test, test_labels_tensor)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =====================================================
# MODEL
# =====================================================
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

# =====================================================
# LOAD SAVED MODEL
# =====================================================
model = FusionClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Best fusion model loaded successfully!")

# =====================================================
# PREDICTIONS
# =====================================================
preds, labels_eval = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        pred = torch.argmax(outputs, dim=1)
        preds.extend(pred.cpu().numpy())
        labels_eval.extend(batch_y.numpy())

# =====================================================
# ACCURACY & REPORT
# =====================================================
accuracy = accuracy_score(labels_eval, preds)
print("\nFusion Accuracy:", accuracy)

report = classification_report(labels_eval, preds, target_names=emotion_names)
print(report)

accuracy_path = "Results/accuracy_tables/fusion_accuracy_report.txt"
with open(accuracy_path, "w") as f:
    f.write(f"Fusion Accuracy: {accuracy}\n\n")
    f.write(report)
print(f"Accuracy report saved at:\n{accuracy_path}")

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(labels_eval, preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=emotion_names, yticklabels=emotion_names)
plt.title("Fusion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
fusion_cm_path = "Results/plots/fusion_confusion_matrix.png"
plt.savefig(fusion_cm_path, dpi=300, bbox_inches="tight")
print(f"Fusion confusion matrix saved at:\n{fusion_cm_path}")
plt.close()

# =====================================================
# FEATURE EXTRACTION FOR TSNE
# =====================================================
fusion_features = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        features = model.network[:-1](batch_x)  # remove final layer
        fusion_features.append(features.cpu())
fusion_features = torch.cat(fusion_features).numpy()
print("Fusion features extracted!")

# =====================================================
# TSNE VISUALIZATION
# =====================================================
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(fusion_features)
plt.figure(figsize=(10,8))
scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels_eval, cmap="tab10")
legend1 = plt.legend(*scatter.legend_elements(), title="Emotions")
plt.gca().add_artist(legend1)
plt.title("Fusion Representation Emotion Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
fusion_tsne_path = "Results/plots/fusion_tsne.png"
plt.savefig(fusion_tsne_path, dpi=300, bbox_inches="tight")
print(f"Fusion t-SNE saved at:\n{fusion_tsne_path}")
plt.close()

print("\nFusion pipeline evaluation completed successfully!")
