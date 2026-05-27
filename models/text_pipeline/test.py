import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ==================================================
# PROJECT PATHS
# ==================================================
os.makedirs("Results/plots", exist_ok=True)
os.makedirs("Results/accuracy_tables", exist_ok=True)

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==================================================
# LOAD TEST EMBEDDINGS & LABELS
# ==================================================
test_embeddings = torch.load("checkpoints/fusion_test_text_embeddings.pt")
test_labels_tensor = torch.load("checkpoints/fusion_test_labels.pt")
labels = test_labels_tensor.tolist()
print("Fusion test embeddings and labels loaded!")

# ==================================================
# LOAD MODEL FROM FOLDER
# ==================================================
model_path = "models/text_pipeline/best_text_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()
print("Best text model loaded successfully!")

# ==================================================
# PREDICTIONS
# ==================================================
preds = []
with torch.no_grad():
    # Here we assume test_embeddings are CLS vectors already
    for emb in test_embeddings:
        emb = emb.unsqueeze(0).to(device)  # batch dimension
        outputs = model.classifier(model.dropout(emb))
        pred = torch.argmax(outputs, dim=1).item()
        preds.append(pred)

# ==================================================
# ACCURACY & REPORT
# ==================================================
accuracy = accuracy_score(labels, preds)
print("Test Accuracy:", accuracy)

report = classification_report(labels, preds, zero_division=0)
print(report)

accuracy_path = "Results/accuracy_tables/text_results.txt"
with open(accuracy_path, "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)
print(f"Accuracy report saved at:\n{accuracy_path}")

# ==================================================
# CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Text Emotion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
conf_matrix_path = "Results/plots/text_confusion_matrix.png"
plt.savefig(conf_matrix_path, bbox_inches="tight", dpi=300)
print(f"Confusion matrix saved at:\n{conf_matrix_path}")
plt.close()

# ==================================================
# TSNE VISUALIZATION
# ==================================================
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(test_embeddings.numpy())
plt.figure(figsize=(10,8))
scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab10")
legend1 = plt.legend(*scatter.legend_elements(), title="Emotions")
plt.gca().add_artist(legend1)
plt.title("Text Emotion Cluster Visualization (CLS Embeddings)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
tsne_path = "Results/plots/text_tsne.png"
plt.savefig(tsne_path, bbox_inches="tight", dpi=300)
print(f"t-SNE plot saved at:\n{tsne_path}")
plt.close()

print("\nText pipeline testing completed successfully!")
