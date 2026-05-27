
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset

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
# LOAD DATASET
# ==================================================
main_folder = None

for f in os.listdir("dataset"):
    if "TESS Toronto emotional speech set data" in f:
        main_folder = os.path.join("dataset", f)
        break

if main_folder is None:
    raise FileNotFoundError("Dataset folder not found.")

print("Using dataset folder:", main_folder)

# ==================================================
# CREATE DATAFRAME
# ==================================================
data = []

for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)

    if os.path.isdir(folder_path):

        emotion = folder.split("_")[-1].lower()

        if emotion == "surprised":
            emotion = "surprise"

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                word = file.split("_")[1].lower()

                data.append([word, emotion])

df = pd.DataFrame(data, columns=["text", "emotion"])

# ==================================================
# LABEL ENCODING
# ==================================================
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["emotion"])

# ==================================================
# TRAIN TEST SPLIT
# ==================================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["emotion"],
    random_state=42
)

test_df = test_df.reset_index(drop=True)

# ==================================================
# LOAD TOKENIZER & MODEL
# ==================================================
model_path = "models/text_pipeline/best_text_model"

tokenizer = DistilBertTokenizer.from_pretrained(model_path)

model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

print("Best text model loaded successfully!")

# ==================================================
# TOKENIZE TEST DATA
# ==================================================
encodings = tokenizer(
    test_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=16,
    return_tensors="pt"
)

input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)

labels = test_df["label"].tolist()

# ==================================================
# PREDICTIONS
# ==================================================
with torch.no_grad():

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    logits = outputs.logits

preds = torch.argmax(logits, dim=1).cpu().numpy()

# ==================================================
# ACCURACY
# ==================================================
accuracy = accuracy_score(labels, preds)

print("Test Accuracy:", accuracy)

# ==================================================
# CLASSIFICATION REPORT
# ==================================================
report = classification_report(labels, preds, zero_division=0)

print(report)

accuracy_path = "Results/accuracy_tables/text_results.txt"

with open(accuracy_path, "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

print("Accuracy report saved!")

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

plt.close()

print("Confusion matrix saved!")

# ==================================================
# EXTRACT CLS EMBEDDINGS
# ==================================================
with torch.no_grad():

    hidden_states = model.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

cls_embeddings = hidden_states.last_hidden_state[:,0,:].cpu().numpy()

# ==================================================
# TSNE
# ==================================================
tsne = TSNE(n_components=2, random_state=42)

reduced = tsne.fit_transform(cls_embeddings)

plt.figure(figsize=(10,8))

scatter = plt.scatter(
    reduced[:,0],
    reduced[:,1],
    c=labels,
    cmap="tab10"
)

legend1 = plt.legend(
    *scatter.legend_elements(),
    title="Emotions"
)

plt.gca().add_artist(legend1)

plt.title("Text Emotion t-SNE Visualization")

tsne_path = "Results/plots/text_tsne.png"

plt.savefig(tsne_path, bbox_inches="tight", dpi=300)

plt.close()

print("t-SNE plot saved!")

print("\nText pipeline testing completed successfully!")
