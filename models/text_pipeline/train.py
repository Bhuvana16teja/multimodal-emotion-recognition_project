import os
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset
import gdown, zipfile

# ==================================================
# PROJECT PATHS
# ==================================================
os.makedirs("dataset", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("models/text_pipeline", exist_ok=True)
os.makedirs("Results/logs", exist_ok=True)

# ==================================================
# PUBLIC DRIVE FILES
# ==================================================
DATASET_ID = "1F3ZSXpEifKW5brudmFKJ0lohcDS4X-Iz"   # tess.zip
FUSION_EMB_ID = "1QLq4IjYt_6dMqW54NY8_ptIeBZvJVXy0" # fusion_train_text_embeddings.pt

# === DOWNLOAD DATASET ===
if not os.path.exists("dataset.zip"):
    gdown.download(f"https://drive.google.com/uc?id={DATASET_ID}", "dataset.zip", quiet=False)
    with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")

# Auto-detect the correct TESS folder
main_folder = None
for f in os.listdir("dataset"):
    if "TESS Toronto emotional speech set data" in f:
        main_folder = os.path.join("dataset", f)
        break
if main_folder is None:
    raise FileNotFoundError("Could not find TESS dataset folder after extraction.")

print("Using dataset folder:", main_folder)

# === DOWNLOAD FUSION EMBEDDINGS ===
if not os.path.exists("checkpoints/fusion_train_text_embeddings.pt"):
    gdown.download(f"https://drive.google.com/uc?id={FUSION_EMB_ID}",
                   "checkpoints/fusion_train_text_embeddings.pt", quiet=False)

# ==================================================
# DATASET CREATION
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
df["text"] = df["text"].apply(lambda t: t.lower().strip())

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["emotion"])

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["emotion"],
    random_state=42
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=16
    )

train_dataset = train_dataset.map(tokenize_function)
test_dataset = test_dataset.map(tokenize_function)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model_path = "models/text_pipeline/best_text_model"
if os.path.exists(model_path):
    print("Loading pretrained text model...")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
else:
    print("Training new text model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=7
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# ==================================================
# TRAINING
# ==================================================
training_args = TrainingArguments(
    output_dir="models/text_pipeline",
    eval_strategy="epoch",   # <-- use this instead of evaluation_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="Results/logs",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

print("Best text model saved at:", model_path)
