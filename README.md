# Multimodal Emotion Recognition System

## Overview

This project implements a Multimodal Emotion Recognition System using:

* Speech-only emotion recognition
* Text-only emotion recognition
* Fusion-based multimodal emotion recognition

The system combines speech embeddings and contextual text embeddings for improved emotional understanding.

---

# Technologies Used

* Python
* PyTorch
* Transformers (DistilBERT)
* LSTM
* Transformer Encoder
* Attention Pooling
* Scikit-learn
* Matplotlib
* Seaborn

---

# Dataset

TESS (Toronto Emotional Speech Set)

Emotions:

* angry
* disgust
* fear
* happy
* neutral
* sad
* surprise

---

# Project Structure

EmotionRecogProject/
│
├── dataset/
│   └── tess.zip
│
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   │
│   ├── text_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   │
│   └── fusion_pipeline/
│       ├── train.py
│       └── test.py
│
├── Results/
│   ├── accuracy_tables/
│   └── plots/
│
├── README.md
└── requirements.txt

---

# Speech Pipeline

Architecture:

* Speech embeddings
* BiLSTM
* Transformer Encoder
* Attention Pooling
* Fully Connected Classifier

Outputs:

* Accuracy report
* Confusion matrix
* t-SNE visualization

---

# Text Pipeline

Architecture:

* Cleaned and tokenized text
* DistilBERT contextual embeddings
* CLS token representation
* Classification head

Outputs:

* Accuracy report
* Confusion matrix
* t-SNE visualization

---

# Fusion Pipeline

Fusion combines:

* Speech embeddings
* Text embeddings

Fusion classifier:

* Fully connected neural network

Outputs:

* Accuracy report
* Confusion matrix
* t-SNE visualization

---

# Evaluator Instructions

GPU runtime is recommended in Google Colab.

---

# Speech-Only Pipeline

## Steps

1. Open Google Colab

2. Enable GPU:
   Runtime → Change Runtime Type → GPU

3. Copy entire:
   speech_pipeline/test.py

4. Paste into Colab

5. Run all cells

The script automatically:

* downloads pretrained model
* downloads embeddings
* generates results

Outputs:

* accuracy report
* confusion matrix
* t-SNE plot

---

# Fusion Pipeline

## Steps

1. Open Google Colab

2. Enable GPU

3. Copy entire:
   fusion_pipeline/test.py

4. Paste into Colab

5. Run all cells

The script automatically:

* downloads pretrained model
* downloads embeddings
* generates plots/results

Outputs:

* fusion accuracy report
* confusion matrix
* t-SNE visualization

---

# Text-Only Pipeline

## IMPORTANT

Before execution in Google Colab run:

pip install --upgrade torch torchvision torchaudio

pip uninstall torchvision -y

---

## Steps

1. Open Google Colab

2. Enable GPU

3. Run the two pip commands above

4. Copy entire:
   text_pipeline/train.py

5. Paste into Colab and run

6. After training completes:
   copy entire text_pipeline/test.py

7. Paste into new Colab cell and run

Outputs:

* text accuracy report
* confusion matrix
* t-SNE plot

---

# Results

All generated outputs are saved inside:

Results/

* accuracy_tables/
* plots/

---

# Author

Bhuvana Teja Kotti

B.Tech CSE (Data Science)

Hyderabad Institute of Technology and Management
