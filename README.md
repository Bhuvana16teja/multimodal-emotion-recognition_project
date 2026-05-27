# Multimodal Emotion Recognition System

## Overview

This project implements a **Multimodal Emotion Recognition System** using:

* Speech-only emotion recognition
* Text-only emotion recognition
* Fusion-based multimodal emotion recognition

The system combines **speech embeddings** and **contextual text embeddings** to improve emotional understanding and classification performance.

---

# Key Features

* Speech emotion recognition using temporal modelling
* Text emotion recognition using contextual NLP embeddings
* Multimodal fusion of speech and text representations
* Attention-based feature aggregation
* t-SNE visualization of learned emotional representations
* Confusion matrix and classification reports
* GPU-supported execution in Google Colab

---

# Dataset

Dataset Used:

**TESS (Toronto Emotional Speech Set)**

Emotion Classes:

* angry
* disgust
* fear
* happy
* neutral
* sad
* surprise

---

# Unseen Speaker Split Strategy

To obtain more reliable and generalized evaluation results, the project follows an **unseen speaker split strategy**.

Training and testing samples were separated carefully so that the model evaluates on speaker-independent emotional representations rather than memorizing speaker-specific characteristics.

This helps:

* reduce overfitting
* improve generalization
* achieve realistic evaluation performance
* validate robustness of multimodal learning

---

# Technologies Used

* Python
* PyTorch
* Transformers (DistilBERT)
* BiLSTM
* Transformer Encoder
* Attention Pooling
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab

---

# Project Structure

```text
EmotionRecogProject/
│
├── models/
│   │
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
│   │
│   ├── accuracy_tables/
│   │
│   └── plots/
│
├── README.md
│
└── requirements.txt
```

---

# Speech-Only Pipeline

## Architecture

Speech embeddings → BiLSTM → Transformer Encoder → Attention Pooling → Fully Connected Classifier

## Outputs

* Accuracy report
* Classification report
* Confusion matrix
* t-SNE visualization

---

# Text-Only Pipeline

## Architecture

Cleaned text → Tokenization → DistilBERT → Contextual embeddings → Classification Head
## NLP Workflow

The text pipeline follows a real NLP-based contextual learning approach:

1. Text preprocessing
2. Tokenization
3. Token-level contextual embedding extraction
4. Transformer-based contextual learning
5. Emotion classification

## Outputs

* Accuracy report
* Confusion matrix
* t-SNE visualization

---

# Fusion Pipeline

## Fusion Strategy

The fusion model combines:

* Speech embeddings
* Text embeddings

using feature concatenation followed by a deep neural classifier.

## Architecture

Speech embeddings + Text embeddings → Fusion Network → Emotion Classification

## Outputs

* Fusion accuracy report
* Confusion matrix
* t-SNE visualization

---

# Evaluator Instructions

GPU runtime is recommended in Google Colab for faster execution.

---

# Speech-Only Pipeline Execution

## Steps

1. Open Google Colab

2. Enable GPU:
   Runtime → Change Runtime Type → GPU

3. Copy entire:
   `speech_pipeline/test.py`

4. Paste into Colab

5. Run all cells

The script automatically:

* downloads pretrained model
* downloads embeddings
* generates plots and reports

---

# Fusion Pipeline Execution

## Steps

1. Open Google Colab

2. Enable GPU

3. Copy entire:
   `fusion_pipeline/test.py`

4. Paste into Colab

5. Run all cells

The script automatically:

* downloads pretrained model
* downloads embeddings
* generates evaluation results

---

# Text-Only Pipeline Execution

## IMPORTANT

Before execution in Google Colab run:


```python
pip uninstall torchvision -y
```

---

## Steps

1. Open Google Colab

2. Enable GPU

3. Run the two pip commands above

4. Copy entire:
   `text_pipeline/train.py`

5. Paste into Colab and run

6. After training completes:

   Copy entire:
   `text_pipeline/test.py`

7. Run test.py

---

# Results

All generated outputs are stored inside:

```text
Results/
│
├── accuracy_tables/
│
└── plots/
```

Generated outputs include:

* Accuracy reports
* Classification reports
* Confusion matrices
* t-SNE visualizations

---

# Author

**Bhuvana Teja Kotti**

B.Tech – Computer Science and Engineering (Data Science)

Hyderabad Institute of Technology and Management
