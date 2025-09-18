---

# 🗨️ Sentiment Analysis on Reddit Gaming Comments

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep--Learning-orange?logo=tensorflow)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployment-yellow?logo=huggingface)

## 📌 Project Overview

This project develops a **deep learning-based sentiment classifier** for Reddit comments collected from **r/gaming**, with the goal of monitoring and interpreting community sentiment. The model classifies comments into:

* **Positive** 🎉
* **Neutral** 😐
* **Negative** 😡

The outcome equips **game developers, publishers, and community managers** with a scalable tool to **track sentiment at scale**, enabling proactive brand management and community engagement.

🔗 **Links & References**

* [Dataset on Kaggle](#) – 📊 Social Media Comments on Gaming Dataset
* [Dataset Source: r/gaming](#) – Reddit community reference
* [Deployed Dashboard](https://huggingface.co/spaces/iankuzuma/Reddit_Gaming_Comment_Sentiment_Prediction_APP) – 🌐 Public Hugging Face App
* [Saved Model Files](#) – 📂 Google Drive

---

## 📊 Dataset Breakdown

### 📝 Overview

The dataset contains **23,000+ Reddit comments** from the r/gaming subreddit, each annotated with sentiment labels. It reflects the **organic, real-world tone of gamer discussions**, including sarcasm, slang, and abbreviations.

### 📂 Columns

| Column      | Description                                                       |
| ----------- | ----------------------------------------------------------------- |
| `Comment`   | User-submitted Reddit comment (including \[deleted] / \[removed]) |
| `sentiment` | Target label: Positive / Neutral / Negative                       |

---

## 🎯 Project Objectives

The goal is to **build, train, and evaluate** a sentiment classification model that supports:

* **Positive Sentiment → Amplify advocacy**

  * Highlight praised features in campaigns
  * Reward advocates with visibility
  * Strengthen community loyalty

* **Neutral Sentiment → Unlock hidden feedback**

  * Analyze “passive” tone for friction points
  * Push targeted surveys / feedback loops
  * Track shifts post-update

* **Negative Sentiment → Mitigate backlash**

  * Early alerting for dissatisfaction spikes
  * Correlate with patches or monetization events
  * Support moderation / crisis response

---

## 🧠 Model Architecture

This project implements a **hybrid NLP architecture**:

* **BERT (Pretrained Tokenizer & Embeddings):** Captures semantic context, slang, abbreviations.
* **BiLSTM Layers:** Preserve sequential flow and nuanced tone.

### ⚙️ Training Pipeline

* **Optimizer:** Adam with LR scheduling
* **Regularization:** Dropout within LSTM
* **Callback:** EarlyStopping on validation loss
* **Split Strategy:** Stratified train-validation split

---

## 📏 Evaluation Metrics

* **Macro F1-Score (Primary Metric):** Ensures equal importance across all sentiment classes.
* **Classification Report:** Precision, recall, F1 per class.
* **Confusion Matrix:** Diagnose misclassifications (e.g., sarcasm misread as negativity).

---

## 🚀 Deployment

The model is deployed via a **Hugging Face dashboard**, allowing users to:

* Input Reddit-style text
* Get real-time sentiment predictions (positive/neutral/negative)
* Explore confidence scores and model breakdowns

---

## 📂 Repository Structure

```
📦 reddit-gaming-sentiment
 ┣ 📜 streamlit_app.py                                                # Hugging Face deployment files
 ┣ 📜 eda.py                                                          # Hugging Face deployment files
 ┣ 📜 prediction.py                                                   # Hugging Face deployment files
 ┣ 📜 r_gaming_comments_sentiments_dataset.csv                        # Dataset
 ┣ 📜 reddit-gaming-comment-sentiment-prediction.ipynb                # Jupyter notebooks for EDA & model training Source code for preprocessing, modeling, evaluation
 ┣ 📜 reddit-gaming-comment-sentiment-prediction-inference.ipynb      # Jupyter notebooks for model inferencing 
 ┣ 📜 requirements.txt                                                # Dependencies
 ┣ 📜 README.md                                                       # Project documentation Hugging Face + model export scripts
 ┣ 📜 url.txt                                                         # Hugging Face + model export links
```

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Libraries:** TensorFlow/Keras, HuggingFace Transformers, scikit-learn, pandas, numpy
* **Architecture:** BERT embeddings + BiLSTM
* **Deployment:** Hugging Face Spaces

---

## 📈 Results & Insights

* Hybrid **BERT + BiLSTM** significantly outperforms vanilla LSTM.
* **Macro F1-scores** balanced across positive, neutral, and negative classes.
* Sentiment tracking highlights **correlation between update cycles and negativity spikes**, supporting real-world business use.

---

## 🙌 Acknowledgements

* Dataset: [Kaggle – Social Media Comments on Gaming](#)
* Community Source: r/gaming subreddit
* Deployment: Hugging Face

---
