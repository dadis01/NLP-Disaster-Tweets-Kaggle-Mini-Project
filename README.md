# Disaster Tweets Classification - NLP Kaggle Project

This repository contains my submission and full analysis for the [Kaggle "Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started) competition. The goal is to build a binary classifier that predicts whether a tweet is about a real disaster or not.

---

## Project Overview

- **Problem**: Given a tweet, predict whether it refers to a real disaster (`target=1`) or not (`target=0`).
- **Techniques Used**: Natural Language Processing (NLP), TF-IDF, Logistic Regression, LSTM, Bi-LSTM
- **Tools**: Python, Pandas, Scikit-learn, TensorFlow/Keras, Seaborn, Matplotlib

---

## Files in the Repository

| File | Description |
|------|-------------|
| `Disaster_Tweets_With_BiLSTM.ipynb` | Final notebook with full pipeline (EDA, TF-IDF, LSTM, Bi-LSTM) |
| `submission.csv` | Example output format for Kaggle submission |
| `README.md` | This file |
| `requirements.txt` | List of required libraries (optional) |
| `kaggle_leaderboard.png` | Screenshot of Kaggle leaderboard entry (add your own) |

---

## Dataset Info

- **Training Data**: 7,613 tweets with target labels
- **Test Data**: 3,263 tweets for prediction
- **Features**:
  - `text`: the tweet
  - `keyword`: optional keyword
  - `location`: optional user-provided location
  - `target`: binary label

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Target label distribution
- Tweet length analysis
- Word clouds and top keywords

### 2. Text Preprocessing
- Clean tweets (lowercasing, punctuation removal, etc.)
- Tokenization & padding for LSTM
- TF-IDF for traditional models

### 3. Models Implemented
| Model       | Highlights                        |
|-------------|-----------------------------------|
| Logistic Regression | Fast, interpretable baseline using TF-IDF |
| LSTM | Captures sequential patterns in tweets |
| Bi-LSTM | Uses both forward and backward context |

---

## Results

| Model       | F1 Score (Val) |
|-------------|----------------|
| Logistic Regression | 0.75 |
| LSTM                | ~0.74 |
| Bi-LSTM             | ~0.73 |

F1 Score was used as the evaluation metric, balancing precision and recall for disaster detection.

---

## Future Improvements

- Use pretrained embeddings (e.g., GloVe, BERT)
- Add location/keyword metadata as additional features
- Use ensemble models for better generalization

---


---

## References

- [Kaggle Competition Page](https://www.kaggle.com/competitions/nlp-getting-started)
- TensorFlow & Keras Documentation
- Scikit-learn & NLP tutorials
