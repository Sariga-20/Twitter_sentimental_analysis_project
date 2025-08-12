# Twitter Sentiment Analysis

## 🚀 Project Overview

This repository contains an end-to-end Twitter sentiment analysis system utilizing both Logistic Regression and LSTM (Long Short-Term Memory) Neural Networks. The project demonstrates the practical differences between traditional ML and deep learning approaches for text classification, comparing their confidence scores and outputs for challenging, mixed-sentiment samples.

---

## 🗂️ Directory Structure

```

├── data/               # Datasets for training and evaluation
├── saved_models/       # Checkpoints and serialized model files
├── app.py              # Main app script for running inference
├── project_code.ipynb  # Development notebook (EDA, training, demo)
├── requirements.txt    # Libraries and Python dependencies
├── .gitignore          # Version control settings

```


---

## 🔧 Setup Instructions

1. **Clone the repository**

git clone


2. **Install dependencies**

pip install -r requirements.txt


## 📂 Dataset

This project uses the [Sentiment140](https://www.kaggle.com/kazanova/sentiment140) Twitter dataset for sentiment analysis.

- **Filename:** `training.1600000.processed.noemoticon.csv`
- **Format:** CSV, 6 columns (`target`, `id`, `date`, `flag`, `user`, `text`)
- **Size:** 1,600,000 tweets annotated for sentiment (0 = negative, 2 = neutral, 4 = positive)

Place the dataset CSV file inside the `data/` directory:

---

## 💡 How to Run

### **Jupyter Notebook**
Open and run `project_code.ipynb` for:
- Data preprocessing
- Model training & evaluation
- Sample predictions and comparison

---

## ⚙️ Models Used

- **Logistic Regression**: Bag-of-words & TF-IDF features, scikit-learn
- **LSTM Neural Network**: Embeddings & sequential processing, TensorFlow/Keras

### Key Technical Insights:
- Logistic regression is fast, interpretable, and often strong on classical datasets.
- LSTM captures deeper semantic and sequential relationships, but can struggle (low confidence) with mixed sentiment phrasing.

---

## 📈 Results & Discussion

This project highlights how differently simple linear models and deep neural architectures handle contradictory sentiment in the same tweet. For neutral/mixed statements, logistic regression usually delivers higher confidence due to word-level weighting, while LSTM reflects sequential confusion and uncertainty with lower confidence.

---

## 🌐 Live Demo

Try the Twitter Sentiment Analysis App here:  
[Open in Streamlit](https://twittersentimentalanalysisproject-h7cirjlxfr6rgb7vuxrappe.streamlit.app/)

---

## 🔬 Future Enhancement

- Add model ensembling or BERT-based transformers for improved performance
- Use more granular sentiment categories (positive/negative/mixed/irrelevant)
- Integrate a frontend API for real-time inference via Flask or FastAPI

---

> **Tip:** Update repository URL, contact email, and any dataset/source links as per your preferences.


