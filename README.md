# Twitter Sentiment Analysis

## 🚀 Project Overview

This repository contains an end-to-end Twitter sentiment analysis system utilizing both Logistic Regression and LSTM (Long Short-Term Memory) Neural Networks. The project demonstrates the practical differences between traditional ML and deep learning approaches for text classification, comparing their confidence scores and outputs for challenging, mixed-sentiment samples.

---

## 🗂️ Directory Structure

```

├── data/ # Datasets for training and evaluation
├── saved_models/ # Checkpoints and serialized model files
├── app.py # Main app script for running inference
├── project_code.ipynb # Development notebook (EDA, training, demo)
├── requirements.txt # Libraries and Python dependencies
├── .gitignore # Version control settings

```


---

## 🔧 Setup Instructions

1. **Clone the repository**

git clone


2. **Install dependencies**

pip install -r requirements.txt


3. **Download/Prepare dataset**
- Place your CSV or data files inside the `data/` directory.
- Update any data path variables in your scripts accordingly.

---

## 💡 How to Run

### **Jupyter Notebook**
Open and run `project_code.ipynb` for:
- Data preprocessing
- Model training & evaluation
- Sample predictions and comparison

### **Command Line Inference**
Run the Python script to test individual sentences:

python app.py --text "This service is good. But I hate it."


**Sample Output:**
Logistic Regression: Neutral / Mixed (80.45% confidence)
LSTM Neural Network: Neutral / Mixed (3.63% confidence)


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
[Open in Streamlit](https://share.streamlit.io/<your-username>/<your-repo-name>/app.py)

---

## 🔬 Extending This Project

- Add model ensembling or BERT-based transformers for improved performance
- Use more granular sentiment categories (positive/negative/mixed/irrelevant)
- Integrate a frontend API for real-time inference via Flask or FastAPI

---

## 🧑‍💻 Author

**Sariga-20**

For questions or collaboration, contact via [your-email@example.com], or DM on GitHub!

---

## 📄 License

MIT License (or specify your own)

---

## Acknowledgements

- scikit-learn, Keras/TensorFlow
- Twitter sentiment datasets (e.g., Sentiment140)

---

> **Tip:** Update repository URL, contact email, and any dataset/source links as per your preferences.


