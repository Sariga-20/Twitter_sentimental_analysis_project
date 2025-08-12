import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="🐦 Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="centered"
)

# =========================
# Load Saved Models & Preprocessors
# =========================
@st.cache_resource
def load_models():
    """Load pre-trained models and tokenizers."""
    logreg = joblib.load("saved_models/best_logreg_model.pkl")
    vectorizer = joblib.load("saved_models/tfidf_vectorizer.pkl")
    lstm_model = load_model("saved_models/lstm_sentiment_model.keras")
    tokenizer = joblib.load("saved_models/tokenizer.pkl")
    return logreg, vectorizer, lstm_model, tokenizer

logreg, vectorizer, lstm_model, tokenizer = load_models()

# =========================
# Utility Functions
# =========================
MAX_LEN = 50

def clean_text(text):
    """Lowercase, remove URLs, mentions, hashtags, special chars."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def predict_logreg(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = logreg.predict(vec)[0]
    prob = logreg.predict_proba(vec)[0][pred]
    return pred, prob

def predict_lstm(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    prob = lstm_model.predict(pad)[0][0]
    pred = int(prob >= 0.5)
    return pred, prob

# =========================
# Header
# =========================
st.markdown(
    """
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #1DA1F2;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #444;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🐦 Twitter Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze tweets with Logistic Regression & LSTM Neural Network</div>', unsafe_allow_html=True)

# =========================
# Input Section
# =========================
st.markdown("---")
tweet = st.text_area("✏️ Enter a tweet or text to analyze sentiment:", placeholder="Type your tweet here...")

if st.button("🚀 Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        pred_lr, prob_lr = predict_logreg(tweet)
        pred_dl, prob_dl = predict_lstm(tweet)

        st.markdown("### 📊 Sentiment Results")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Logistic Regression (TF-IDF)")
            if pred_lr == 1:
                st.success(f"😀 Positive — Confidence: {prob_lr:.2%}")
            else:
                st.error(f"😠 Negative — Confidence: {prob_lr:.2%}")

        with col2:
            st.markdown("#### LSTM Neural Network")
            if pred_dl == 1:
                st.success(f"😀 Positive — Confidence: {prob_dl:.2%}")
            else:
                st.error(f"😠 Negative — Confidence: {(1-prob_dl):.2%}")

        st.markdown("---")
        st.info("✅ Works best on English tweets and casual language.\n💡 Avoid very short or neutral text for better accuracy.")

# =========================
# Creative Footer with Branding
# =========================
st.markdown(
    """
    <style>
        .footer {
            font-size: 14px;
            color: #888;
            text-align: center;
            padding-top: 15px;
        }
        .name {
            font-weight: bold;
            color: #ff4b1f;
        }
    </style>
    <hr>
    <div class="footer">
        Made with ❤️ using Streamlit, Scikit-learn & TensorFlow<br>
        Developed by <span class="name">Sariga</span> 🚀
    </div>
    """,
    unsafe_allow_html=True
)

