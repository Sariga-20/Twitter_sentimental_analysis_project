import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# =========================
# Load Saved Models & Preprocessors
# =========================
@st.cache_resource
def load_models():
    logreg = joblib.load("saved_models/best_logreg_model.pkl")
    vectorizer = joblib.load("saved_models/vectorizer.pkl")
    lstm_model = load_model("saved_models/lstm_sentiment_model.keras")  
    tokenizer = joblib.load("saved_models/tokenizer.pkl")

    return logreg, vectorizer, lstm_model, tokenizer

logreg, vectorizer, lstm_model, tokenizer = load_models()

# =========================
# Utility Functions
# =========================
MAX_LEN = 50

def clean_text(text):
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
# Streamlit App UI
# =========================
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")

st.title("🐦 Twitter Sentiment Analysis")
st.write("This app analyzes the **sentiment** of tweets using:")
st.markdown("- 📊 **Logistic Regression + TF-IDF** (Classic ML)")
st.markdown("- 🤖 **LSTM Neural Network** (Deep Learning)")

# Input text area
tweet = st.text_area("✏️ Enter a tweet or text:", placeholder="Type your tweet here...")

if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Predictions
        pred_lr, prob_lr = predict_logreg(tweet)
        pred_dl, prob_dl = predict_lstm(tweet)

        # Display results
        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Logistic Regression")
            if pred_lr == 1:
                st.success(f"Positive 😀 ({prob_lr:.2%} confidence)")
            else:
                st.error(f"Negative 😠 ({prob_lr:.2%} confidence)")

        with col2:
            st.markdown("### 🤖 LSTM Neural Network")
            if pred_dl == 1:
                st.success(f"Positive 😀 ({prob_dl:.2%} confidence)")
            else:
                st.error(f"Negative 😠 ({(1-prob_dl):.2%} confidence)")

        st.markdown("---")
        st.info("✅ This model works best on English tweets and informal text. "
                "For best accuracy, avoid very short or ambiguous text.")

# Footer
st.markdown(
    "<hr/><center>Made with ❤️ using Streamlit, Scikit-learn, and TensorFlow</center>",
    unsafe_allow_html=True
)
