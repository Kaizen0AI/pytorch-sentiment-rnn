import streamlit as st
import torch
import json
import os
from src.model import GRUClassifier
from src.constants import EMBED_DIM, HIDDEN_DIM
from src.evaluation import predict_sentiment, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
if not os.path.exists("artifacts/vocab.json"):
    st.error("Vocabulary file not found.")
    st.stop()

@st.cache_resource
def load_artifacts():

    if not os.path.exists("artifacts/vocab.json"):
        raise FileNotFoundError("artifact/vocab.json not found. Please run the data preparation script first.")

    if not os.path.exists("checkpoints/best_model.pt"):
        raise FileNotFoundError("checkpoints/best_model.pt not found. Please run the training script first.")

    with open("artifacts/vocab.json", "r") as f:
        vocab = json.load(f)

    model = GRUClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
    ).to(device)

    load_checkpoint(model, "checkpoints/best_model.pt", device)
    model.eval()

    return vocab, model

vocab, model = load_artifacts()

st.title("IMDB Sentiment Analyzer")

st.write("Enter a movie review below, and the model will predict whether the sentiment is positive or negative.")

text_input = st.text_area("Enter text for sentiment analysis:", height=150)

if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter a review for analysis.")
    else:
        sentiment, probability = predict_sentiment(text_input, model, vocab, device)
        st.subheader(f"Sentiment: {sentiment.capitalize()}")
        st.write(f"Predicted probability: {probability:.4f}")