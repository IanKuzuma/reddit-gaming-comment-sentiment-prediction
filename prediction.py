# Import libraries
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel

# Load model and preprocessing tools once
@st.cache_resource
def load_assets():
    model = load_model('model_bert', custom_objects={"TFBertModel": TFBertModel})
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, tokenizer, label_encoder

# Text preprocessing
def preprocess_text(text):
    text = text.lower().strip()
    return text

# Tokenize input text
def tokenize_text(text, tokenizer, max_len=128):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )
    return {"input_ids": encoded['input_ids'], "attention_mask": encoded['attention_mask']}

# Run Streamlit App
def run():
    st.title("REDDIT r/gaming COMMENT SENTIMENT CLASSIFIER - APP")

    st.markdown("This tool analyzes Reddit gaming comments and classifies them into **Positive**, **Neutral**, or **Negative** sentiments using a BERT-enhanced RNN model.")

    # Load model and tokenizer
    model, tokenizer, label_encoder = load_assets()

    # Comment input form
    with st.form("sentiment_form"):
        st.subheader("Enter a Reddit Comment")
        user_input = st.text_area("Comment Text", height=200)
        submitted = st.form_submit_button("Classify")

    if submitted:
        if user_input.strip() == "":
            st.warning("Please enter a comment before submitting.")
            return

        # Preprocess and tokenize
        preprocessed = preprocess_text(user_input)
        tokenized = tokenize_text(preprocessed, tokenizer)

        # Predict
        prediction = model.predict(tokenized, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_sentiment = label_encoder.inverse_transform([predicted_label])[0]

        # Show prediction result
        st.markdown("### Prediction Result:")
        if predicted_sentiment == "positive":
            st.success("🟢 Sentiment: **Positive**")
            st.write("This comment shows strong approval or praise — perfect for identifying marketing-friendly or viral community trends.")
        elif predicted_sentiment == "neutral":
            st.info("🟡 Sentiment: **Neutral**")
            st.write("This comment expresses balanced or factual tone — useful for extracting objective feedback.")
        else:
            st.error("🔴 Sentiment: **Negative**")
            st.write("This comment shows criticism or dissatisfaction — useful for flagging areas needing improvement or moderation.")

if __name__ == '__main__':
    run()
