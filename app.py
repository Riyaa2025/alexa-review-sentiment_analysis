import streamlit as st
import pickle
import numpy as np

# Load Models
vectorizer = pickle.load(open("Models/countVectorizer.pkl", "rb"))
model = pickle.load(open("Models/model_rf.pkl", "rb"))  
scaler = pickle.load(open("Models/scaler.pkl", "rb"))

# Title
st.set_page_config(page_title="Alexa Sentiment Analyzer", page_icon="🎙️")
st.title("🎧 Amazon Alexa Sentiment Analyzer")
st.markdown("Analyze the sentiment of Alexa user reviews — is it Positive or Negative?")

# Input Text
user_input = st.text_area("📝 Enter your review belo" \
"w:", height=150)

# Predict Button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform
        transformed = vectorizer.transform([user_input])
        scaled = scaler.transform(transformed.toarray())

        # Predict
        pred = model.predict(scaled)[0]
        sentiment = "😊 Positive" if pred == 1 else "☹️ Negative"

        # Display
        st.subheader("🧠 Model Prediction")
        st.success(f"The sentiment is **{sentiment}**")
