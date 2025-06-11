# 🗣️ Amazon Alexa Review – Sentiment Analysis

This is a Streamlit web app that predicts whether a customer review for an Amazon Alexa product is **positive** or **negative** using a trained machine learning model.

## 📁 Project Files
- `app.py`: 🎛️ Streamlit interface for prediction
- `Amazon_Alexa_Review_Sentiment_Analysis.ipynb`: 📊 Notebook for EDA, preprocessing, and model training
- `Models/`: 🧠 Pickled vectorizer, trained model (Random Forest), and scaler

## 🛠️ Tech Stack
- Python, pandas, numpy  
- scikit-learn
- NLTK (text preprocessing)  
- CountVectorizer  
- Streamlit (deployment)

## 💡 How It Works
- You enter a product review
- The app cleans and vectorizes the text
- The Random Forest model predicts whether the sentiment is **positive** or **negative**


