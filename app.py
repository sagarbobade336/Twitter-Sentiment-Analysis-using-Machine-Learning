import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦")

st.title("🐦 Twitter Sentiment Analysis")
st.write("Analyze tweet sentiment using Machine Learning")

tweet = st.text_area("Enter Tweet Text")

if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(tweet)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        elif prediction == 0:
            st.error("😠 Negative Sentiment")
        else:
            st.info("😐 Neutral Sentiment")
