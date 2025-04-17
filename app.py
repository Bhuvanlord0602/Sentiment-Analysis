import streamlit as st
import sqlite3
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Optional Twitter scraping
try:
    from ntscraper import Nitter
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

nltk.download('stopwords')
stop_words = stopwords.words('english')

# ---------------------- DATABASE SETUP ----------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

def register_user(username, name, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, name, password) VALUES (?, ?, ?)", (username, name, password))
        conn.commit()
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# ---------------------- ML MODELS LOADER ----------------------
def load_model(model_name):
    with open(f"{model_name}_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_name}_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# ---------------------- TEXT PREPROCESSING ----------------------
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# ---------------------- SENTIMENT PREDICTION ----------------------
def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"

# ---------------------- TWITTER SCRAPER ----------------------
def get_tweets(username):
    if not SCRAPER_AVAILABLE:
        return []
    scraper = Nitter()
    tweets_data = scraper.get_tweets(username, mode='user', number=5)
    return [tweet['text'] for tweet in tweets_data.get('tweets', [])]

# ---------------------- STREAMLIT APP ----------------------
def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="centered")
    st.title("🔐 Sentiment Analyzer with User Auth")

    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    init_db()

    if choice == "Register":
        st.subheader("Create New Account")
        username = st.text_input("Username")
        name = st.text_input("Name")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            success, msg = register_user(username, name, password)
            if success:
                st.success(str(msg))
            else:
                st.error(str(msg))

    elif choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.success(f"Welcome {user[1]}!")
                run_sentiment_analysis()
            else:
                st.error("Invalid credentials")

# ---------------------- MAIN LOGIC AFTER LOGIN ----------------------
def run_sentiment_analysis():
    st.header("🔍 Sentiment Analysis")
    
    model_choice = st.selectbox("Choose a model", ["logistic", "svm", "naivebayes"])
    model, vectorizer = load_model(model_choice)

    option = st.radio("Select input method", ["Input text", "Analyze tweets from username"])

    if option == "Input text":
        user_text = st.text_area("Enter text to analyze")
        if st.button("Analyze"):
            sentiment = predict_sentiment(user_text, model, vectorizer)
            st.success(f"Sentiment: {sentiment}")

    elif option == "Analyze tweets from username":
        if not SCRAPER_AVAILABLE:
            st.warning("Nitter module not installed. Twitter scraping unavailable.")
            return
        twitter_user = st.text_input("Enter Twitter username")
        if st.button("Fetch and Analyze"):
            tweets = get_tweets(twitter_user)
            if tweets:
                for tweet in tweets:
                    sentiment = predict_sentiment(tweet, model, vectorizer)
                    color = "lightgreen" if sentiment == "Positive" else "tomato"
                    st.markdown(f"<div style='background-color:{color}; padding:10px; border-radius:5px;'>{tweet}<br><b>{sentiment}</b></div>", unsafe_allow_html=True)
            else:
                st.info("No tweets found or failed to fetch.")

if __name__ == '__main__':
    main()
