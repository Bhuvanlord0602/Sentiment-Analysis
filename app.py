import streamlit as st
import sqlite3
import bcrypt
import pickle
import re
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

# -------------- SETUP ----------------
nltk.download('stopwords')
stop_words = stopwords.words('english')

# DB setup (auto-creates if not exists)
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 username TEXT PRIMARY KEY,
                 name TEXT NOT NULL,
                 password TEXT NOT NULL,
                 role TEXT DEFAULT 'user')''')
    conn.commit()
    conn.close()

init_db()

# -------------- AUTHENTICATION ----------------
def get_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def register_user(username, name, password):
    if get_user(username):
        return False, "Username already exists"
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username, name, password) VALUES (?, ?, ?)", 
              (username, name, hashed_pw))
    conn.commit()
    conn.close()
    return True, "Registration successful"

def verify_password(stored_password, input_password):
    return bcrypt.checkpw(input_password.encode(), stored_password.encode())

# -------------- MODEL LOAD ----------------
@st.cache_resource
def load_model_vectorizer():
    with open('model.pkl', 'rb') as mfile:
        model = pickle.load(mfile)
    with open('vectorizer.pkl', 'rb') as vfile:
        vectorizer = pickle.load(vfile)
    return model, vectorizer

model, vectorizer = load_model_vectorizer()
scraper = Nitter()

# -------------- SENTIMENT FUNCTION ----------------
def predict_sentiment(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text_vector = vectorizer.transform([text])
    result = model.predict(text_vector)
    return "Positive" if result == 1 else "Negative"

# -------------- STREAMLIT UI ----------------
st.title("🔐 Twitter Sentiment Analyzer")

# Login/Register
if "auth" not in st.session_state:
    st.session_state.auth = False

auth_mode = st.sidebar.selectbox("Choose option", ["Login", "Register"])
if auth_mode == "Register":
    st.subheader("Register")
    name = st.text_input("Full Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        success, msg = register_user(username, name, password)
        st.success(msg) if success else st.error(msg)

elif auth_mode == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = get_user(username)
        if user and verify_password(user[2], password):
            st.session_state.auth = True
            st.session_state.username = user[0]
            st.session_state.name = user[1]
            st.success(f"Welcome, {user[1]}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Main app
if st.session_state.auth:
    st.sidebar.success(f"Logged in as {st.session_state['name']}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.header("🔍 Analyze Sentiment")
    choice = st.selectbox("Choose input method", ["Input Text", "Twitter Username"])
    
    if choice == "Input Text":
        input_text = st.text_area("Enter text")
        if st.button("Analyze Text"):
            sentiment = predict_sentiment(input_text)
            st.success(f"Sentiment: {sentiment}")
    
    else:
        username = st.text_input("Enter Twitter username")
        if st.button("Fetch Tweets"):
            tweets_data = scraper.get_tweets(username, mode='user', number=5)
            if 'tweets' in tweets_data:
                for tweet in tweets_data['tweets']:
                    sentiment = predict_sentiment(tweet['text'])
                    color = 'green' if sentiment == "Positive" else 'red'
                    st.markdown(f"<div style='background:{color}; padding:10px; border-radius:5px;'>"
                                f"<strong>{sentiment}:</strong> {tweet['text']}</div>", 
                                unsafe_allow_html=True)
            else:
                st.error("No tweets found or error fetching.")
