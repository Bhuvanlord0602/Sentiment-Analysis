import pickle
import re
import sqlite3
import os
import bcrypt
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords')

# ---------------- Model & Vectorizer Test ----------------
"hi how are you"
def load_model_vectorizer():
    with open("model.pkl", "rb") as mf:
        model = pickle.load(mf)
    with open("vectorizer.pkl", "rb") as vf:
        vectorizer = pickle.load(vf)
    return model, vectorizer

def preprocess_text(text):
    stop_words = stopwords.words('english')
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

def test_model_prediction():
    model, vectorizer = load_model_vectorizer()
    test_text = "I hate bugs, but I love fixing them!"
    clean_text = preprocess_text(test_text)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"✅ Prediction works. Output: {sentiment}")

# ---------------- Database & Auth Test ----------------

DB_NAME = "users_test.db"

def init_test_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS users")
    c.execute('''CREATE TABLE users (
                 username TEXT PRIMARY KEY,
                 name TEXT NOT NULL,
                 password TEXT NOT NULL,
                 role TEXT DEFAULT 'user')''')
    conn.commit()
    conn.close()

def register_user(username, name, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        c.execute("INSERT INTO users (username, name, password) VALUES (?, ?, ?)", 
                  (username, name, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def verify_password(stored_password, input_password):
    return bcrypt.checkpw(input_password.encode(), stored_password.encode())

def test_auth_flow():

    init_test_db()
    registered = register_user(test_username, test_name, test_password)
    assert registered, "❌ Registration failed"
    print("✅ Registration successful")

    user = get_user(test_username)
    assert user is not None, "❌ User not found"
    print("✅ User found")

    assert verify_password(user[2], test_password), "❌ Password verification failed"
    print("✅ Password verified")

# ---------------- Run All Tests ----------------

if __name__ == "__main__":
    try:
        test_model_prediction()
        test_auth_flow()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print("\n❌ Test failed:", str(e))
        exit(1)
