"""
app.py
------
Flask application for Movie Review Sentiment Analysis.
Run AFTER train_model.py has produced model/sentiment_model.keras
"""

import os
import pickle
import random
import sqlite3

import numpy as np
from flask import (Flask, flash, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "cinema-secret-key-2024-xk92"

DB_PATH        = os.path.join("database", "reviews.db")
MODEL_PATH     = os.path.join("model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join("model", "tokenizer.pkl")
MAX_LEN        = 200

# Pool of movies (name → poster filename)
MOVIE_POOL = [
    {"name": "Interstellar Drift", "poster": "movie1.jpg"},
    {"name": "Eclipse",            "poster": "movie2.jpg"},
    {"name": "The Last Kingdom",   "poster": "movie3.jpg"},
]

# ── Load Model ─────────────────────────────────────────────────────────────────
model     = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Please run 'python train_model.py' first."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("[OK] Model and tokenizer loaded.")

# ── Database ───────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs("database", exist_ok=True)
    conn = get_db()
    cur  = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    NOT NULL UNIQUE,
            email    TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            movie       TEXT    NOT NULL,
            review_text TEXT    NOT NULL,
            sentiment   TEXT    NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    print("[OK] Database initialised.")

# ── Inference ──────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text: str) -> str:
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs  = model.predict(padded, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return LABEL_MAP[idx]

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("review"))
    return redirect(url_for("login"))


# ── Register ───────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("review"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email",    "").strip().lower()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm",  "").strip()

        # Validation
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        hashed = generate_password_hash(password)
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed),
            )
            conn.commit()
            conn.close()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
            return render_template("register.html")

    return render_template("register.html")


# ── Login ──────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("review"))

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password   = request.form.get("password",   "").strip()

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            # Pick 3 movies for this session (all 3 from pool)
            session["movies"] = MOVIE_POOL
            return redirect(url_for("review"))
        else:
            flash("Invalid credentials. Please try again.", "error")

    return render_template("login.html")


# ── Review ─────────────────────────────────────────────────────────────────────
@app.route("/review")
def review():
    if "user_id" not in session:
        return redirect(url_for("login"))
    movies = session.get("movies", MOVIE_POOL)
    return render_template("review.html", movies=movies, username=session["username"])


# ── Submit ─────────────────────────────────────────────────────────────────────
@app.route("/submit", methods=["POST"])
def submit():
    if "user_id" not in session:
        return redirect(url_for("login"))

    movies = session.get("movies", MOVIE_POOL)
    conn   = get_db()

    results = []
    for i, movie in enumerate(movies):
        text = request.form.get(f"review_{i}", "").strip()
        if not text:
            text = "No review provided."
        sentiment = predict_sentiment(text)
        conn.execute(
            "INSERT INTO reviews (movie, review_text, sentiment) VALUES (?, ?, ?)",
            (movie["name"], text, sentiment),
        )
        results.append({
            "movie":     movie["name"],
            "poster":    movie["poster"],
            "text":      text,
            "sentiment": sentiment,
        })

    conn.commit()

    # Overall totals from ALL reviews in the database
    totals_rows = conn.execute(
        "SELECT sentiment, COUNT(*) as cnt FROM reviews GROUP BY sentiment"
    ).fetchall()
    conn.close()

    totals = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for row in totals_rows:
        totals[row["sentiment"]] = row["cnt"]
    total_all = sum(totals.values())

    return render_template(
        "results.html",
        results=results,
        totals=totals,
        total_all=total_all,
        username=session["username"],
    )


# ── Logout ─────────────────────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    load_model_and_tokenizer()
    app.run(debug=True, port=5000)
