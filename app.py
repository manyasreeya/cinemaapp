"""
Flask application for Movie Review Sentiment Analysis.

The application expects:
    model/sentiment_model.keras
    model/tokenizer.pkl

It also creates:
    database/reviews.db
"""

import os
import pickle
import sqlite3

import numpy as np
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import (
    check_password_hash,
    generate_password_hash,
)

# Reduce TensorFlow log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# APP SETUP
# ============================================================

app = Flask(__name__)

app.secret_key = "cinema-secret-key-2024-xk92"


# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# FILE PATHS
# ============================================================

DB_PATH = os.path.join(
    BASE_DIR,
    "database",
    "reviews.db",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "sentiment_model.keras",
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tokenizer.pkl",
)

MAX_LEN = 200


# ============================================================
# MOVIE POOL
# ============================================================

MOVIE_POOL = [
    {
        "name": "Interstellar Drift",
        "poster": "movie1.jpg",
    },
    {
        "name": "Eclipse",
        "poster": "movie2.jpg",
    },
    {
        "name": "The Last Kingdom",
        "poster": "movie3.jpg",
    },
]


# ============================================================
# MODEL
# ============================================================

model = None
tokenizer = None


def load_model_and_tokenizer():
    global model
    global tokenizer

    print("[INFO] Loading sentiment model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    with open(
        TOKENIZER_PATH,
        "rb",
    ) as file:
        tokenizer = pickle.load(file)

    print("[OK] Model and tokenizer loaded.")


# ============================================================
# DATABASE
# ============================================================

def get_db():
    """
    Create SQLite database connection.
    """

    os.makedirs(
        os.path.dirname(DB_PATH),
        exist_ok=True,
    )

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_db():
    """
    Create database and tables.
    """

    print("[INFO] Initialising database...")

    os.makedirs(
        os.path.dirname(DB_PATH),
        exist_ok=True,
    )

    conn = get_db()

    try:

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie TEXT NOT NULL,
                review_text TEXT NOT NULL,
                sentiment TEXT NOT NULL
            );
            """
        )

        conn.commit()

        print("[OK] Database initialised.")

    finally:

        conn.close()


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def predict_sentiment(text: str) -> str:
    """
    Predict sentiment for a review.
    """

    if model is None:
        raise RuntimeError(
            "Sentiment model is not loaded."
        )

    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer is not loaded."
        )

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(
        padded,
        verbose=0,
    )[0]

    index = int(
        np.argmax(probabilities)
    )

    return LABEL_MAP.get(
        index,
        "Neutral",
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    if "user_id" in session:
        return redirect(
            url_for("review")
        )

    return redirect(
        url_for("login")
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"],
)
def register():

    if "user_id" in session:
        return redirect(
            url_for("review")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            "",
        ).strip()

        email = request.form.get(
            "email",
            "",
        ).strip().lower()

        password = request.form.get(
            "password",
            "",
        ).strip()

        confirm = request.form.get(
            "confirm",
            "",
        ).strip()

        # -----------------------------
        # Validation
        # -----------------------------

        if (
            not username
            or not email
            or not password
        ):
            flash(
                "All fields are required.",
                "error",
            )

            return render_template(
                "register.html"
            )

        if password != confirm:

            flash(
                "Passwords do not match.",
                "error",
            )

            return render_template(
                "register.html"
            )

        if len(password) < 6:

            flash(
                "Password must be at least 6 characters.",
                "error",
            )

            return render_template(
                "register.html"
            )

        hashed_password = (
            generate_password_hash(
                password
            )
        )

        conn = None

        try:

            conn = get_db()

            conn.execute(
                """
                INSERT INTO users
                (username, email, password)
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    email,
                    hashed_password,
                ),
            )

            conn.commit()

            flash(
                "Account created! Please log in.",
                "success",
            )

            return redirect(
                url_for("login")
            )

        except sqlite3.IntegrityError:

            if conn:
                conn.rollback()

            flash(
                "Username or email already exists.",
                "error",
            )

            return render_template(
                "register.html"
            )

        finally:

            if conn:
                conn.close()

    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"],
)
def login():

    if "user_id" in session:
        return redirect(
            url_for("review")
        )

    if request.method == "POST":

        identifier = request.form.get(
            "identifier",
            "",
        ).strip()

        password = request.form.get(
            "password",
            "",
        ).strip()

        conn = get_db()

        try:

            user = conn.execute(
                """
                SELECT *
                FROM users
                WHERE username = ?
                   OR email = ?
                """,
                (
                    identifier,
                    identifier,
                ),
            ).fetchone()

        finally:

            conn.close()

        if user and check_password_hash(
            user["password"],
            password,
        ):

            session["user_id"] = user["id"]

            session["username"] = (
                user["username"]
            )

            session["movies"] = MOVIE_POOL

            return redirect(
                url_for("review")
            )

        flash(
            "Invalid credentials. Please try again.",
            "error",
        )

    return render_template(
        "login.html"
    )


# ============================================================
# REVIEW PAGE
# ============================================================

@app.route("/review")
def review():

    if "user_id" not in session:
        return redirect(
            url_for("login")
        )

    movies = session.get(
        "movies",
        MOVIE_POOL,
    )

    return render_template(
        "review.html",
        movies=movies,
        username=session.get(
            "username",
            "",
        ),
    )


# ============================================================
# SUBMIT REVIEWS
# ============================================================

@app.route(
    "/submit",
    methods=["POST"],
)
def submit():

    if "user_id" not in session:
        return redirect(
            url_for("login")
        )

    movies = session.get(
        "movies",
        MOVIE_POOL,
    )

    conn = get_db()

    results = []

    try:

        # -----------------------------
        # Process each movie review
        # -----------------------------

        for i, movie in enumerate(movies):

            text = request.form.get(
                f"review_{i}",
                "",
            ).strip()

            if not text:
                text = "No review provided."

            sentiment = predict_sentiment(
                text
            )

            conn.execute(
                """
                INSERT INTO reviews
                (movie, review_text, sentiment)
                VALUES (?, ?, ?)
                """,
                (
                    movie["name"],
                    text,
                    sentiment,
                ),
            )

            results.append(
                {
                    "movie": movie["name"],
                    "poster": movie["poster"],
                    "text": text,
                    "sentiment": sentiment,
                }
            )

        conn.commit()

        # -----------------------------
        # Get sentiment totals
        # -----------------------------

        totals_rows = conn.execute(
            """
            SELECT sentiment,
                   COUNT(*) AS cnt
            FROM reviews
            GROUP BY sentiment
            """
        ).fetchall()

        totals = {
            "Positive": 0,
            "Negative": 0,
            "Neutral": 0,
        }

        for row in totals_rows:

            sentiment = row["sentiment"]

            if sentiment in totals:
                totals[sentiment] = (
                    row["cnt"]
                )

        total_all = sum(
            totals.values()
        )

    finally:

        conn.close()

    return render_template(
        "results.html",
        results=results,
        totals=totals,
        total_all=total_all,
        username=session.get(
            "username",
            "",
        ),
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success",
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# START APPLICATION
# ============================================================

# Initialise database and model when Flask starts.
init_db()
load_model_and_tokenizer()


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )