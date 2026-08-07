"""
train_model.py
--------------
One-time script to:
  1. Download the SST-3 dataset (Stanford Sentiment Treebank, 3-class)
  2. Train a custom LSTM model from scratch
  3. Save the model and tokenizer to ./model/

Run this ONCE before starting app.py:
    python train_model.py
"""

import os
import pickle
import numpy as np

# ── TensorFlow / Keras ─────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# ── Dataset: SST-3 via HuggingFace datasets ────────────────────────────────────
print("=" * 60)
print("  Movie Sentiment Model Trainer")
print("=" * 60)

print("\n[1/4] Loading SST dataset ...")

try:
    from datasets import load_dataset
    # sst2 has binary labels; we use 'SetFit/sst5' which has 5 labels
    # and remap to 3 classes: 0,1 -> Negative, 2 -> Neutral, 3,4 -> Positive
    raw = load_dataset("SetFit/sst5", trust_remote_code=True)

    def remap(example):
        label = example["label"]
        if label in [0, 1]:
            example["sentiment"] = 0   # Negative
        elif label == 2:
            example["sentiment"] = 1   # Neutral
        else:
            example["sentiment"] = 2   # Positive
        return example

    raw = raw.map(remap)

    train_texts  = raw["train"]["text"]
    train_labels = raw["train"]["sentiment"]
    val_texts    = raw["validation"]["text"]
    val_labels   = raw["validation"]["sentiment"]

except Exception as e:
    print(f"  HuggingFace dataset failed ({e}), falling back to CSV download ...")
    import urllib.request, csv, io

    # Fallback: download SST-3 CSV from a public mirror
    URL = "https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/train.tsv"
    try:
        with urllib.request.urlopen(URL, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        rows = list(csv.reader(io.StringIO(content), delimiter="\t"))
        all_texts  = [r[0] for r in rows if len(r) >= 2]
        all_labels = []
        for r in rows:
            if len(r) >= 2:
                score = float(r[1])
                if score <= 0.4:
                    all_labels.append(0)   # Negative
                elif score <= 0.6:
                    all_labels.append(1)   # Neutral
                else:
                    all_labels.append(2)   # Positive
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            all_texts, all_labels, test_size=0.1, random_state=42
        )
    except Exception as e2:
        print(f"  CSV download also failed ({e2}). Generating synthetic dataset ...")
        # Last resort: build a small but reasonable synthetic dataset
        # so the app is fully functional even without internet
        positives = [
            "This film is absolutely wonderful and moving",
            "A masterpiece of modern cinema, truly breathtaking",
            "Incredible performances and a gripping story",
            "One of the best movies I have ever seen",
            "Brilliant direction, stunning visuals, outstanding",
            "A heartwarming and deeply satisfying experience",
            "The acting was superb and the plot was riveting",
            "Magnificent film, loved every single moment of it",
            "An unforgettable journey, pure cinematic gold",
            "Fantastic storytelling with exceptional characters",
            "Absolutely loved this film from start to finish",
            "A triumph of storytelling and visual artistry",
            "The best movie of the year without any doubt",
            "Beautifully crafted, emotionally resonant masterpiece",
            "A wonderful film that deserves all the praise",
        ] * 40
        negatives = [
            "This movie was absolutely terrible and boring",
            "A complete waste of time, very disappointing",
            "Dull, lifeless and utterly predictable plot",
            "The worst film I have seen in many years",
            "Poor acting, bad script, awful direction",
            "A disaster from start to finish, avoid it",
            "Painfully slow and deeply unsatisfying experience",
            "Mediocre at best, dreadful and forgettable",
            "Terrible performances and a nonsensical story",
            "One of the worst movies ever made, truly bad",
            "Absolutely hated this film, total waste of money",
            "Boring from beginning to end, very disappointing",
            "A failure on every possible level imaginable",
            "Poorly written, poorly acted, poorly directed",
            "An embarrassing mess that makes no sense at all",
        ] * 40
        neutrals = [
            "It was an okay film, nothing special really",
            "Average movie with some good and bad parts",
            "Decent enough but not particularly memorable",
            "Had some interesting moments but fell flat overall",
            "Neither great nor terrible, just mediocre",
            "A passable film that does what it sets out to do",
            "Some good ideas but the execution was mixed",
            "Not bad, not great, just average all around",
            "A fair attempt with mixed results throughout",
            "Watchable but not something I would recommend",
            "It had its moments but overall just mediocre",
            "Fairly standard stuff, nothing to write home about",
            "Acceptable but unremarkable in almost every way",
            "Has a few highlights but mostly forgettable",
            "A middle of the road film that plays it safe",
        ] * 40
        all_texts  = positives + negatives + neutrals
        all_labels = [2]*len(positives) + [0]*len(negatives) + [1]*len(neutrals)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            all_texts, all_labels, test_size=0.1, random_state=42
        )

print(f"  Train samples : {len(train_texts)}")
print(f"  Val   samples : {len(val_texts)}")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print("\n[2/4] Fitting tokenizer ...")

VOCAB_SIZE   = 10_000
MAX_LEN      = 200
EMBED_DIM    = 64
LSTM_UNITS   = 64
BATCH_SIZE   = 64
EPOCHS       = 8

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

def encode(texts, labels):
    seqs    = tokenizer.texts_to_sequences(texts)
    padded  = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded, np.array(labels)

X_train, y_train = encode(train_texts, train_labels)
X_val,   y_val   = encode(val_texts,   val_labels)

print(f"  Vocab size    : {VOCAB_SIZE}")
print(f"  Max sequence  : {MAX_LEN}")

# ── Model ──────────────────────────────────────────────────────────────────────
print("\n[3/4] Building and training model ...")

model = keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    layers.SpatialDropout1D(0.3),
    layers.Bidirectional(layers.LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(3, activation="softmax"),   # 3 classes: Neg / Neu / Pos
], name="movie_sentiment")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n  [OK] Final val accuracy : {val_acc * 100:.1f}%")

# ── Save ───────────────────────────────────────────────────────────────────────
print("\n[4/4] Saving model and tokenizer ...")

os.makedirs("model", exist_ok=True)
model.save("model/sentiment_model.keras")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("  [OK] Saved -> model/sentiment_model.keras")
print("  [OK] Saved -> model/tokenizer.pkl")
print("\n  Training complete! You can now run:  python app.py\n")
