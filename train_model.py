"""
train_model.py
----------------
Train a Sentiment Analysis model from:
Women's E-Commerce Clothing Reviews (Kaggle: nicapotato/womens-ecommerce-clothing-reviews)

Usage:
1. Download the dataset from Kaggle and extract the .csv file.
2. Put the CSV file in the same folder as this script (or pass a path via --csv).
3. Run:
      python train_model.py --csv "clean_womens_clothing_reviews.csv"
4. This produces sentiment_model.joblib and vectorizer.joblib.
   Put both files in the same folder as app.py.

Note: all print() messages here are in English on purpose, to avoid
UnicodeEncodeError crashes on Windows PowerShell consoles that default
to a non-UTF-8 codepage (cp874/cp1252). Thai text in app.py is fine
because Streamlit renders it in the browser, not the console.
"""

import argparse
import re
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def clean_text(text: str) -> str:
    """Basic text cleaning for a review string."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters a-z only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_sentiment(rating: int) -> str:
    """Map a 1-5 Rating to a 3-class sentiment label."""
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def main(csv_path: str):
    print(f"Loading data from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)

    # drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    required_cols = {"Review Text", "Rating"}
    if not required_cols.issubset(df.columns):
        print(f"ERROR: CSV must contain columns {required_cols}", flush=True)
        print(f"Found columns: {list(df.columns)}", flush=True)
        sys.exit(1)

    df = df.dropna(subset=["Review Text"]).copy()
    df["sentiment"] = df["Rating"].apply(rating_to_sentiment)
    df["clean_text"] = df["Review Text"].apply(clean_text)

    print("Class distribution:", flush=True)
    print(df["sentiment"].value_counts(), flush=True)

    X = df["clean_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text with TF-IDF...", flush=True)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...", flush=True)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n", flush=True)
    print(classification_report(y_test, y_pred), flush=True)

    joblib.dump(model, "sentiment_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("\nSaved sentiment_model.joblib and vectorizer.joblib successfully.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="Womens Clothing E-Commerce Reviews.csv",
        help="path to the dataset CSV file",
    )
    args = parser.parse_args()
    main(args.csv)
