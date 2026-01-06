import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text
import joblib
import os


def build_tfidf_features(
    data_path: str,
    max_features: int = 5000
):
    """
    Load dataset, clean text, and convert to TF-IDF features.
    """
    df = pd.read_csv(data_path, encoding="latin-1")

    df = df.rename(columns={
        "v1": "label",
        "v2": "text"
    })

    df = df[["label", "text"]]

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Initialize TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].map({"ham": 0, "spam": 1})

    return X, y, vectorizer


if __name__ == "__main__":
    DATA_PATH = "data/raw/spam.csv"

    X, y, vectorizer = build_tfidf_features(DATA_PATH)

    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("TF-IDF feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)
