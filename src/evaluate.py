import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from src.preprocess import clean_text


def load_data(data_path: str):
    df = pd.read_csv(data_path, encoding="latin-1")

    df = df.rename(columns={
        "v1": "label",
        "v2": "text"
    })

    df = df[["label", "text"]]
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"].map({"ham": 0, "spam": 1})

    return X, y


def evaluate_model(data_path: str):
    # Load artifacts
    model = joblib.load("models/spam_classifier.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

    # Load data
    X_text, y = load_data(data_path)

    # Vectorize
    X = vectorizer.transform(X_text)

    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    DATA_PATH = "data/raw/spam.csv"
    evaluate_model(DATA_PATH)
