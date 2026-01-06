import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from features import build_tfidf_features


def train_model(data_path: str):
    """
    Train a Logistic Regression classifier on TF-IDF features.
    """
    X, y, vectorizer = build_tfidf_features(data_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Initialize model
    model = LogisticRegression(max_iter=1000)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


if __name__ == "__main__":
    DATA_PATH = "data/raw/spam.csv"

    model, vectorizer = train_model(DATA_PATH)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/spam_classifier.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("\nModel and vectorizer saved successfully.")
