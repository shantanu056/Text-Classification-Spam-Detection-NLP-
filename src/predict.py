import joblib
import numpy as np
from src.preprocess import clean_text


def load_artifacts():
    model = joblib.load("models/spam_classifier.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return model, vectorizer


def predict_text(text: str):
    model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector)[0].max()

    label = "Spam" if prediction == 1 else "Ham"

    return label, confidence


if __name__ == "__main__":
    print("Enter a message to classify (or press Enter to exit):\n")

    while True:
        text = input("> ")
        if not text.strip():
            break

        label, confidence = predict_text(text)
        print(f"Prediction: {label} | Confidence: {confidence:.2f}\n")
