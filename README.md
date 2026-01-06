# Text Classification – Spam Detection (NLP)

An end-to-end **Natural Language Processing (NLP)** project that classifies SMS messages as **Spam** or **Ham** using **TF-IDF** features and **Logistic Regression**.

This project demonstrates a clean, modular, and production-style NLP pipeline covering data preprocessing, feature extraction, model training, evaluation, and inference.

---

##  Problem Statement
Spam messages are a common issue in communication systems.  
The goal of this project is to build a machine learning model that can automatically classify incoming SMS messages as:
- **Spam**
- **Ham (Not Spam)**

---

##  Features
- Text preprocessing (cleaning, stopword removal)
- TF-IDF feature extraction (unigrams + bigrams)
- Logistic Regression classifier
- Independent evaluation & error analysis
- Interactive prediction on new messages
- Modular and reusable codebase

---

##  NLP Concepts Used
- Tokenization & normalization
- Stopword removal
- TF-IDF vectorization
- Supervised text classification
- Precision, Recall, F1-score
- Confusion matrix analysis

---

##  Project Structure
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/3b9026c1-7fb1-4333-876f-b69f1cc63f12" />

---

##  Dataset
- **SMS Spam Collection Dataset**
- Source: UCI Machine Learning Repository / Kaggle
- ~5,500 SMS messages
- Labels: `spam`, `ham`

---

##  Installation & Setup

### 1️⃣ Clone the repository
bash
git clone <your-github-repo-url>
cd Text Classification – Spam Detection (NLP)


### 2️⃣ Create a virtual environment
  python -m venv venv

Activate:

* Windows - venv\Scripts\activate

* Mac / Linux - source venv/bin/activate


### 3️⃣ Install dependencies
pip install -r requirements.txt

Usage:
* Feature Extraction - python -m src.features

* Train Model - python -m src.train

* Evaluate Model - python -m src.evaluate

* Predict on New Text - python -m src.predict


Example:
> Win a free iPhone now!!!
Prediction: Spam | Confidence: 0.97


### Model Performance (Approx.):
* Accuracy: ~97%
* Strong precision & recall for spam detection


### Evaluation Metrics:
  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix


### Future Improvements

* Replace TF-IDF with BERT embeddings
* Add Streamlit web UI
* Support multi-class classification
* Hyperparameter tuning
* Deploy as an API

Author:
  Shantanu B
