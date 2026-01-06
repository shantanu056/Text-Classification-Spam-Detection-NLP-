import re
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean and normalize input text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation & numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

if __name__ == "__main__":
    sample = "WIN a FREE iPhone now!!!"
    print(clean_text(sample))
