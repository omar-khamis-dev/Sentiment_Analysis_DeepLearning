import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    """
    Clean raw text by:
      - Lowercasing
      - Removing URLs, mentions, hashtags, numbers, and special characters
      - Removing stopwords

    Args:
        text (str): Input raw text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove numbers and special characters (keep words only)
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]

    return " ".join(tokens)