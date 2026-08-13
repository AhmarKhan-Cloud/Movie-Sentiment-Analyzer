"""
preprocessing.py
----------------
Text preprocessing module for IMDB sentiment classification.
Handles lowercasing, HTML tag removal, punctuation, stopwords, and tokenization.
Used by all three feature pipelines (BoW, TF-IDF, Word Embeddings).
"""

import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from paths import project_path

# Use bundled NLTK data in nltk_data/ for offline demo (no internet required)
NLTK_DATA_DIR = project_path('nltk_data')
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)


def _ensure_stopwords():
    """Load stopwords from local nltk_data/; download once if missing."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True, download_dir=NLTK_DATA_DIR)


_ensure_stopwords()

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def remove_html_tags(text):
    """Strip HTML tags from raw review text (e.g., <br />, <b>)."""
    return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text):
    """Remove any URLs present in the text."""
    return re.sub(r'http\S+|www\S+', ' ', text)


def remove_punctuation(text):
    """Remove all punctuation characters."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text):
    """Remove standalone digits (not part of words)."""
    return re.sub(r'\b\d+\b', ' ', text)


def normalize_whitespace(text):
    """Collapse multiple spaces into one and strip edges."""
    return re.sub(r'\s+', ' ', text).strip()


def clean_text(text, use_stemming=False):
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove HTML tags
      3. Remove URLs
      4. Remove punctuation
      5. Remove digits
      6. Remove stopwords
      7. Optional stemming
      8. Normalize whitespace

    Parameters
    ----------
    text : str
        Raw review text.
    use_stemming : bool
        Apply Porter stemming if True. Default False (preserves word shape
        for embeddings; stemming helps BoW/TF-IDF slightly).

    Returns
    -------
    str
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)

    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    if use_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]

    return normalize_whitespace(' '.join(tokens))


def tokenize(text):
    """
    Return a list of tokens from already-cleaned text.
    Used by the Word2Vec embedding pipeline.
    """
    return text.split()


def preprocess_series(series, use_stemming=False, verbose=True):
    """
    Apply clean_text to an entire pandas Series.

    Parameters
    ----------
    series : pd.Series
        Column of raw review strings.
    use_stemming : bool
    verbose : bool
        Print progress every 5000 rows.

    Returns
    -------
    pd.Series
        Cleaned text series.
    """
    cleaned = []
    total = len(series)
    for i, text in enumerate(series):
        cleaned.append(clean_text(text, use_stemming=use_stemming))
        if verbose and (i + 1) % 5000 == 0:
            print(f"  Preprocessed {i + 1}/{total} reviews...")
    return cleaned
