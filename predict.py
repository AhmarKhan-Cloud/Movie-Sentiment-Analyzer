"""
predict.py
----------
Inference module. Loads trained models once at startup and provides
predict_all() for use by the Flask app.

All three methods are called simultaneously so the Flask UI can
display a side-by-side comparison on every prediction.
"""

import os
import pickle
import numpy as np
from gensim.models import Word2Vec
from paths import project_path
from preprocessing import clean_text, tokenize

MODELS_DIR    = project_path('models')
EMBEDDING_DIM = 100


# ── Lazy-load models (loaded once when Flask starts) ───────────────────────
_bow_vectorizer   = None
_bow_model        = None
_tfidf_vectorizer = None
_tfidf_model      = None
_w2v_model        = None
_emb_model        = None


def _load(filename):
    """Load a pickle file from the models directory."""
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_models():
    """
    Load all saved models from disk.
    Called once when Flask app starts.
    Raises FileNotFoundError if train.py has not been run yet.
    """
    global _bow_vectorizer, _bow_model
    global _tfidf_vectorizer, _tfidf_model
    global _w2v_model, _emb_model

    required_files = [
        'bow_vectorizer.pkl', 'bow_model.pkl',
        'tfidf_vectorizer.pkl', 'tfidf_model.pkl',
        'word2vec.model', 'embedding_model.pkl'
    ]
    for fname in required_files:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file '{path}' not found.\n"
                "Please run 'python train.py' first to train and save models."
            )

    _bow_vectorizer   = _load('bow_vectorizer.pkl')
    _bow_model        = _load('bow_model.pkl')
    _tfidf_vectorizer = _load('tfidf_vectorizer.pkl')
    _tfidf_model      = _load('tfidf_model.pkl')
    _w2v_model        = Word2Vec.load(os.path.join(MODELS_DIR, 'word2vec.model'))
    _emb_model        = _load('embedding_model.pkl')

    print("[predict.py] All models loaded successfully.")


def _average_word_vectors(tokens, model, dim):
    """Average Word2Vec embeddings for a list of tokens."""
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)


def predict_all(raw_text):
    """
    Run sentiment prediction using all three feature methods.

    Parameters
    ----------
    raw_text : str
        Raw user-entered review text.

    Returns
    -------
    dict
        {
          'cleaned_text': str,
          'results': [
            {
              'method': str,
              'label': 'Positive' | 'Negative',
              'confidence': float (0–100),
              'probabilities': {'Positive': float, 'Negative': float}
            },
            ...
          ]
        }
    """
    # Preprocess once; reuse across all three methods
    cleaned = clean_text(raw_text, use_stemming=False)

    results = []

    # ── Bag-of-Words ─────────────────────────────────────────────────────
    bow_vec = _bow_vectorizer.transform([cleaned])
    bow_prob = _bow_model.predict_proba(bow_vec)[0]
    bow_pred = int(_bow_model.predict(bow_vec)[0])
    results.append({
        'method'       : 'Bag-of-Words',
        'label'        : 'Positive' if bow_pred == 1 else 'Negative',
        'confidence'   : round(float(max(bow_prob)) * 100, 2),
        'probabilities': {
            'Positive' : round(float(bow_prob[1]) * 100, 2),
            'Negative' : round(float(bow_prob[0]) * 100, 2)
        }
    })

    # ── TF-IDF ───────────────────────────────────────────────────────────
    tfidf_vec = _tfidf_vectorizer.transform([cleaned])
    tfidf_prob = _tfidf_model.predict_proba(tfidf_vec)[0]
    tfidf_pred = int(_tfidf_model.predict(tfidf_vec)[0])
    results.append({
        'method'       : 'TF-IDF',
        'label'        : 'Positive' if tfidf_pred == 1 else 'Negative',
        'confidence'   : round(float(max(tfidf_prob)) * 100, 2),
        'probabilities': {
            'Positive' : round(float(tfidf_prob[1]) * 100, 2),
            'Negative' : round(float(tfidf_prob[0]) * 100, 2)
        }
    })

    # ── Word Embeddings (Word2Vec) ────────────────────────────────────────
    tokens  = tokenize(cleaned)
    emb_vec = _average_word_vectors(tokens, _w2v_model, EMBEDDING_DIM).reshape(1, -1)
    emb_prob = _emb_model.predict_proba(emb_vec)[0]
    emb_pred = int(_emb_model.predict(emb_vec)[0])
    results.append({
        'method'       : 'Word Embeddings',
        'label'        : 'Positive' if emb_pred == 1 else 'Negative',
        'confidence'   : round(float(max(emb_prob)) * 100, 2),
        'probabilities': {
            'Positive' : round(float(emb_prob[1]) * 100, 2),
            'Negative' : round(float(emb_prob[0]) * 100, 2)
        }
    })

    return {
        'cleaned_text' : cleaned,
        'results'      : results
    }
