"""
train.py
--------
Full ML training pipeline for Project #11: Text Feature Representation Comparison.
Trains three feature representations on IMDB dataset and compares them:
  1. Bag-of-Words  (BoW)  -> Logistic Regression
  2. TF-IDF              -> Logistic Regression
  3. Word2Vec Embeddings -> Logistic Regression (averaged embeddings)

Run this ONCE before launching the Flask app.
Output: saved models in /models/ folder + evaluation plots + metrics summary.

Usage (in Spyder or terminal):
    python train.py
"""

# ─────────────────────────── Imports ────────────────────────────────────────
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend for Spyder/headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec

from paths import ensure_project_root, project_path
from preprocessing import preprocess_series, tokenize

warnings.filterwarnings('ignore')

# ─────────────────────────── Config ─────────────────────────────────────────
DATASET_PATH  = project_path('dataset', 'IMDB Dataset.csv')
MODELS_DIR    = project_path('models')
PLOTS_DIR     = project_path('static')
RANDOM_STATE  = 42
TEST_SIZE     = 0.20          # 80/20 split
MAX_FEATURES  = 10000         # BoW and TF-IDF vocabulary cap
MAX_ITER      = 500           # Logistic Regression max iterations
EMBEDDING_DIM = 100           # Word2Vec vector size
W2V_WINDOW    = 5             # Word2Vec context window
W2V_MIN_COUNT = 2             # Ignore tokens with fewer occurrences
W2V_EPOCHS    = 10            # Training epochs for Word2Vec


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ════════════════════════════════════════════════════════════════════════════
def load_dataset():
    """Load IMDB CSV from local dataset/ folder."""
    print("\n[1/6] Loading IMDB dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'.\n"
            "Download 'IMDB Dataset.csv' from Kaggle and place it in the dataset/ folder."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f"       Loaded {len(df)} reviews.")
    print(f"       Columns: {list(df.columns)}")
    print(f"       Sentiment distribution:\n{df['sentiment'].value_counts()}")

    # Encode labels: positive=1, negative=0
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['sentiment'])   # positive->1, negative->0
    print(f"       Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESS
# ════════════════════════════════════════════════════════════════════════════
def preprocess_data(df):
    """Clean all reviews using preprocessing.py pipeline."""
    print("\n[2/6] Preprocessing reviews (this may take 2–4 minutes)...")
    t0 = time.time()
    df['clean_text'] = preprocess_series(df['review'], use_stemming=False, verbose=True)
    print(f"       Done in {time.time() - t0:.1f}s")
    print(f"       Sample cleaned review: {df['clean_text'].iloc[0][:120]}...")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 3. TRAIN/TEST SPLIT
# ════════════════════════════════════════════════════════════════════════════
def split_data(df):
    """Stratified 80/20 train-test split."""
    print(f"\n[3/6] Splitting dataset ({int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test)...")
    X = df['clean_text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"       Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING (3 approaches)
# ════════════════════════════════════════════════════════════════════════════

# ── 4a. Bag-of-Words ────────────────────────────────────────────────────────
def build_bow_features(X_train, X_test):
    """
    Bag-of-Words (CountVectorizer).
    Each document = vector of raw token counts (unordered).
    Vocabulary capped at MAX_FEATURES most frequent terms.
    """
    print("\n  → Building Bag-of-Words features...")
    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),    # unigrams + bigrams
        min_df=2               # ignore very rare terms
    )
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow  = vectorizer.transform(X_test)
    print(f"     BoW matrix: train {X_train_bow.shape}, test {X_test_bow.shape}")
    return vectorizer, X_train_bow, X_test_bow


# ── 4b. TF-IDF ──────────────────────────────────────────────────────────────
def build_tfidf_features(X_train, X_test):
    """
    TF-IDF (Term Frequency – Inverse Document Frequency).
    Weights term counts by how rare the term is across all documents.
    Reduces weight of frequent-but-uninformative words.
    """
    print("\n  → Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,     # apply log(TF) to dampen high counts
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"     TF-IDF matrix: train {X_train_tfidf.shape}, test {X_test_tfidf.shape}")
    return vectorizer, X_train_tfidf, X_test_tfidf


# ── 4c. Word Embeddings (Word2Vec) ──────────────────────────────────────────
def build_embedding_features(X_train, X_test):
    """
    Word2Vec averaged embeddings.
    Trains a Skip-gram Word2Vec model on the training corpus.
    Each document is represented as the MEAN of its token vectors.
    This captures semantic similarity unlike BoW/TF-IDF.
    """
    print("\n  → Training Word2Vec model (this may take 2–3 minutes)...")

    # Tokenize
    train_tokens = [tokenize(doc) for doc in X_train]
    test_tokens  = [tokenize(doc) for doc in X_test]

    # Train Word2Vec on TRAINING data only (prevent data leakage)
    t0 = time.time()
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=EMBEDDING_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=1,            # Skip-gram (better for small datasets vs CBOW)
        epochs=W2V_EPOCHS,
        workers=4,
        seed=RANDOM_STATE
    )
    print(f"     Word2Vec trained in {time.time() - t0:.1f}s")
    print(f"     Vocabulary size: {len(w2v_model.wv):,} words")

    def average_word_vectors(token_list, model, dim):
        """Average all in-vocabulary word vectors for a document."""
        vectors = [
            model.wv[word]
            for word in token_list
            if word in model.wv
        ]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(dim)   # OOV document -> zero vector

    print("     Generating document embedding vectors...")
    X_train_emb = np.array([average_word_vectors(doc, w2v_model, EMBEDDING_DIM) for doc in train_tokens])
    X_test_emb  = np.array([average_word_vectors(doc, w2v_model, EMBEDDING_DIM) for doc in test_tokens])
    print(f"     Embedding matrix: train {X_train_emb.shape}, test {X_test_emb.shape}")

    return w2v_model, X_train_emb, X_test_emb


# ════════════════════════════════════════════════════════════════════════════
# 5. TRAIN & EVALUATE MODELS
# ════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(name, X_train_feat, X_test_feat, y_train, y_test):
    """
    Train Logistic Regression on given features and return full evaluation metrics.
    Same classifier for ALL three representations (fair comparison).
    """
    print(f"\n  → Training Logistic Regression [{name}]...")
    t0 = time.time()
    clf = LogisticRegression(
        max_iter=MAX_ITER,
        C=1.0,                 # regularization strength
        solver='lbfgs',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train_feat, y_train)
    train_time = time.time() - t0
    print(f"     Trained in {train_time:.1f}s")

    # Predictions
    y_pred      = clf.predict(X_test_feat)
    y_prob      = clf.predict_proba(X_test_feat)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print(f"\n     ── {name} Results ──")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}")
    print(f"     Recall    : {rec:.4f}")
    print(f"     F1 Score  : {f1:.4f}")
    print(f"     ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])}")

    return {
        'name'       : name,
        'model'      : clf,
        'accuracy'   : acc,
        'precision'  : prec,
        'recall'     : rec,
        'f1'         : f1,
        'auc'        : auc,
        'cm'         : cm,
        'fpr'        : fpr,
        'tpr'        : tpr,
        'train_time' : train_time,
        'y_pred'     : y_pred,
        'y_prob'     : y_prob
    }


# ════════════════════════════════════════════════════════════════════════════
# 6. SAVE MODELS
# ════════════════════════════════════════════════════════════════════════════
def save_models(bow_vec, bow_res, tfidf_vec, tfidf_res, w2v_model, emb_res):
    """Pickle all vectorizers and classifiers for Flask inference."""
    print("\n[5/6] Saving models to disk...")

    def save(obj, filename):
        path = os.path.join(MODELS_DIR, filename)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"       Saved: {path}")

    save(bow_vec,          'bow_vectorizer.pkl')
    save(bow_res['model'], 'bow_model.pkl')

    save(tfidf_vec,          'tfidf_vectorizer.pkl')
    save(tfidf_res['model'], 'tfidf_model.pkl')

    # Word2Vec saved via gensim native format + classifier via pickle
    w2v_path = os.path.join(MODELS_DIR, 'word2vec.model')
    w2v_model.save(w2v_path)
    print(f"       Saved: {w2v_path}")
    save(emb_res['model'], 'embedding_model.pkl')

    print("       All models saved successfully.")


# ════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════
def generate_plots(results_list):
    """
    Generate and save 4 comparison plots:
      1. Bar chart: Accuracy / Precision / Recall / F1 / AUC
      2. Confusion matrices (3 side-by-side)
      3. ROC curves (3 overlaid)
      4. Training time comparison
    """
    print("\n[6/6] Generating comparison plots...")

    names   = [r['name'] for r in results_list]
    colors  = ['#4C72B0', '#DD8452', '#55A868']   # blue, orange, green
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

    # ── Plot 1: Metric Comparison Bar Chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    for i, (res, col) in enumerate(zip(results_list, colors)):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=res['name'],
                      color=col, edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Feature Representation Comparison – Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.75, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("       Saved: static/metrics_comparison.png")

    # ── Plot 2: Confusion Matrices ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Confusion Matrices – All Feature Methods', fontsize=14, fontweight='bold')
    class_names = ['Negative', 'Positive']
    for ax, res, col in zip(axes, results_list, colors):
        cm_norm = res['cm'].astype('float') / res['cm'].sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap='Blues', linewidths=0.5,
                    annot_kws={'size': 12})
        ax.set_title(f"{res['name']}\nAcc: {res['accuracy']:.4f}", fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("       Saved: static/confusion_matrices.png")

    # ── Plot 3: ROC Curves ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for res, col in zip(results_list, colors):
        ax.plot(res['fpr'], res['tpr'], lw=2, color=col,
                label=f"{res['name']} (AUC = {res['auc']:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves – Feature Representation Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("       Saved: static/roc_curves.png")

    # ── Plot 4: Training Time ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    times = [r['train_time'] for r in results_list]
    bars = ax.bar(names, times, color=colors, edgecolor='white', linewidth=0.8, width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature Method', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Classifier Training Time Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'training_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("       Saved: static/training_time.png")

    print("       All plots saved.")


# ════════════════════════════════════════════════════════════════════════════
# 8. PRINT SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════
def print_summary(results_list):
    """Print a formatted comparison table to console."""
    print("\n" + "═" * 72)
    print("  FINAL COMPARISON SUMMARY")
    print("═" * 72)
    header = f"{'Method':<18} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}"
    print(header)
    print("─" * 72)
    for r in results_list:
        row = (f"{r['name']:<18} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
               f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f}")
        print(row)
    print("═" * 72)

    # Best model
    best = max(results_list, key=lambda x: x['f1'])
    print(f"\n  Best model by F1: {best['name']} (F1 = {best['f1']:.4f})\n")


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ensure_project_root()

    print("=" * 72)
    print("  PROJECT #11: Text Feature Representation Comparison")
    print("  Dataset: IMDB Movie Reviews (50,000 samples)")
    print("=" * 72)

    total_start = time.time()

    # ── Step 1–3: Load, Preprocess, Split ───────────────────────────────────
    df                                   = load_dataset()
    df                                   = preprocess_data(df)
    X_train, X_test, y_train, y_test     = split_data(df)

    # ── Step 4: Build Features ───────────────────────────────────────────────
    print("\n[4/6] Extracting features (3 methods)...")

    bow_vec,   X_train_bow,   X_test_bow   = build_bow_features(X_train, X_test)
    tfidf_vec, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train, X_test)
    w2v_model, X_train_emb,  X_test_emb   = build_embedding_features(X_train, X_test)

    # ── Step 5: Train & Evaluate ─────────────────────────────────────────────
    print("\n[5/6] Training & evaluating models...")

    bow_res  = train_and_evaluate('Bag-of-Words', X_train_bow,   X_test_bow,   y_train, y_test)
    tfidf_res = train_and_evaluate('TF-IDF',      X_train_tfidf, X_test_tfidf, y_train, y_test)
    emb_res  = train_and_evaluate('Word2Vec',     X_train_emb,   X_test_emb,   y_train, y_test)

    results_list = [bow_res, tfidf_res, emb_res]

    # ── Step 6: Save & Plot ───────────────────────────────────────────────────
    save_models(bow_vec, bow_res, tfidf_vec, tfidf_res, w2v_model, emb_res)
    generate_plots(results_list)
    print_summary(results_list)

    print(f"\n  Total pipeline time: {time.time() - total_start:.1f}s")
    print("  Training complete. You can now run: python app.py\n")
