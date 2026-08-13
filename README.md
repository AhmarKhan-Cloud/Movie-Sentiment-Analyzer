# Movie Sentiment Analyzer – Text Feature Representation Comparison
**IMDB Movie Reviews | Bag-of-Words vs TF-IDF vs Word Embeddings**

---

## Project Overview

This project empirically compares three text feature representation methods on the IMDB sentiment classification task using the same Logistic Regression classifier for a fair comparison:

| Method | Representation | Dimensionality | Captures Semantics? |
|--------|---------------|----------------|---------------------|
| Bag-of-Words | Raw term counts | 10,000 sparse | ❌ No |
| TF-IDF | Weighted term counts | 10,000 sparse | ⚠️ Partial |
| Word Embeddings (Word2Vec) | Averaged dense vectors | 100 dense | ✅ Yes |

---

## Project Structure

```
project11_fresh/
├── app.py              ← Flask web application
├── train.py            ← Full ML training pipeline (run first)
├── run_demo.py         ← One-click Flask launcher (run for demo)
├── predict.py          ← Inference module
├── preprocessing.py    ← Text cleaning utilities
├── paths.py            ← Project root paths (Spyder-safe)
├── requirements.txt
├── environment.yml
├── models/             ← Saved models (created after train.py)
├── nltk_data/          ← Bundled NLTK stopwords (offline demo)
├── static/
│   ├── css/style.css
│   └── *.png           ← Evaluation plots (after train.py)
├── templates/
│   ├── index.html
│   ├── result.html
│   └── metrics.html
└── dataset/
    └── IMDB Dataset.csv
```

---

## Setup Instructions

### Step 1 – Get the Dataset

Download `IMDB Dataset.csv` from Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place the CSV file inside the `dataset/` folder:
```
dataset/IMDB Dataset.csv
```

### Step 2 – Create Conda Environment

Using Anaconda/Miniconda (Anaconda Prompt):
```bash
conda env create -f environment.yml
conda activate imdb_nlp
```

OR using pip only:
```bash
pip install -r requirements.txt
```

### Step 3 – Train Models (Run Once)

In **Spyder** or Anaconda Prompt:
```bash
python train.py
```

This will:
- Load and preprocess 50,000 IMDB reviews
- Extract BoW, TF-IDF, and Word2Vec features
- Train and evaluate 3 Logistic Regression classifiers
- Save all models to `models/`
- Save evaluation plots to `static/`
- Print a comparison metrics table

Expected runtime: ~5–10 minutes on a standard laptop.

### Step 4 – Launch Flask App

**Option A – One-click demo (recommended for evaluation):**
```bash
python run_demo.py
```

**Option B – Direct launch:**
```bash
python app.py
```

Open browser: **http://127.0.0.1:5000**

---

## Deploy on Streamlit Community Cloud

This project also includes a Streamlit entrypoint:

```bash
streamlit run streamlit_app.py
```

Deployment flow:

```text
Kaggle IMDb Dataset
        ↓
ML Models
        ↓
predict.py
        ↓
Streamlit
        ↓
Public URL
```

Before deploying, make sure these folders/files are committed to GitHub:

- `streamlit_app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `models/`
- `static/*.png`
- `nltk_data/`
- `predict.py`, `preprocessing.py`, and `paths.py`

On Streamlit Community Cloud:

1. Push this project to a GitHub repository.
2. Open `https://share.streamlit.io`.
3. Choose **Create app**.
4. Select your GitHub repository and branch.
5. Set the main file path to `streamlit_app.py`.
6. In **Advanced settings**, choose Python `3.10`.
7. Deploy and wait for the public `streamlit.app` URL.

The Streamlit app loads the saved models from `models/` and calls `predict.py`
for inference, so the Kaggle dataset itself is not required at runtime unless
you plan to retrain models in the cloud.

---

## Evaluation Metrics

All metrics computed on a stratified 20% held-out test set (10,000 samples).

| Metric | Description |
|--------|-------------|
| Accuracy | % of correct predictions |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | Harmonic mean of Precision & Recall |
| ROC-AUC | Area under the ROC curve |

---

## Running in Spyder

1. Open Spyder from Anaconda Navigator.
2. Set **Python interpreter** to the `imdb_nlp` conda environment  
   (Tools → Preferences → Python interpreter).
3. Open `train.py` → Run (F5) — only needed once before the demo.
4. Open `run_demo.py` → Run (F5) — starts the Flask web app.
5. Open browser at **http://127.0.0.1:5000**

> **Note:** You do not need to manually set the working directory.  
> `paths.py` resolves all file paths from the project folder automatically.  
> NLTK stopwords are bundled in `nltk_data/` so no internet is required during the demo.

---

## Key Design Decisions

- **Same classifier** (Logistic Regression) for all three methods → isolates feature quality.
- **Word2Vec trained on training data only** → no data leakage.
- **Averaged word vectors** → simple, interpretable, and fast.
- **Offline only** → no cloud calls; all data and models stored locally.
- **Flask** provides a side-by-side prediction comparison for all three methods simultaneously.
