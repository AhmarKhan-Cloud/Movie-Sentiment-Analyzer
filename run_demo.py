"""
run_demo.py
-----------
One-click launcher for the evaluation demo.

Checks that trained models exist, then starts the Flask web app.
Run this in Spyder (F5) after train.py has completed once.

Usage:
    python run_demo.py
"""

import os
import sys

from paths import ensure_project_root, project_path

REQUIRED_MODELS = [
    'bow_vectorizer.pkl',
    'bow_model.pkl',
    'tfidf_vectorizer.pkl',
    'tfidf_model.pkl',
    'word2vec.model',
    'embedding_model.pkl',
]


def main():
    ensure_project_root()

    missing = [
        fname for fname in REQUIRED_MODELS
        if not os.path.exists(project_path('models', fname))
    ]
    if missing:
        print("ERROR: Trained models not found. Run train.py first.\n")
        print("Missing files:")
        for fname in missing:
            print(f"  - models/{fname}")
        print("\nIn Spyder: open train.py and press F5. Then run this script again.")
        sys.exit(1)

    print("Models found. Starting Flask app...")
    print("Open your browser at: http://127.0.0.1:5000\n")

    from app import app
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)


if __name__ == '__main__':
    main()
