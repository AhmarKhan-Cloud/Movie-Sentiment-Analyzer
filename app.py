"""
app.py
------
Flask web interface for Project #11: Text Feature Representation Comparison.
Provides two routes:
  GET  /          -> index page with text input form
  POST /predict   -> run all 3 models and display comparison results
  GET  /results   -> metrics summary page with saved evaluation plots

Run:
    python app.py
Then open browser: http://127.0.0.1:5000
"""

import os
from flask import Flask, render_template, request, jsonify

from paths import ensure_project_root, project_path
from predict import load_models, predict_all

ensure_project_root()

app = Flask(
    __name__,
    template_folder=project_path('templates'),
    static_folder=project_path('static'),
)

MOVIE_OPTIONS = [
    'The Shawshank Redemption',
    'The Godfather',
    'The Dark Knight',
    'Forrest Gump',
    'Inception',
    'Titanic',
    'The Matrix',
    'Interstellar',
]

# Load models once at startup (not on every request)
print("Loading ML models...")
try:
    load_models()
    MODELS_READY = True
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    MODELS_READY = False


@app.route('/')
def index():
    """Home page with text input form."""
    return render_template(
        'index.html',
        models_ready=MODELS_READY,
        movie_options=MOVIE_OPTIONS
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept review text, run all 3 classifiers, return results page.
    Handles both form submission and AJAX JSON requests.
    """
    if not MODELS_READY:
        error_msg = "Models not trained yet. Please run 'python train.py' first."
        if request.is_json:
            return jsonify({'error': error_msg}), 503
        return render_template(
            'index.html',
            error=error_msg,
            models_ready=False,
            movie_options=MOVIE_OPTIONS
        )

    # Get text from form or JSON body
    if request.is_json:
        data = request.get_json()
        movie_name = data.get('movie_name', '').strip()
        review_text = data.get('text', '').strip()
    else:
        movie_name = request.form.get('movie_name', '').strip()
        review_text = request.form.get('review_text', '').strip()

    # Validate input
    if not movie_name:
        error_msg = "Please select a movie before entering your review."
        return render_template(
            'index.html',
            error=error_msg,
            models_ready=MODELS_READY,
            movie_options=MOVIE_OPTIONS,
            selected_movie=movie_name,
            review_text=review_text
        )

    if not review_text:
        error_msg = "Please enter some review text."
        return render_template(
            'index.html',
            error=error_msg,
            models_ready=MODELS_READY,
            movie_options=MOVIE_OPTIONS,
            selected_movie=movie_name,
            review_text=review_text
        )

    if len(review_text) < 10:
        error_msg = "Review text is too short. Please enter at least 10 characters."
        return render_template(
            'index.html',
            error=error_msg,
            models_ready=MODELS_READY,
            movie_options=MOVIE_OPTIONS,
            selected_movie=movie_name,
            review_text=review_text
        )

    # Run inference
    try:
        prediction_data = predict_all(review_text)
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        return render_template(
            'index.html',
            error=error_msg,
            models_ready=MODELS_READY,
            movie_options=MOVIE_OPTIONS,
            selected_movie=movie_name,
            review_text=review_text
        )

    if request.is_json:
        prediction_data['movie_name'] = movie_name
        return jsonify(prediction_data)

    return render_template(
        'result.html',
        movie_name     = movie_name,
        original_text  = review_text,
        cleaned_text   = prediction_data['cleaned_text'],
        results        = prediction_data['results']
    )


@app.route('/metrics')
def metrics():
    """Show evaluation plots and metrics from training."""
    # Check which plot images exist
    plot_files = ['metrics_comparison.png', 'confusion_matrices.png',
                  'roc_curves.png', 'training_time.png']
    plots_exist = {f: os.path.exists(project_path('static', f)) for f in plot_files}
    return render_template('metrics.html', plots=plots_exist)


@app.route('/health')
def health():
    """Simple health check endpoint."""
    return jsonify({
        'status'       : 'ok',
        'models_ready' : MODELS_READY
    })


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  IMDB Sentiment – Text Feature Comparison App")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    # debug=False + use_reloader=False avoids double-start issues in Spyder
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
