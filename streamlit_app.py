"""
streamlit_app.py
----------------
Streamlit Community Cloud entry point for the IMDB sentiment analyzer.

Deployment flow:
    Kaggle IMDb Dataset -> ML Models -> predict.py -> Streamlit -> Public URL
"""

import os

import streamlit as st

from paths import ensure_project_root, project_path
from predict import load_models, predict_all


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

SAMPLE_REVIEWS = {
    'Positive Sample': (
        'This film is an absolute masterpiece. The direction is confident, '
        'the performances are outstanding, and the story stays emotionally '
        'powerful from beginning to end.'
    ),
    'Negative Sample': (
        'This movie was a frustrating waste of time. The plot felt confused, '
        'the dialogue was awkward, and the characters never became believable.'
    ),
    'Mixed Sample': (
        'The film has a strong lead performance and a few memorable scenes, '
        'but the pacing is uneven and the second half loses focus.'
    ),
}

PLOT_FILES = [
    ('metrics_comparison.png', 'Metric Comparison'),
    ('roc_curves.png', 'ROC Curves'),
    ('confusion_matrices.png', 'Confusion Matrices'),
    ('training_time.png', 'Training Time'),
]


st.set_page_config(
    page_title='IMDB Sentiment Analyzer',
    page_icon=None,
    layout='wide',
)


def inject_css():
    st.markdown(
        """
        <style>
          :root {
            --cream: #fff7ea;
            --panel: #ffffff;
            --ink: #241b17;
            --muted: #765f54;
            --red: #b7282f;
            --red-dark: #7e1820;
            --gold: #d99a22;
            --border: #ead7ba;
            --green: #2f8f63;
            --danger: #c7453d;
          }

          .stApp {
            color: var(--ink);
            background:
              linear-gradient(90deg, rgba(183, 40, 47, 0.07) 0 1px, transparent 1px 40px),
              linear-gradient(180deg, #fffdf7 0%, var(--cream) 48%, #fff4dc 100%);
          }

          [data-testid="stHeader"] {
            background: transparent;
          }

          .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1120px;
          }

          .cinema-header {
            background: linear-gradient(135deg, var(--red-dark), var(--red) 58%, #d35535);
            border-bottom: 5px solid var(--gold);
            border-radius: 8px;
            box-shadow: 0 18px 45px rgba(82, 45, 27, 0.15);
            color: #fffaf0;
            margin-bottom: 1.5rem;
            padding: 1.5rem 1.7rem;
          }

          .kicker {
            color: #f8d98b;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
          }

          .cinema-header h1 {
            color: #fffdf7;
            font-size: clamp(1.7rem, 4vw, 2.6rem);
            line-height: 1.05;
            margin: 0;
          }

          .cinema-header p:last-child {
            color: #ffe6ba;
            margin: 0.5rem 0 0;
          }

          .hero-panel, .feature-card, .review-panel, .result-card, .plot-panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 12px 32px rgba(82, 45, 27, 0.10);
            padding: 1.2rem;
          }

          .hero-panel {
            background:
              linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,240,213,0.92)),
              repeating-linear-gradient(90deg, transparent 0 22px, rgba(183,40,47,0.07) 22px 24px);
            margin-bottom: 1.2rem;
          }

          .section-label {
            color: var(--red);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
          }

          .hero-panel h2, .review-panel h2 {
            color: var(--red-dark);
            letter-spacing: 0;
            margin-bottom: 0.5rem;
          }

          .feature-code, .method-code {
            background: #241b17;
            border-radius: 4px;
            color: #fff8e8;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin-bottom: 0.7rem;
            padding: 0.34rem 0.48rem;
          }

          .muted {
            color: var(--muted);
          }

          .result-card-positive {
            border-top: 5px solid var(--green);
          }

          .result-card-negative {
            border-top: 5px solid var(--danger);
          }

          .verdict-positive, .verdict-negative {
            border-radius: 999px;
            display: inline-block;
            font-weight: 900;
            letter-spacing: 0.05em;
            margin: 0.2rem 0 0.7rem;
            padding: 0.35rem 0.8rem;
          }

          .verdict-positive {
            background: #e8f6ee;
            border: 1px solid #9fd5b9;
            color: var(--green);
          }

          .verdict-negative {
            background: #fdecea;
            border: 1px solid #eda7a2;
            color: var(--danger);
          }

          .stButton > button {
            background: linear-gradient(135deg, var(--red), var(--red-dark));
            border: 0;
            border-radius: 999px;
            color: #fffaf0;
            font-weight: 800;
            padding: 0.55rem 1.2rem;
          }

          .stButton > button:hover {
            border: 0;
            color: #fffaf0;
            filter: brightness(0.96);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner='Loading trained ML models...')
def load_cached_models():
    ensure_project_root()
    load_models()
    return True


def render_feature_cards():
    col1, col2, col3 = st.columns(3)
    cards = [
        ('BOW', 'Bag-of-Words', 'Counts raw word occurrences. Simple, interpretable, and strong for sentiment words.'),
        ('TF', 'TF-IDF', 'Weights words by rarity so distinctive terms carry more influence.'),
        ('W2V', 'Word Embeddings', 'Averages Word2Vec vectors to represent semantic similarity in dense form.'),
    ]
    for col, (code, title, body) in zip((col1, col2, col3), cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                  <span class="feature-code">{code}</span>
                  <h3>{title}</h3>
                  <p class="muted">{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_card(result):
    status_class = 'positive' if result['label'] == 'Positive' else 'negative'
    method_code = {
        'Bag-of-Words': 'BOW',
        'TF-IDF': 'TF',
        'Word Embeddings': 'W2V',
    }.get(result['method'], 'ML')

    st.markdown(
        f"""
        <div class="result-card result-card-{status_class}">
          <span class="method-code">{method_code}</span>
          <h3>{result['method']}</h3>
          <span class="verdict-{status_class}">{result['label'].upper()}</span>
          <p><strong>Confidence:</strong> {result['confidence']}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(round(result['confidence'])))
    st.write(
        f"Positive: {result['probabilities']['Positive']}% | "
        f"Negative: {result['probabilities']['Negative']}%"
    )


def render_metrics_tab():
    st.markdown('<p class="section-label">Model Archive</p>', unsafe_allow_html=True)
    st.subheader('Evaluation Results')
    st.write(
        'Results from training on 40,000 IMDB reviews and testing on '
        '10,000 held-out samples with the same Logistic Regression classifier.'
    )

    for file_name, caption in PLOT_FILES:
        image_path = project_path('static', file_name)
        if os.path.exists(image_path):
            st.markdown('<div class="plot-panel">', unsafe_allow_html=True)
            st.image(image_path, caption=caption, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f'{caption} plot not found. Run train.py to generate it.')


def main():
    inject_css()

    st.markdown(
        """
        <div class="cinema-header">
          <div class="kicker">Cinema Review Lab</div>
          <h1>IMDB Sentiment Analyzer</h1>
          <p>Compare Bag-of-Words, TF-IDF, and Word Embeddings</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        load_cached_models()
        models_ready = True
    except FileNotFoundError as exc:
        models_ready = False
        st.error(str(exc))

    predict_tab, metrics_tab = st.tabs(['Predict Sentiment', 'Evaluation Metrics'])

    with predict_tab:
        st.markdown(
            """
            <div class="hero-panel">
              <p class="section-label">Now Showing</p>
              <h2>Select a movie, write a review, and analyze its sentiment.</h2>
              <p class="muted">
                Kaggle IMDb Dataset -> ML Models -> predict.py -> Streamlit -> Public URL
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_feature_cards()

        st.markdown('<div class="review-panel">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Review Desk</p>', unsafe_allow_html=True)
        st.subheader('Movie Sentiment Check')

        movie_name = st.selectbox('Movie name', MOVIE_OPTIONS, index=None, placeholder='Select a movie')
        sample_choice = st.radio(
            'Optional sample review',
            ['Write my own'] + list(SAMPLE_REVIEWS.keys()),
            horizontal=True,
        )
        default_review = '' if sample_choice == 'Write my own' else SAMPLE_REVIEWS[sample_choice]
        review_text = st.text_area(
            'Your review',
            value=default_review,
            height=180,
            placeholder='Write at least 10 characters about the selected movie.',
        )
        analyze_clicked = st.button('Analyze Sentiment', disabled=not models_ready)
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze_clicked:
            if not movie_name:
                st.warning('Please select a movie before entering your review.')
            elif not review_text.strip():
                st.warning('Please enter some review text.')
            elif len(review_text.strip()) < 10:
                st.warning('Review text is too short. Please enter at least 10 characters.')
            else:
                prediction_data = predict_all(review_text.strip())
                st.markdown('<p class="section-label">Screening Report</p>', unsafe_allow_html=True)
                st.subheader(f'Prediction Results for {movie_name}')
                st.write(review_text.strip())

                cols = st.columns(3)
                for col, result in zip(cols, prediction_data['results']):
                    with col:
                        render_prediction_card(result)

                pos_count = sum(r['label'] == 'Positive' for r in prediction_data['results'])
                final_label = 'Positive' if pos_count >= 2 else 'Negative'
                st.success(f'Majority vote: this review is likely {final_label}.')

                with st.expander('View preprocessed text'):
                    st.write(prediction_data['cleaned_text'])

    with metrics_tab:
        render_metrics_tab()


if __name__ == '__main__':
    main()
