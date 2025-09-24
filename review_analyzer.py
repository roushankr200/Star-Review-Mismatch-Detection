import streamlit as st
import pandas as pd

st.title("Star-Review Mismatch Detector")

@st.cache_resource
def load_model():
    try:
        from transformers import pipeline
    except Exception as import_error:
        # Attempt a one-time inline install (useful on fresh deployments)
        try:
            import sys, subprocess
            with st.spinner('Installing missing dependencies (transformers/torch)...'):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'transformers', 'torch'])
            from transformers import pipeline  # retry after install
        except Exception as retry_error:
            st.error("Missing dependency: transformers/torch. Install with 'pip install -r requirements.txt'.\nDetails: {}".format(retry_error))
            st.stop()
    return pipeline('sentiment-analysis')

classifier = load_model()

def analyze_text_sentiment(classifier, text):
    result = classifier(text)
    label = result[0].get('label', '')
    return label.upper()

def map_rating_to_sentiment(star_rating):
    if star_rating in (4, 5):
        return 'POSITIVE'
    if star_rating in (1, 2):
        return 'NEGATIVE'
    return 'NEUTRAL'

def detect_mismatch(classifier, review_text, star_rating):
    text_sentiment = analyze_text_sentiment(classifier, review_text)
    rating_sentiment = map_rating_to_sentiment(star_rating)
    is_mismatch = (
        (text_sentiment == 'POSITIVE' and rating_sentiment == 'NEGATIVE') or
        (text_sentiment == 'NEGATIVE' and rating_sentiment == 'POSITIVE')
    )
    return is_mismatch, text_sentiment, rating_sentiment

st.subheader('Enter a review and its star rating to check for mismatches.')

review_input = st.text_area('Review Text')
star_input = st.slider('Star Rating', 1, 5, 3)

if st.button('Analyze Review'):
    if review_input.strip():
        is_mismatch, text_sentiment, rating_sentiment = detect_mismatch(classifier, review_input, star_input)
        st.write('Text Sentiment:', text_sentiment)
        st.write('Star Rating Sentiment:', rating_sentiment)
        if is_mismatch:
            st.warning('⚠️ Mismatch Detected!')
        else:
            st.success('✅ No mismatch detected!')
    else:
        st.warning('Please enter review text before analyzing.')

st.subheader('Batch Processing')

uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as read_error:
        st.error(f'Failed to read CSV: {read_error}')
        st.stop()

    required_columns = {'review_text', 'star_rating'}
    if not required_columns.issubset(df.columns):
        st.error("CSV must contain columns: 'review_text' and 'star_rating'.")
    else:
        results = []
        for _, row in df.iterrows():
            review_text = str(row['review_text'])
            try:
                star_rating = int(row['star_rating'])
            except Exception:
                # If conversion fails, mark as NaN-like and skip this row
                continue

            is_mismatch, text_sentiment, rating_sentiment = detect_mismatch(
                classifier, review_text, star_rating
            )
            results.append({
                'review_text': review_text,
                'star_rating': star_rating,
                'text_sentiment': text_sentiment,
                'rating_sentiment': rating_sentiment,
                'is_mismatch': is_mismatch,
            })

        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download Results as CSV',
                data=csv_data,
                file_name='mismatch_results.csv',
                mime='text/csv'
            )
        else:
            st.warning('No valid rows to process in the uploaded CSV.')

