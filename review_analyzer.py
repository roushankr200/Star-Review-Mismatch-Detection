import streamlit as st
from transformers import pipeline

st.title("Star-Review Mismatch Detector")

@st.cache_resource
def load_model():
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

