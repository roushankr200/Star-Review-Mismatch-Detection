import streamlit as st
import streamlit.components.v1 as components
import numpy as np
try:
    import plotly.express as px
    from streamlit_plotly_events import plotly_events
except Exception:
    try:
        import sys, subprocess
        with st.spinner('Installing plotting dependencies (plotly, streamlit-plotly-events)...'):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'plotly', 'streamlit-plotly-events'])
        import plotly.express as px
        from streamlit_plotly_events import plotly_events
    except Exception:
        px = None
        plotly_events = None
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

@st.cache_resource
def load_zero_shot_model():
    try:
        from transformers import pipeline
    except Exception as import_error:
        try:
            import sys, subprocess
            with st.spinner('Installing missing dependencies (transformers/torch)...'):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'transformers', 'torch'])
            from transformers import pipeline
        except Exception as retry_error:
            st.error("Missing dependency: transformers/torch. Install with 'pip install -r requirements.txt'.\nDetails: {}".format(retry_error))
            st.stop()
    return pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

zero_shot_classifier = load_zero_shot_model()

candidate_labels = ['Customer Service', 'Shipping and Delivery', 'Product Quality', 'Pricing and Value', 'Website Usability']

@st.cache_resource
def load_sarcasm_model():
    try:
        from transformers import pipeline
    except Exception:
        try:
            import sys, subprocess
            with st.spinner('Installing missing dependencies (transformers/torch)...'):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'transformers', 'torch'])
            from transformers import pipeline
        except Exception as e:
            st.error("Missing dependency: transformers/torch. Install with 'pip install -r requirements.txt'.\nDetails: {}".format(e))
            st.stop()
    # English sarcasm detector
    return pipeline('text-classification', model='helinivan/english-sarcasm-detector')

sarcasm_classifier = load_sarcasm_model()

def analyze_text_sentiment(classifier, text):
    result = classifier(text)
    label = result[0].get('label', '')
    return label.upper()

def analyze_text_sentiment_with_score(classifier, text):
    result = classifier(text)
    label = result[0].get('label', '')
    score = result[0].get('score', 0.0)
    return label.upper(), float(score)

def explain_sentiment(text):
    try:
        import shap
    except Exception:
        try:
            import sys, subprocess
            with st.spinner('Installing missing dependency: shap...'):
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'shap'])
            import shap  # retry
        except Exception as e:
            return f"<div>Could not load SHAP for explanations: {e}</div>"

    try:
        masker = shap.maskers.Text()

        def predict_proba(texts):
            results = classifier(list(texts))
            proba = []
            for r in results:
                label = str(r.get('label', '')).upper()
                score = float(r.get('score', 0.0))
                prob_pos = score if label == 'POSITIVE' else 1.0 - score
                proba.append([1.0 - prob_pos, prob_pos])
            return np.array(proba)

        explainer = shap.Explainer(predict_proba, masker)
        explanation = explainer([text])

        # Use positive class (index 1) for visualization
        try:
            pos_explanation = explanation[:, :, 1]
        except Exception:
            pos_explanation = explanation

        shap_fig = shap.force_plot(
            getattr(explainer, 'expected_value', [0.0, 0.0])[1] if isinstance(getattr(explainer, 'expected_value', 0.0), (list, tuple, np.ndarray)) else getattr(explainer, 'expected_value', 0.0),
            pos_explanation.values[0] if hasattr(pos_explanation, 'values') else np.array([]),
            pos_explanation.data[0] if hasattr(pos_explanation, 'data') else text,
            matplotlib=False
        )
        shap_html = f"<head>{shap.getjs()}</head><body>{shap_fig.html()}</body>"
        return shap_html
    except Exception as e:
        return f"<div>SHAP explanation unavailable: {e}</div>"

def detect_sarcasm(text, threshold: float = 0.7) -> bool:
    try:
        result = sarcasm_classifier(text)
        if isinstance(result, list) and len(result) > 0:
            label = str(result[0].get('label', '')).upper()
            score = float(result[0].get('score', 0.0))
            is_sarcastic_label = ('SARCASM' in label) or ('SARCASTIC' in label)
            return bool(is_sarcastic_label and score >= threshold)
    except Exception:
        pass
    return False

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
confidence_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.75)

if st.button('Analyze Review'):
    if review_input.strip():
        text_sentiment, confidence_score = analyze_text_sentiment_with_score(classifier, review_input)
        rating_sentiment = map_rating_to_sentiment(star_input)
        base_mismatch = ((text_sentiment == 'POSITIVE' and rating_sentiment == 'NEGATIVE') or
                         (text_sentiment == 'NEGATIVE' and rating_sentiment == 'POSITIVE'))

        st.write('Text Sentiment:', text_sentiment)
        st.write('Star Rating Sentiment:', rating_sentiment)

        if base_mismatch and confidence_score > confidence_threshold:
            st.warning(f'⚠️ Mismatch Detected with {confidence_score:.2%} confidence!')
            shap_html = explain_sentiment(review_input)
            components.html(shap_html, height=200)
            if detect_sarcasm(review_input):
                st.info('ℹ️ Mismatch detected, but sarcasm may be a factor. Manual review recommended.')
        elif base_mismatch and confidence_score <= confidence_threshold:
            st.info('A potential mismatch was detected, but it is below your confidence threshold.')
            shap_html = explain_sentiment(review_input)
            components.html(shap_html, height=200)
            if detect_sarcasm(review_input):
                st.info('ℹ️ Mismatch detected, but sarcasm may be a factor. Manual review recommended.')
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

    required_columns = {'review_text', 'star_rating', 'date', 'author_id'}
    if not required_columns.issubset(df.columns):
        st.error("CSV must contain columns: 'review_text', 'star_rating', 'date', and 'author_id'.")
    else:
        # Normalize date column early
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception:
            st.error("Failed to parse 'date' column. Ensure it is a valid date format.")
            st.stop()

        results = []
        for _, row in df.iterrows():
            review_text = str(row['review_text'])
            try:
                star_rating = int(row['star_rating'])
            except Exception:
                continue
            date_value = row.get('date')
            if pd.isna(date_value):
                continue
            author_id = row.get('author_id')
            if pd.isna(author_id):
                continue

            text_sentiment, confidence_score = analyze_text_sentiment_with_score(classifier, review_text)
            rating_sentiment = map_rating_to_sentiment(star_rating)
            base_mismatch = (
                (text_sentiment == 'POSITIVE' and rating_sentiment == 'NEGATIVE') or
                (text_sentiment == 'NEGATIVE' and rating_sentiment == 'POSITIVE')
            )

            results.append({
                'review_text': review_text,
                'star_rating': star_rating,
                'date': date_value,
                'author_id': author_id,
                'text_sentiment': text_sentiment,
                'rating_sentiment': rating_sentiment,
                'confidence_score': float(confidence_score),
                'is_mismatch': bool(base_mismatch),
            })

        if results:
            results_df = pd.DataFrame(results)
            st.session_state['results_df'] = results_df.copy()
            st.dataframe(results_df)

            mismatches_df = results_df[results_df['is_mismatch'] == True]
            if not mismatches_df.empty:
                st.subheader('Top 5 Most Confident Mismatches')
                top5 = mismatches_df.sort_values('confidence_score', ascending=False).head(5)
                st.dataframe(top5)

                topics = []
                for text in mismatches_df['review_text'].tolist():
                    try:
                        z = zero_shot_classifier(text, candidate_labels)
                        topics.append(z.get('labels', ['Unknown'])[0] if isinstance(z, dict) else 'Unknown')
                    except Exception:
                        topics.append('Unknown')
                mismatches_df = mismatches_df.copy()
                mismatches_df['topic'] = topics

                st.subheader('Mismatched Review Topics')
                topic_counts = mismatches_df['topic'].value_counts()
                if px is not None and plotly_events is not None and not topic_counts.empty:
                    topic_df = topic_counts.reset_index()
                    topic_df.columns = ['topic', 'count']
                    fig_topics = px.bar(topic_df, x='topic', y='count', title='Topics among mismatches')
                    selected_points = plotly_events(fig_topics, click_event=True, hover_event=False, select_event=False, override_height=400, override_width='100%')
                    if selected_points:
                        sel_topic = selected_points[0].get('x')
                        st.write(f"Filtering mismatched reviews for topic: {sel_topic}")
                        drill_df = mismatches_df[mismatches_df['topic'] == sel_topic]
                        st.dataframe(drill_df)
                else:
                    st.bar_chart(topic_counts)

            # Historical Trend Analysis
            st.subheader('Mismatch Rate Over Time')
            freq = st.selectbox('Group by time interval', ['D', 'W', 'M'], index=1, help='D = day, W = week, M = month')
            try:
                results_df = results_df.copy()
                results_df['period_start'] = results_df['date'].dt.to_period(freq).dt.start_time
                by_period = (results_df
                             .groupby('period_start')
                             .agg(total_reviews=('review_text', 'count'),
                                  mismatches=('is_mismatch', 'sum'))
                            )
                by_period['mismatch_rate'] = by_period['mismatches'] / by_period['total_reviews']
                if px is not None and plotly_events is not None and not by_period.empty:
                    rate_df = by_period.reset_index()
                    fig_trend = px.line(rate_df, x='period_start', y='mismatch_rate', markers=True, title='Mismatch Rate Over Time')
                    selected_points_ts = plotly_events(fig_trend, click_event=True, hover_event=False, select_event=False, override_height=450, override_width='100%')
                    if selected_points_ts:
                        sel_x = selected_points_ts[0].get('x')
                        try:
                            clicked_ts = pd.to_datetime(sel_x)
                            st.write(f"Showing mismatched reviews for period starting: {clicked_ts}")
                            period_filtered = results_df[(results_df['period_start'] == clicked_ts) & (results_df['is_mismatch'] == True)]
                            st.dataframe(period_filtered)
                        except Exception:
                            pass
                else:
                    st.line_chart(by_period['mismatch_rate'].fillna(0.0))
            except Exception as e:
                st.info(f'Unable to compute trend: {e}')

            # Suspicious Author Analysis
            st.subheader('Suspicious Author Analysis')
            min_reviews = st.number_input('Minimum reviews per author', min_value=1, max_value=1000, value=5, step=1)
            try:
                by_author = (results_df
                             .groupby('author_id')
                             .agg(total_reviews=('review_text', 'count'),
                                  mismatches=('is_mismatch', 'sum'))
                            )
                by_author['mismatch_rate'] = by_author['mismatches'] / by_author['total_reviews']
                filtered_authors = by_author[by_author['total_reviews'] >= int(min_reviews)].sort_values(['mismatch_rate', 'total_reviews'], ascending=[False, False]).head(10)
                st.dataframe(filtered_authors.reset_index())
            except Exception as e:
                st.info(f'Unable to compute author stats: {e}')

            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download Results as CSV',
                data=csv_data,
                file_name='mismatch_results.csv',
                mime='text/csv'
            )
        else:
            st.warning('No valid rows to process in the uploaded CSV.')

