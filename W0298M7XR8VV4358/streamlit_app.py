import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time
import os
import tempfile
import sqlite3
from datetime import datetime
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    try:
        import sys, subprocess
        with st.spinner('Installing web scraping dependencies (requests, beautifulsoup4)...'):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'requests', 'beautifulsoup4'])
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        requests = None
        BeautifulSoup = None
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

try:
    import networkx as nx
except Exception:
    try:
        import sys, subprocess
        with st.spinner('Installing graph dependency (networkx)...'):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'networkx'])
        import networkx as nx
    except Exception:
        nx = None

try:
    from pyvis.network import Network
except Exception:
    try:
        import sys, subprocess
        with st.spinner('Installing visualization dependency (pyvis)...'):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'pyvis'])
        from pyvis.network import Network
    except Exception:
        Network = None
import pandas as pd

st.title("Star-Review Mismatch Detector")

if 'moderation_queue' not in st.session_state:
    st.session_state['moderation_queue'] = []

@st.cache_resource
def init_feedback_db(db_path: str = 'moderator_feedback.db'):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            review_text TEXT,
            text_sentiment TEXT,
            rating_sentiment TEXT,
            confidence_score REAL,
            ai_probability REAL,
            suspicion_score REAL,
            author_id TEXT,
            product_id TEXT,
            decision TEXT
        )
        """
    )
    conn.commit()
    return conn

def log_moderation_feedback(conn: sqlite3.Connection, item: dict, decision: str):
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (
                timestamp, review_text, text_sentiment, rating_sentiment,
                confidence_score, ai_probability, suspicion_score,
                author_id, product_id, decision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                item.get('review_text'),
                item.get('text_sentiment'),
                item.get('rating_sentiment'),
                float(item.get('confidence_score', 0.0)) if item.get('confidence_score') is not None else None,
                float(item.get('ai_probability', 0.0)) if item.get('ai_probability') is not None else None,
                float(item.get('suspicion_score', 0.0)) if item.get('suspicion_score') is not None else None,
                str(item.get('author_id')) if item.get('author_id') is not None else None,
                str(item.get('product_id')) if item.get('product_id') is not None else None,
                decision
            )
        )
        conn.commit()
    except Exception:
        pass

feedback_conn = init_feedback_db()

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

@st.cache_resource
def load_ai_text_detector():
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
    return pipeline('text-classification', model='roberta-base-openai-detector')

ai_text_detector = load_ai_text_detector()

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

def is_ai_generated(text: str) -> float:
    try:
        result = ai_text_detector(text)
        if isinstance(result, list) and len(result) > 0:
            label = str(result[0].get('label', '')).upper()
            score = float(result[0].get('score', 0.0))
            is_ai_label = ('FAKE' in label) or ('AI' in label) or ('AI-GENERATED' in label)
            return float(score if is_ai_label else 1.0 - score)
    except Exception:
        pass
    return 0.0

def calculate_suspicion_score(review_data: dict) -> float:
    sentiment_weight = 0.5
    ai_weight = 0.3
    sarcasm_weight = -0.2

    sentiment_confidence = float(review_data.get('sentiment_confidence', 0.0))
    ai_probability = float(review_data.get('ai_probability', 0.0))
    sarcasm_score = float(review_data.get('sarcasm_score', 0.0))

    raw_score = (
        sentiment_confidence * sentiment_weight +
        ai_probability * ai_weight +
        sarcasm_score * sarcasm_weight
    )
    # Normalize to [0, 100]
    normalized = max(0.0, min(1.0, raw_score)) * 100.0
    return float(normalized)

def build_review_graph(df: pd.DataFrame):
    if nx is None:
        raise RuntimeError('networkx is not available')
    graph = nx.Graph()
    # Ensure only necessary columns
    sub = df[['author_id', 'product_id']].dropna()
    # Add nodes for authors
    for author in sub['author_id'].unique():
        graph.add_node(str(author))
    # For each product, fully connect authors who reviewed it
    for product_id, group in sub.groupby('product_id'):
        authors = list({str(a) for a in group['author_id'].tolist()})
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a, b = authors[i], authors[j]
                if graph.has_edge(a, b):
                    # Increment weight for multiple co-reviews
                    graph[a][b]['weight'] = graph[a][b].get('weight', 1) + 1
                else:
                    graph.add_edge(a, b, weight=1, product=str(product_id))
    return graph

def scrape_reviews(url: str) -> pd.DataFrame:
    if requests is None or BeautifulSoup is None:
        raise RuntimeError('Scraping libraries are not available')
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ReviewAnalyzer/1.0)'}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        reviews = []
        # Heuristic selectors (may vary per site)
        review_blocks = soup.select('[data-review], .review, .a-section.review, .review-card, .review-item')
        if not review_blocks:
            review_blocks = soup.find_all(['article', 'div'], class_=lambda c: c and 'review' in str(c).lower())
        for block in review_blocks:
            text_el = block.find(['p', 'span', 'div'], class_=lambda c: c and ('content' in str(c).lower() or 'text' in str(c).lower())) or block.find('p')
            text = text_el.get_text(strip=True) if text_el else ''
            # Star patterns
            star = None
            star_el = block.find(['i', 'span', 'div'], attrs={'aria-label': True})
            if star_el and star_el.get('aria-label'):
                import re
                m = re.search(r"(\d+(?:\.\d+)?)\s*out of\s*5", star_el.get('aria-label'), re.I)
                if m:
                    try:
                        star = int(round(float(m.group(1))))
                    except Exception:
                        pass
            if star is None:
                possible = block.find_all(text=True)
                for t in possible:
                    s = str(t)
                    if 'out of 5' in s:
                        try:
                            val = float(s.split('out of 5')[0].strip().split()[-1])
                            star = int(round(val))
                            break
                        except Exception:
                            continue
            if text:
                reviews.append({'review_text': text, 'star_rating': int(star) if star is not None else 3})
        return pd.DataFrame(reviews)
    except Exception as e:
        st.info(f'Unable to scrape reviews: {e}')
        return pd.DataFrame(columns=['review_text', 'star_rating'])

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
        ai_prob = is_ai_generated(review_input)
        st.metric(label='AI-Generated Probability', value=f"{ai_prob*100:.1f}%")
        if ai_prob >= 0.80:
            st.warning('High likelihood of AI-generated text detected.')

        sarcasm_flag = detect_sarcasm(review_input)
        sarcasm_score = 1.0 if sarcasm_flag else 0.0
        sentiment_conf_for_score = confidence_score if base_mismatch else 0.0
        suspicion_score = calculate_suspicion_score({
            'sentiment_confidence': sentiment_conf_for_score,
            'ai_probability': ai_prob,
            'sarcasm_score': sarcasm_score,
        })
        st.metric(label='Suspicion Score', value=f"{suspicion_score:.0f}/100")

        if base_mismatch and confidence_score > confidence_threshold:
            shap_html = explain_sentiment(review_input)
            components.html(shap_html, height=200)
            if sarcasm_flag:
                st.info('ℹ️ Mismatch detected, but sarcasm may be a factor. Manual review recommended.')
        elif base_mismatch and confidence_score <= confidence_threshold:
            st.info('A potential mismatch was detected, but it is below your confidence threshold.')
            shap_html = explain_sentiment(review_input)
            components.html(shap_html, height=200)
            if sarcasm_flag:
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

    required_columns = {'review_text', 'star_rating', 'date', 'author_id', 'product_id'}
    if not required_columns.issubset(df.columns):
        st.error("CSV must contain columns: 'review_text', 'star_rating', 'date', 'author_id', and 'product_id'.")
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
            product_id = row.get('product_id')
            if pd.isna(product_id):
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
                'product_id': product_id,
                'text_sentiment': text_sentiment,
                'rating_sentiment': rating_sentiment,
                'confidence_score': float(confidence_score),
                'is_mismatch': bool(base_mismatch),
                'ai_probability': is_ai_generated(review_text),
                'suspicion_score': calculate_suspicion_score({
                    'sentiment_confidence': float(confidence_score) if base_mismatch else 0.0,
                    'ai_probability': is_ai_generated(review_text),
                    'sarcasm_score': 1.0 if detect_sarcasm(review_text) else 0.0,
                }),
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

            # Product Level Analysis
            st.header('Product Level Analysis')
            try:
                by_product = (results_df
                              .groupby('product_id')
                              .agg(total_reviews=('review_text', 'count'),
                                   mismatches=('is_mismatch', 'sum')))
                by_product['mismatch_rate'] = by_product['mismatches'] / by_product['total_reviews']
                by_product_sorted = by_product.sort_values(['mismatch_rate', 'total_reviews'], ascending=[False, False])
                st.dataframe(by_product_sorted.reset_index())

                product_options = by_product_sorted.index.astype(str).tolist()
                selected_products = st.multiselect('Select products to inspect', product_options)
                if selected_products:
                    filtered_full = results_df[results_df['product_id'].astype(str).isin(selected_products)]
                    st.dataframe(filtered_full)
            except Exception as e:
                st.info(f'Unable to compute product stats: {e}')

            # Live Moderation Queue Simulation
            st.subheader('Live Moderation Queue Simulation')
            simulate_live = st.toggle('Simulate Live Review Feed')
            if simulate_live:
                try:
                    # Stream in one new mismatched review per rerun
                    for _, row in mismatches_df.iterrows():
                        key = f"{row.get('author_id')}|{row.get('product_id')}|{row.get('date')}|{row.get('review_text')[:50]}"
                        existing_keys = {item.get('key') for item in st.session_state['moderation_queue']}
                        if key not in existing_keys:
                            st.session_state['moderation_queue'].append({
                                'key': key,
                                'review_text': row.get('review_text'),
                                'star_rating': int(row.get('star_rating')) if pd.notna(row.get('star_rating')) else None,
                                'text_sentiment': row.get('text_sentiment'),
                                'rating_sentiment': row.get('rating_sentiment'),
                                'confidence_score': float(row.get('confidence_score', 0.0)),
                                'ai_probability': float(row.get('ai_probability', 0.0)),
                                'author_id': row.get('author_id'),
                                'product_id': row.get('product_id'),
                                'date': row.get('date')
                            })
                            time.sleep(2)
                            st.rerun()
                            break
                except Exception as e:
                    st.info(f'Unable to stream live reviews: {e}')

            # Display Moderation Queue
            if st.session_state['moderation_queue']:
                st.subheader('Moderation Queue')
                for idx, item in enumerate(list(st.session_state['moderation_queue'])):
                    with st.expander(item.get('review_text', 'Review')):
                        st.write({'author_id': item.get('author_id'),
                                  'product_id': item.get('product_id'),
                                  'date': item.get('date'),
                                  'star_rating': item.get('star_rating'),
                                  'text_sentiment': item.get('text_sentiment'),
                                  'rating_sentiment': item.get('rating_sentiment'),
                                  'confidence_score': item.get('confidence_score'),
                                  'ai_probability': item.get('ai_probability')})
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button('Approve', key=f"approve_{item['key']}"):
                                try:
                                    log_moderation_feedback(feedback_conn, item, 'Approved')
                                finally:
                                    st.session_state['moderation_queue'].pop(idx)
                                    st.rerun()
                        with col2:
                            if st.button('Reject', key=f"reject_{item['key']}"):
                                try:
                                    log_moderation_feedback(feedback_conn, item, 'Rejected')
                                finally:
                                    st.session_state['moderation_queue'].pop(idx)
                                    st.rerun()

            # Fraud Ring Detection
            st.header("Fraud Ring Detection")
            if Network is None or nx is None:
                st.info('Graph libraries not available. Try rerunning to auto-install dependencies.')
            else:
                if st.button('Generate Fraud Network Graph'):
                    try:
                        graph = build_review_graph(results_df)
                        communities = list(nx.algorithms.community.greedy_modularity_communities(graph))

                        # Build pyvis network
                        net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#333333', notebook=False, directed=False)
                        # Assign community colors
                        community_map = {}
                        for idx_c, comm in enumerate(communities):
                            for node in comm:
                                community_map[node] = idx_c
                        for node in graph.nodes():
                            color_idx = community_map.get(node, -1)
                            color = px.colors.qualitative.Plotly[color_idx % len(px.colors.qualitative.Plotly)] if (px is not None and color_idx >= 0) else None
                            net.add_node(node, label=node, color=color)
                        for u, v, data in graph.edges(data=True):
                            net.add_edge(u, v, value=data.get('weight', 1))

                        with tempfile.TemporaryDirectory() as tmpdir:
                            html_path = os.path.join(tmpdir, 'fraud_network.html')
                            net.show(html_path)
                            with open(html_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            components.html(html_content, height=650, scrolling=True)

                        # Display communities table
                        comm_rows = []
                        for idx_c, comm in enumerate(communities):
                            comm_rows.append({'community_id': idx_c, 'authors': sorted(list(comm))})
                        if comm_rows:
                            comm_df = pd.DataFrame(comm_rows)
                            st.dataframe(comm_df)
                        else:
                            st.info('No communities detected.')
                    except Exception as e:
                        st.info(f'Unable to generate fraud graph: {e}')

            # Campaign Anomaly Detection
            st.header("Campaign Anomaly Detection")
            try:
                # Temporal patterns by hour
                if 'timestamp' in df.columns:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    except Exception:
                        pass
                if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    hourly = df.dropna(subset=['timestamp']).copy()
                    hourly['hour'] = hourly['timestamp'].dt.hour
                    st.subheader('Reviews by Hour of Day')
                    if px is not None:
                        fig_hour = px.bar(hourly.groupby('hour').size().reset_index(name='count'), x='hour', y='count', title='Hourly Review Volume (All Products)')
                        st.plotly_chart(fig_hour, use_container_width=True)
                    else:
                        st.bar_chart(hourly.groupby('hour').size())

                # Geo patterns map
                st.subheader('Geographic Patterns')
                if 'author_location' in df.columns:
                    # Expect columns lat, lon if available, else skip plotting
                    lat_lon_cols = {'lat', 'latitude', 'lon', 'lng', 'longitude'}
                    has_lat = any(col in df.columns for col in ['lat', 'latitude'])
                    has_lon = any(col in df.columns for col in ['lon', 'lng', 'longitude'])
                    if has_lat and has_lon:
                        prod_options = sorted(results_df['product_id'].astype(str).unique().tolist()) if 'product_id' in results_df.columns else []
                        selected_prod = st.selectbox('Select product to map', prod_options) if prod_options else None
                        map_df = results_df.copy()
                        if selected_prod is not None:
                            map_df = map_df[map_df['product_id'].astype(str) == selected_prod]
                        # Rename to lat/lon if necessary
                        if 'latitude' in df.columns and 'lat' not in df.columns:
                            df = df.rename(columns={'latitude': 'lat'})
                        if 'longitude' in df.columns and 'lon' not in df.columns:
                            df = df.rename(columns={'longitude': 'lon'})
                        if 'lng' in df.columns and 'lon' not in df.columns:
                            df = df.rename(columns={'lng': 'lon'})
                        if {'lat', 'lon'}.issubset(df.columns):
                            st.map(df[['lat', 'lon']].dropna())
                    else:
                        st.info('Location coordinates not found. Provide lat/lon columns to enable the map.')

                # Burst score per product
                st.subheader('Burst Scores by Product')
                if 'timestamp' in df.columns and 'product_id' in results_df.columns:
                    tmp = results_df.copy()
                    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        try:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        except Exception:
                            pass
                    # Join timestamps from original df if results_df lacks it
                    if 'timestamp' not in results_df.columns:
                        tmp = tmp.merge(df[['review_text', 'timestamp']], on='review_text', how='left')
                    tmp = tmp.dropna(subset=['timestamp']).copy()
                    tmp['hour_bucket'] = tmp['timestamp'].dt.floor('H')
                    burst = (tmp.groupby(['product_id', 'hour_bucket'])
                               .size()
                               .reset_index(name='count'))
                    burst_scores = burst.groupby('product_id')['count'].max().reset_index(name='burst_score')
                    burst_scores = burst_scores.sort_values('burst_score', ascending=False)
                    st.dataframe(burst_scores)
            except Exception as e:
                st.info(f'Unable to compute anomaly detection: {e}')

            # Competitive Intelligence
            st.header('Competitive Intelligence')
            comp_url = st.text_input('Paste competitor product URL')
            if comp_url:
                if st.button('Scrape and Analyze'):
                    comp_df = scrape_reviews(comp_url)
                    if not comp_df.empty:
                        st.write('Scraped Reviews Preview:')
                        st.dataframe(comp_df.head(20))
                        # Reuse the batch pipeline on scraped data with minimal columns
                        # Add placeholders for required fields
                        comp_df['date'] = pd.Timestamp.utcnow().normalize()
                        comp_df['author_id'] = 'unknown'
                        comp_df['product_id'] = 'competitor'
                        # Process as if uploaded
                        st.write('Running analysis on scraped data...')
                        # Minimal reuse: iterate and compute
                        comp_results = []
                        for _, row in comp_df.iterrows():
                            review_text = str(row['review_text'])
                            try:
                                star_rating = int(row['star_rating'])
                            except Exception:
                                star_rating = 3
                            text_sentiment, confidence_score = analyze_text_sentiment_with_score(classifier, review_text)
                            rating_sentiment = map_rating_to_sentiment(star_rating)
                            base_mismatch = ((text_sentiment == 'POSITIVE' and rating_sentiment == 'NEGATIVE') or
                                             (text_sentiment == 'NEGATIVE' and rating_sentiment == 'POSITIVE'))
                            comp_results.append({
                                'review_text': review_text,
                                'star_rating': star_rating,
                                'date': row['date'],
                                'author_id': row['author_id'],
                                'product_id': row['product_id'],
                                'text_sentiment': text_sentiment,
                                'rating_sentiment': rating_sentiment,
                                'confidence_score': float(confidence_score),
                                'is_mismatch': bool(base_mismatch),
                                'ai_probability': is_ai_generated(review_text),
                                'topic': None,
                            })
                        comp_results_df = pd.DataFrame(comp_results)
                        st.session_state['results_df_competitor'] = comp_results_df.copy()
                        st.dataframe(comp_results_df)
                        # Basic topic classification on mismatches
                        try:
                            comp_mismatch = comp_results_df[comp_results_df['is_mismatch'] == True]
                            if not comp_mismatch.empty:
                                topics = []
                                for text in comp_mismatch['review_text'].tolist():
                                    try:
                                        z = zero_shot_classifier(text, candidate_labels)
                                        topics.append(z.get('labels', ['Unknown'])[0] if isinstance(z, dict) else 'Unknown')
                                    except Exception:
                                        topics.append('Unknown')
                                comp_mismatch = comp_mismatch.copy()
                                comp_mismatch['topic'] = topics
                                st.subheader('Competitor: Top Topics in Mismatches')
                                st.dataframe(comp_mismatch['topic'].value_counts().reset_index().rename(columns={'index': 'topic', 'topic': 'count'}))
                        except Exception:
                            pass
    # Admin / Model Training Section
    st.header('Model Training')
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button('Export Training Data'):
            try:
                cur = feedback_conn.cursor()
                cur.execute('SELECT timestamp, review_text, text_sentiment, rating_sentiment, confidence_score, ai_probability, suspicion_score, author_id, product_id, decision FROM feedback')
                rows = cur.fetchall()
                export_df = pd.DataFrame(rows, columns=['timestamp', 'review_text', 'text_sentiment', 'rating_sentiment', 'confidence_score', 'ai_probability', 'suspicion_score', 'author_id', 'product_id', 'decision'])
                if not export_df.empty:
                    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download Labeled Data CSV', data=csv_bytes, file_name='moderator_feedback_export.csv', mime='text/csv')
                else:
                    st.info('No feedback data available yet.')
            except Exception as e:
                st.info(f'Unable to export training data: {e}')
    with col_b:
        if st.button('Fine-tune Model'):
            st.info('Fine-tuning job triggered. In a production setup, this would call a training script with the exported data.')
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download Results as CSV',
                data=csv_data,
                file_name='mismatch_results.csv',
                mime='text/csv'
            )
        else:
            st.warning('No valid rows to process in the uploaded CSV.')
