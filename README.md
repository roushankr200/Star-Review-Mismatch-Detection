## Star-Review Mismatch Detector

Detect reviews where the written sentiment and the star rating do not match (e.g., 5⭐ with negative text). Built with Streamlit and Hugging Face Transformers.

### Features
- **NLP sentiment analysis** using a pre-trained transformer pipeline
- **Rating-to-sentiment mapping** (1–2: Negative, 3: Neutral, 4–5: Positive)
- **Mismatch detection** and clear UI feedback

### Project Structure
- `review_analyzer.py`: Streamlit app and core logic
- `requirements.txt`: Python dependencies

### Prerequisites
- Python 3.10+ recommended
- Internet access on first run (to download the sentiment model)

### Setup
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run
```bash
streamlit run review_analyzer.py
```
Then open the app URL shown in the terminal (typically `http://localhost:8501`).

### How It Works
1. Load a cached Hugging Face `pipeline('sentiment-analysis')`
2. Analyze review text → text sentiment (`POSITIVE`/`NEGATIVE`/model-dependent)
3. Map star rating (1–5) → sentiment label
4. Flag mismatch if text is POSITIVE but rating is NEGATIVE, or vice versa

### Usage
1. Enter review text
2. Select star rating (1–5)
3. Click "Analyze Review"
4. View text sentiment, rating sentiment, and mismatch status

### Notes
- On the first run, the model is downloaded automatically. This can take ~1–2 minutes depending on your connection.
- The app uses CPU by default; no GPU setup is required.

### Troubleshooting
- "ModuleNotFoundError: transformers/torch":
  - Run: `python -m pip install -r requirements.txt`
- Port already in use:
  - Run: `streamlit run review_analyzer.py --server.port 8502`
- Slow first inference:
  - Subsequent inferences will be faster thanks to model caching.

### License
MIT (replace or update as needed).

