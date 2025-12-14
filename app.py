# app.py - The Backend Web API

import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import random
import os

# --- 1. INITIALIZE APP AND LOAD COMPONENTS ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the fitted model and vectorizer
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('sentiment_model.pkl')
    print("Model and Vectorizer loaded successfully.")
except FileNotFoundError:
    # IMPORTANT: Ensure 'tfidf_vectorizer.pkl' and 'sentiment_model.pkl' are in the same directory.
    print("ERROR: Model or Vectorizer file not found. Ensure you ran the saving step in the notebook.")
    vectorizer = None
    model = None

# Load the reviews dataset
try:
    reviews_df = pd.read_csv('Books_ratings.csv')
    print(f"Reviews dataset loaded successfully. ({len(reviews_df)} reviews)")
except FileNotFoundError:
    print("ERROR: Books_ratings.csv not found.")
    reviews_df = None

# --- 2. REPLICATE PREPROCESSING LOGIC FROM NOTEBOOK ---
# Note: You need to have nltk data (stopwords, wordnet) downloaded on the server
# or ensure the necessary data is available when deploying.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters (matching the notebook logic: r'[^a-z\s]')
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    tokens = text.split()
    # Stopword removal & Lemmatization
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# --- 3. API ENDPOINT FOR PREDICTION ---

@app.route('/predict_sentiment', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500

    # Get the text input from the frontend
    data = request.get_json(force=True)
    review_text = data.get('text', '')

    if not review_text:
        return jsonify({"prediction": "No text provided"}), 200

    # 1. Preprocess the text
    clean_review = preprocess_text(review_text)

    # 2. Vectorize the text using the *fitted* TF-IDF vectorizer
    vector = vectorizer.transform([clean_review])

    # 3. Predict the sentiment
    prediction = model.predict(vector)[0]
    
    # Get prediction probabilities for confidence score
    proba = model.predict_proba(vector)[0]
    confidence = float(proba.max()) * 100  # Convert to percentage

    # 4. Return the result
    return jsonify({
        "input_text": review_text,
        "predicted_sentiment": prediction,
        "confidence": round(confidence, 2),
        "model": "Logistic Regression"
    })


@app.route('/get_random_review', methods=['GET'])
def get_random_review():
    """
    Get a random book review from the dataset. filter by sentiment using ?sentiment=positive/negative/neutral
    """
    if reviews_df is None:
        return jsonify({"error": "Reviews dataset not loaded"}), 500
    
    # Get sentiment filter from query parameters (optional)
    sentiment_filter = request.args.get('sentiment', '').lower()
    
    # Create a copy of the dataframe to filter
    filtered_df = reviews_df.copy()
    
    # Apply sentiment filter if provided
    if sentiment_filter in ['positive', 'negative', 'neutral']:
        # Map sentiment to score ranges
        if sentiment_filter == 'negative':
            filtered_df = filtered_df[filtered_df['review/score'] <= 2]
        elif sentiment_filter == 'neutral':
            filtered_df = filtered_df[filtered_df['review/score'] == 3]
        elif sentiment_filter == 'positive':
            filtered_df = filtered_df[filtered_df['review/score'] >= 4]
    
    # Get a random review
    if len(filtered_df) == 0:
        return jsonify({"error": "No reviews found with the specified filter"}), 404
    
    random_review = filtered_df.sample(n=1).iloc[0]
    
    return jsonify({
        "text": str(random_review['review/text']),
        "title": str(random_review['Title']),
        "score": int(random_review['review/score']),
        "summary": str(random_review['review/summary']) if pd.notna(random_review['review/summary']) else ""
    })


# --- 4. SERVE FRONTEND ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# --- 5. START THE SERVER ---
if __name__ == '__main__':
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get('PORT', 5000))
    # Running on http://0.0.0.0:PORT/
    app.run(host='0.0.0.0', port=port, debug=False)
