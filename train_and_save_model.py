# Script to train the model and save the required files
# This replicates the notebook logic to create model files for the Flask API

import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("=" * 60)
print("TRAINING SENTIMENT ANALYSIS MODEL")
print("=" * 60)

# Download NLTK data
print("\n[1/7] Downloading NLTK resources...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load dataset
print("\n[2/7] Loading dataset...")
df = pd.read_csv('Books_ratings.csv')
print(f"   ✓ Loaded {len(df)} reviews")

# Create sentiment labels
print("\n[3/7] Creating sentiment labels...")
def label_sentiment(score):
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['sentiment'] = df['review/score'].apply(label_sentiment)
print(f"   ✓ Created sentiment labels")

# Text preprocessing
print("\n[4/7] Preprocessing text...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    tokens = text.split()
    # Stopword removal & Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_review'] = df['review/text'].apply(preprocess_text)
print(f"   ✓ Preprocessed {len(df)} reviews")

# Feature extraction
print("\n[5/7] Extracting features with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']
print(f"   ✓ Created TF-IDF vectors with {X.shape[1]} features")

# Train-test split
print("\n[6/7] Training Logistic Regression model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the best model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   ✓ Model trained successfully!")
print(f"   ✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Save the model and vectorizer
print("\n[7/7] Saving model files...")
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("   ✓ Saved: tfidf_vectorizer.pkl")

joblib.dump(model, 'sentiment_model.pkl')
print("   ✓ Saved: sentiment_model.pkl")

print("\n" + "=" * 60)
print("SUCCESS! Model files created.")
print("=" * 60)
print("\nYou can now:")
print("1. Restart your Flask server (python app.py)")
print("2. Open index.html in your browser")
print("3. Test the sentiment analysis!")
print("\n" + "=" * 60)
