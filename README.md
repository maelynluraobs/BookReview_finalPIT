# 📚 Amazon Book Review Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning web application that automatically classifies Amazon book reviews into sentiment categories with **84.5% accuracy**.

## 🎯 Live Demo

**[View Live Application](https://book-review-analyzer.onrender.com)** *(Replace with your Render URL after deployment)*

if local naman run in cmd: python app.py 

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Team](#team)

## 🌟 Overview

This project analyzes Amazon book reviews and predicts sentiment using Natural Language Processing and Machine Learning. The system classifies reviews into three categories:

- 🔴 **Bad Review (Negative):** 1-2 stars
- 🟡 **Mixed Review (Neutral):** 3 stars  
- 🟢 **Great Review (Positive):** 4-5 stars

## ✨ Features

- **Real-time Sentiment Analysis** - Instant prediction with confidence scores
- **Live Review Generation** - Fetches random reviews from 29,352 Amazon book reviews
- **Modern Library Theme UI** - Beautiful, book-themed interface
- **Interactive Visualization** - Star ratings and color-coded results
- **RESTful API** - Easy integration with other applications
- **84.5% Accuracy** - Trained on actual Amazon reviews

## 🛠️ Tech Stack

### Backend
- **Python 3.11**
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **NLTK** - Natural language processing
- **pandas** - Data processing

### Machine Learning
- **Algorithm:** Logistic Regression
- **Features:** TF-IDF Vectorization (5000 features)
- **Preprocessing:** Tokenization, Lemmatization, Stopword removal

### Frontend
- **HTML5/CSS3** - Modern library theme
- **Vanilla JavaScript** - No frameworks needed
- **Responsive Design** - Works on all devices

## 📊 Dataset

- **Source:** Amazon Books Reviews
- **Size:** 29,352 reviews
- **Features:** Review text, ratings (1-5 stars), book titles
- **Labels:** Derived from star ratings

## 🚀 Installation

### Prerequisites
```bash
Python 3.11+
pip
```

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/amazon-book-review-analyzer.git
cd amazon-book-review-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

4. **Train the model** (first time only)
```bash
python train_and_save_model.py
```

5. **Run the application**
```bash
python app.py
```

6. **Open in browser**
```
http://localhost:5000
```

## 💻 Usage

### Web Interface

1. **Generate Review** - Click to load a random Amazon book review
2. **Analyze Sentiment** - Get instant prediction with confidence score
3. **Custom Input** - Type your own review for analysis
4. **View Details** - See book information, star ratings, and model confidence

### API Endpoints

#### Predict Sentiment
```bash
POST /predict_sentiment
Content-Type: application/json

{
  "text": "This book was amazing! I loved every page."
}
```

**Response:**
```json
{
  "input_text": "This book was amazing! I loved every page.",
  "predicted_sentiment": "Positive",
  "confidence": 92.45,
  "model": "Logistic Regression"
}
```

#### Get Random Review
```bash
GET /get_random_review?sentiment=positive
```

**Response:**
```json
{
  "text": "Full review text...",
  "title": "Book Title",
  "score": 5,
  "summary": "Review summary"
}
```

## 🌐 Deployment

### Deploy to Render (Free)

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Render**
- Sign up at [render.com](https://render.com)
- Connect your GitHub repository
- Select **Web Service**
- Use these settings:
  - **Build Command:** `pip install -r requirements.txt && python train_and_save_model.py`
  - **Start Command:** `gunicorn app:app`

3. **Access your app**
Your app will be live at: `https://your-app-name.onrender.com`

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

## 📁 Project Structure

```
amazon-book-review-analyzer/
├── app.py                          # Flask backend server
├── index.html                      # Frontend web interface
├── train_and_save_model.py         # Model training script
├── Books_ratings.csv               # Dataset (29,352 reviews)
├── book_review_dataset.ipynb       # Jupyter notebook
├── requirements.txt                # Python dependencies
├── sentiment_model.pkl             # Trained model (generated)
├── tfidf_vectorizer.pkl           # TF-IDF vectorizer (generated)
├── render.yaml                     # Render deployment config
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── DEPLOYMENT_GUIDE.md            # Deployment instructions
└── section-Obejero-Llatuna.md     # Full documentation
```

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **84.5%** |
| Precision (Negative) | 82% |
| Precision (Neutral) | 71% |
| Precision (Positive) | 89% |
| Recall (Negative) | 79% |
| Recall (Neutral) | 68% |
| Recall (Positive) | 92% |

### Sample Predictions

| Review | Predicted | Confidence |
|--------|-----------|------------|
| "Amazing book! Couldn't put it down" | Positive | 94.2% |
| "Terrible waste of time" | Negative | 91.7% |
| "It was okay, nothing special" | Neutral | 73.5% |

## 👥 Team

**Project Members:**
- Maelyn L. Obejero
- Angel Llatuna

**Course:** IT 414 - Elective 3 Final Project  
**Institution:** [Your University Name]  
**Date:** December 2024

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Amazon Books Reviews Dataset
- scikit-learn documentation
- Flask framework
- NLTK library

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ If you found this project helpful, please consider giving it a star!**

