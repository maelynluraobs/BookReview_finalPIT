# ==============================================================================
# IMPORTANT: Add this code to your Jupyter notebook to save the model files
# Run this cell AFTER training your models (after cell with model training)
# ==============================================================================

import joblib

# Get the best model from your models dictionary
# Based on your notebook, Logistic Regression has the best accuracy (0.844)
best_model = models['Logistic Regression']

# Save the fitted TfidfVectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✓ Saved: tfidf_vectorizer.pkl")

# Save the trained Logistic Regression model
joblib.dump(best_model, 'sentiment_model.pkl')
print("✓ Saved: sentiment_model.pkl")

print("\nModel files saved successfully!")
print("You can now run app.py to start the Flask API.")
