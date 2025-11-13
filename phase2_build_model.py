# --- Phase 2: Build Your Own ML Model ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import twitter_samples, stopwords

# Import modules from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# --- 1. Load and Prepare Data ---
nltk.download('twitter_samples')
nltk.download('stopwords')

# Load the positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

pos_df = pd.DataFrame({'text': positive_tweets, 'sentiment': 1})
neg_df = pd.DataFrame({'text': negative_tweets, 'sentiment': 0})
df = pd.concat([pos_df, neg_df], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Loaded {len(df)} labeled tweets for training.")
print(df.head())

# --- 2. Feature Extraction (Text to Numbers) ---
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X = vectorizer.fit_transform(df['text']) # The features (the text)
y = df['sentiment']                       # The target (the label)

print(f"Text converted to a feature matrix of shape: {X.shape}")

# --- 3. Split Data for Training and Testing ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# --- 4. Train the ML Model ---
print("\nTraining the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print a detailed report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# --- 6. Visualize Evaluation (Confusion Matrix) ---
print("Generating Confusion Matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                      display_labels=['Negative', 'Positive'],
                                      cmap='Blues',
                                      ax=ax)
ax.set_title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
print("Confusion Matrix plot saved as: confusion_matrix.png")
plt.show()