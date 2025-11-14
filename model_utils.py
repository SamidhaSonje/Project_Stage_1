import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
# Ensure required resources are downloaded
nltk.download('twitter_samples', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Sentiment Analysis with VADER ---
def analyze_sentiment_vader(tweet):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for Tweet in tweet:
        score = analyzer.polarity_scores(str(Tweet))['compound']
        if score >= 0.05:
            sentiment = 'Positive'
        elif score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        results.append((Tweet, score, sentiment))
    return pd.DataFrame(results, columns=["Tweet", "Compound", "Sentiment"])

def load_training_data():
    import pandas as pd
    df = pd.read_csv("C:/Users/mp/Desktop/BE/BE Project/web_scrapping/backend/twitter_dataset_kaggle.csv")
    print(df.columns)
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Generate sentiment scores
    if 'Tweet' in df.columns:
        df["compound"] = df["Tweet"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    elif 'Text' in df.columns:
         df["compound"] = df["Text"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    def label_sentiment(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    # Label sentiment as positive, negative, or neutral
    df["sentiment"] = df["compound"].apply(label_sentiment)

    return df

# --- Train Logistic Regression Model ---
def train_sentiment_model():
    from nltk.corpus import twitter_samples
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    neu_tweets = twitter_samples.strings('neutral_tweets.json')

    df = pd.DataFrame({
        'Text': pos_tweets + neg_tweets +neu_tweets,
        'sentiment': [2]*len(neu_tweets) + [1]*len(pos_tweets) + [0]*len(neg_tweets)
    })
    return df

# --- Train a Model Dynamically ---
def train_sentiment_model(model_type="Logistic Regression"):
    df = load_training_data()

    from sklearn.utils import resample
    # Separate classes
    neg = df[df['sentiment'] == 'Negative']
    neu = df[df['sentiment'] == 'Neutral']
    pos = df[df['sentiment'] == 'Positive']
    # Find max class size
    max_size = max(len(neg), len(neu), len(pos))
    # Upsample minority classes
    neg_up = resample(neg, replace=True, n_samples=max_size, random_state=42)
    neu_up = resample(neu, replace=True, n_samples=max_size, random_state=42)
    pos_up = resample(pos, replace=True, n_samples=max_size, random_state=42)
    # Combine
    df = pd.concat([neg_up, neu_up, pos_up])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    if 'Tweet' in df.columns:
        X = vectorizer.fit_transform(df['Tweet'])
    elif 'Text' in df.columns:
         X = vectorizer.fit_transform(df['Text'])
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    y = df['sentiment'].map(label_map)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model type
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Naive Bayes":
        model = MultinomialNB()
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "SVM":
        model = LinearSVC()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'{model_type} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{model_type.replace(' ', '_')}.png")
    plt.close()

    return model, vectorizer, accuracy, report, f"outputs/confusion_matrix_{model_type.replace(' ', '_')}.png"


# --- Predict New Tweets ---
def predict_sentiment(model, vectorizer, texts):
    X_new = vectorizer.transform(texts)
    preds = model.predict(X_new)

    reverse_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return [reverse_map[p] for p in preds]
    
    
    
