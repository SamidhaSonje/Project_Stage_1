# --- Phase 1: Analyze Your Data ---
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load your dataset
df = pd.read_csv("twitter_dataset_kaggle.csv")

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Define functions to get and categorize sentiment
def get_vader_sentiment(text):
    # Ensure text is a string
    text = str(text)
    return analyzer.polarity_scores(text)['compound']

def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the functions to your 'Text' column
df['compound_score'] = df['Text'].apply(get_vader_sentiment)
df['Sentiment'] = df['compound_score'].apply(categorize_sentiment)

print("Analysis complete. Head of the new data:")
print(df[['Text', 'Sentiment', 'compound_score']].head())

# Save the augmented data
output_csv = "twitter_with_sentiment.csv"
df.to_csv(output_csv, index=False)
print(f"\nSuccessfully saved data with sentiment scores to: {output_csv}")

# --- Visualize the Results ---
sentiment_counts = df['Sentiment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
        colors=['#4CAF50', '#9E9E9E', '#F44336'], startangle=140)
plt.title('Sentiment Distribution in Your Dataset')
plt.axis('equal')
plt.savefig("sentiment_distribution_pie_chart.png")
print("Successfully saved sentiment pie chart.")
plt.show()