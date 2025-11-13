import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from backend.model_utils import analyze_sentiment_vader, train_sentiment_model, predict_sentiment
from PIL import Image

st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")

st.title(" Twitter Sentiment Analysis Dashboard")

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose Section", ["Analyze Dataset", "Build & Evaluate Model", "Predict Sentiment"])

# --- 1. Analyze Dataset ---
if section == "Analyze Dataset":
    st.header(" Sentiment Analysis on Dataset (VADER)")

    uploaded_file = st.file_uploader("Upload a CSV file with a 'Text' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
       
        # --- Key Metrics ---
        total_tweets = len(df)
        st.write("Total Tweets : ",total_tweets)
        
        st.write("Preview of Uploaded Data", df)

        st.write("Running Sentiment Analysis...")
        if 'Tweet' in df.columns:
            results = analyze_sentiment_vader(df['Tweet'])
        elif 'Text' in df.columns:
            results = analyze_sentiment_vader(df['Text'])
        st.dataframe(results)

        sentiment_counts = results['Sentiment'].value_counts()

        positive_tweets = len(results[results['Sentiment'] == 'Positive'])
        neutral_tweets = len(results[results['Sentiment'] == 'Neutral'])
        negative_tweets = len(results[results['Sentiment'] == 'Negative'])
       
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tweets", f"{total_tweets}")
        col2.metric("Positive", f"{positive_tweets} ({positive_tweets/total_tweets:.1%})")
        col3.metric("Neutral", f"{neutral_tweets} ({neutral_tweets/total_tweets:.1%})")
        col4.metric("Negative", f"{negative_tweets} ({negative_tweets/total_tweets:.1%})")

        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
               colors=['#4CAF50', '#9E9E9E', '#F44336'], startangle=140, radius=0.7)
        ax.set_title("Sentiment Distribution in Dataset")
        st.pyplot(fig)

        st.download_button("Download Results", results.to_csv(index=False), "sentiment_results.csv", "text/csv")

# --- 2. Build & Evaluate Model ---
elif section == "Build & Evaluate Model":
    st.header(" Train and Evaluate ML Model")
    model_choice = st.selectbox("Select a Machine Learning Model", 
                                ["Logistic Regression", "Naive Bayes", "Random Forest", "SVM"])
    
    if st.button("Train Selected Model"):
        with st.spinner(f"Training {model_choice} model..."):
            model, vectorizer, acc, report, cm_path = train_sentiment_model(model_choice)
            st.success(f"{model_choice} trained successfully with accuracy: {acc*100:.2f}%")

            st.subheader(" Classification Report")
            st.text(report)

            st.subheader(" Confusion Matrix")
            image = Image.open(cm_path)
            st.image(image, caption=f"{model_choice} Confusion Matrix")

# --- 3. Predict Sentiment ---
elif section == "Predict Sentiment":
    st.header(" Predict Sentiment of Custom Texts")

    model_choice = st.selectbox("Select Model for Prediction", 
                                ["Logistic Regression", "Naive Bayes", "Random Forest", "SVM"])
    model, vectorizer, _, _, _ = train_sentiment_model(model_choice)
    user_input = st.text_area("Enter tweets (one per line):")

    if st.button("Predict Sentiments"):
        texts = [t.strip() for t in user_input.split("\n") if t.strip()]
        preds = predict_sentiment(model, vectorizer, texts)
        results = pd.DataFrame({"Tweet": texts, "Predicted Sentiment": preds})
        st.dataframe(results)
