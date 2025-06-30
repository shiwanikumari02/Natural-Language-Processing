import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Text preprocessing function (move this outside load_model_and_vectorizer)
def preprocess_text(text):
    if isinstance(text, float):
        return ''
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@st.cache_resource
def load_model_and_vectorizer():
    try:
        # Construct the full path to the CSV file
        csv_path = os.path.join(script_dir, 'Comment_Classification.csv')
        
        # Load dataset with error handling
        if not os.path.exists(csv_path):
            st.error(f"Error: File not found at {csv_path}")
            st.stop()
            
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Comment'])
        
        # Now we can use the globally defined preprocess_text
        df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
        
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['Processed_Comment'])
        y = df['Sentiment']
        
        # Train classifier
        classifier = SVC(kernel='linear', random_state=42)
        classifier.fit(X, y)
        
        return vectorizer, classifier
        
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        st.stop()

vectorizer, classifier = load_model_and_vectorizer()

# Function to predict sentiment
def predict_sentiment(comment):
    processed = preprocess_text(comment)
    vec = vectorizer.transform([processed])
    prediction = classifier.predict(vec)
    return prediction[0]

# Streamlit UI
st.title("Comment Sentiment Analysis")
st.write("This app analyzes the sentiment of text comments (positive, neutral, or negative).")

# Input text area
user_input = st.text_area("Enter your comment here:", "This product is amazing!")

if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        
        # Display results with appropriate emoji
        if sentiment == "positive":
            st.success(f"Sentiment: {sentiment} 😊")
        elif sentiment == "negative":
            st.error(f"Sentiment: {sentiment} 😞")
        else:
            st.info(f"Sentiment: {sentiment} 😐")
    else:
        st.warning("Please enter a comment to analyze.")


# cd "C:\Users\User\Documents\shiwani python\Module_7">> streamlit run sentiment_app.py