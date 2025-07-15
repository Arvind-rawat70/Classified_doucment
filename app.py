import streamlit as st
import pandas as pd
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os

# Download stopwords if not present
nltk.download('stopwords')

# === Text Preprocessing ===
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def cleaned_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# === Streamlit App ===
st.title("🛡️ Classified Document Detector")
st.markdown("Upload a dataset or use the default 1000-sample dataset to train. Then test it with your own text.")

# File upload or fallback to default CSV
uploaded_file = st.file_uploader("📁 Upload your CSV (must have 'text' and 'label' columns)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset uploaded successfully.")
else:
    default_path = "data/classified_documents_1000.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info("ℹ️ Using default dataset: `classified_documents_1000.csv`")
    else:
        st.error("❌ Default dataset not found. Please upload a CSV.")
        st.stop()

# Show dataset preview
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Train model
if st.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        # Check required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("❌ The dataset must contain 'text' and 'label' columns.")
        else:
            # Preprocess text and encode labels
            df['cleaned_text'] = df['text'].apply(cleaned_text)
            label_encoder = LabelEncoder()
            df['encoded_label'] = label_encoder.fit_transform(df['label'])

            X = df['cleaned_text']
            y = df['encoded_label']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

            # Build pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
            ])

            pipeline.fit(X_train, y_train)

            # Save model
            joblib.dump(pipeline, "model.pkl")
            joblib.dump(label_encoder, "label_encoder.pkl")

            st.success("✅ Model trained and saved successfully!")

# Prediction Section
st.subheader("🧪 Test on New Document")

user_text = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Classify"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            model = joblib.load("model.pkl")
            label_encoder = joblib.load("label_encoder.pkl")

            cleaned = cleaned_text(user_text)
            prediction = model.predict([cleaned])
            label = label_encoder.inverse_transform(prediction)[0]

            st.markdown(f"🔒 **Predicted Label:** `{label}`")

        except FileNotFoundError:
            st.error("❌ Model not found. Please train the model first.")
