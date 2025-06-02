import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("spam_classifier_email.pkl")
vectorizer = joblib.load("tfidf_vectorizer_email.pkl")

# Web app title
st.title("📧 Email Spam Detector (Naive Bayes)")
st.write("Enter an email message below to classify it as spam or not.")

# Text input
email_text = st.text_area("Enter email content:", height=200)

if st.button("Classify"):
    if email_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        text_vectorized = vectorizer.transform([email_text])
        prediction = model.predict(text_vectorized)[0]
        result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
        st.success(f"Prediction: **{result}**")
