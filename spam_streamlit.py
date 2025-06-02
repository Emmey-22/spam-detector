import streamlit as st
import joblib
import re
import pandas as pd

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text


# UI layout
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")
st.title("📧 Spam Detector (Naive Bayes)")
st.markdown("Classify messages as **Spam** or **Ham** using a trained machine learning model.")

# --- Option 1: Single message input ---
st.header("🔍 Check a Single Message")

message = st.text_area("Enter your message below:")

if st.button("Check Message"):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    if prediction == 1:
        st.error("🚫 Spam Detected")
    else:
        st.success("✅ This message is Ham")

# --- Option 2: Bulk Upload ---
st.header("📂 Bulk Check via File Upload")
uploaded_file = st.file_uploader("Upload a CSV or TXT file with messages", type=["csv", "txt"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            # For .txt files, assume one message per line
            df = pd.read_csv(uploaded_file, header=None, names=["message"])

        if "message" not in df.columns:
            st.warning("❗ The file must contain a column named 'message'")
        else:
            df["cleaned"] = df["message"].apply(clean_text)
            X = vectorizer.transform(df["cleaned"])
            df["prediction"] = model.predict(X)
            df["label"] = df["prediction"].map({0: "Ham ✅", 1: "Spam 🚫"})

            st.success("✅ Prediction completed!")
            st.dataframe(df[["message", "label"]])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", csv, "spam_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
