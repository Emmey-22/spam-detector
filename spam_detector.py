import pandas as pd
import re

# Load the dataset
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

# Map labels: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean the text (lowercase + remove punctuation)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Quick sanity check
print(df.head())

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_text'])  # features
y = df['label']                                 # target labels

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to classify new messages
def predict_message(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Spam" if prediction == 1 else "Ham"

# Test with custom messages
test_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.",
    "Hey, are we still meeting for lunch today?",
    "URGENT! You’ve been selected for a FREE cruise. Call now!",
    "Hello Mum, I’ll call you after work."
]

for msg in test_messages:
    print(f"Message: {msg}")
    print("Prediction:", predict_message(msg))
    print("-" * 60)

import joblib

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
