from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("spam_classifier_email.pkl")
vectorizer = joblib.load("tfidf_vectorizer_email.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    email_text = ""

    if request.method == "POST":
        if "reset" in request.form:
            return render_template("index.html", result=None, email_text="")

        email_text = request.form.get("email_text", "")

        uploaded_file = request.files.get("email_file")
        if uploaded_file and uploaded_file.filename:
            email_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if email_text.strip():
            vectorized = vectorizer.transform([email_text])
            prediction = model.predict(vectorized)[0]
            result = "Spam ❌" if prediction == 1 else "Not Spam ✅"

    return render_template("index.html", result=result, email_text=email_text)

if __name__ == "__main__":
    app.run(debug=True)
 