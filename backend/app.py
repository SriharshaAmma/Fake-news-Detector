from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🧠 Fake News Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Clean and transform text
        cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        # Define simple logic for real/fake classification
        fake_labels = ["fake", "left-news", "right-news", "satire"]
        real_labels = ["real", "reliable", "politics", "worldnews"]

        # Determine explanation
        text_lower = text.lower()
        if prediction in fake_labels:
            result = "Fake News ❌"
            if any(word in text_lower for word in ["miracle", "cure", "secret", "shocking", "unbelievable"]):
                reason = "Because it contains exaggerated or sensational claims."
            elif "scientist" in text_lower or "research" in text_lower:
                reason = "Because it makes scientific claims without credible evidence."
            else:
                reason = "Because it lacks reliable or verifiable information."
        else:
            result = "Real News ✅"
            if "government" in text_lower or "official" in text_lower:
                reason = "Because it reports verifiable facts or official statements."
            else:
                reason = "Because it appears factual and unbiased."

        return jsonify({
            "prediction": result,
            "reason": reason
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
