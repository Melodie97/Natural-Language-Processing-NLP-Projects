import torch
from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    try:
        data = request.json
        text = data["text"]

        # Perform sentiment analysis
        result = sentiment_analysis(text)

        return jsonify({"sentiment": result[0]["label"], "score": result[0]["score"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)