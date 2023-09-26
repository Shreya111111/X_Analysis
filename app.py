from flask import Flask, render_template, request, jsonify
import torch  # Add this line to import torch

# Import the sentiment analysis model and tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# Load the sentiment analysis model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name,static_url_path='/static')

# Define a function for sentiment analysis
def analyze_sentiment(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    sentiments = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiments[sentiment]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        tweet = request.form.get("tweet")
        result = analyze_sentiment(tweet)
    return render_template("index.html", result=result)

@app.route("/analyze", methods=["POST"])  # Add this route for tweet analysis
def analyze():
    tweet = request.form.get("tweet")
    sentiment = analyze_sentiment(tweet)
    return jsonify(sentiment)

if __name__ == "__main__":
    app.run(debug=True)
