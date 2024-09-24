import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer
from App_dangerrousness import (
    BERTClassifier,
    TextClassificationDataset,
    predict_sentiment,
)  # Import your BERTClassifier class
from Review_class_prediction_Flask import *
from config import *


# Initialize Flask app
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.route("/")
def home():
    return "Welcome to the Flask App!"


def predict_text_class(text):
    # Tokenize input text
    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # Move inputs to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model = load_model(
        bert_model_name, num_classes, batch_size, learning_rate, model_path, device
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get the predicted class
    _, predicted_class = torch.max(outputs, dim=1)
    return predicted_class.item()


# API route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Extract the text from the incoming request
    data = request.get_json()
    file_name = data.get("file_name")

    if file_name is None:
        return jsonify({"error": "No text provided"}), 400

    # Predict the class of the text
    predicted_class = predict_text_class(file_name)

    # Return the result as a JSON response
    return jsonify({"File_name": file_name, "predicted_class_file": predicted_class})


# Start the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
