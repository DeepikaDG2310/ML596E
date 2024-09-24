import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer
from App_dangerrousness import BERTClassifier
from config import *  # Ensure this contains your model parameters
import pandas as pd
import io

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    bert_model_name, num_classes, batch_size, learning_rate, model_path, device
):
    model = BERTClassifier(bert_model_name, num_classes, batch_size, learning_rate).to(
        device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model


model = load_model(
    bert_model_name, num_classes, batch_size, learning_rate, model_path, device
)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file:
        return jsonify({"error": "Invalid file"}), 400

    # Read the file into a pandas DataFrame
    try:
        df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
    except Exception as e:
        return jsonify({"error": "Failed to read CSV file", "message": str(e)}), 400

    if "review" not in df.columns:
        return jsonify({"error": 'CSV must contain a "review" column'}), 400

    # Process each review in the CSV
    predictions = []
    for review in df["review"]:
        pred_class = predict_text_class(review)
        predicted_class = "Safe" if pred_class == 1 else "not-safe"
        predictions.append({"review": review, "predicted_class": predicted_class})

    # Return the predictions
    return jsonify(predictions)


def predict_text_class(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    _, predicted_class = torch.max(outputs, dim=1)
    return predicted_class.item()


# Start the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
