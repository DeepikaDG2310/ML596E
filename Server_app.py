import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer
from App_dangerrousness import BERTClassifier
from config import *  # Ensure this contains your model parameters

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(bert_model_name, num_classes, batch_size, learning_rate, model_path, device):
    model = BERTClassifier(bert_model_name, num_classes, batch_size, learning_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model(bert_model_name, num_classes, batch_size, learning_rate, model_path, device)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    
    if text is None:
        return jsonify({'error': 'No text provided'}), 400
    
    pred_class = predict_text_class(text)
    predicted_class = "Safe" if pred_class == 1 else "not-safe"
    
    return jsonify({
        'text': text,
        'predicted_class': predicted_class
    })

def predict_text_class(text):
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    _, predicted_class = torch.max(outputs, dim=1)
    return predicted_class.item()

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
