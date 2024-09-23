import pandas as pd
import torch 

def load_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    #chnage back
    labels = None
    print(df[:5])
    if 'label' in df.columns:
    
        labels = [1 if sentiment == "Safe" else 0 for sentiment in df['label'].tolist()]

    return texts, labels

def predict_batch(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
    return predictions