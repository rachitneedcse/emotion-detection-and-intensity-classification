import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo_id = "rachitneedcse/best_model_bert_intensity"
# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(repo_id)
model = BertForSequenceClassification.from_pretrained(repo_id)

model.to(device)
model.eval()

emotion_labels = ["anger", "fear", "joy", "sadness", "surprise"]

def get_emotion_intensity(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
    emotion_intensity = [(emotion_labels[i], probabilities[i]) for i in range(len(probabilities))]
    emotion_intensity.sort(key=lambda x: x[1], reverse=True)
    
    intensity_predictions = [f"{emotion}: {score:.2f}" for emotion, score in emotion_intensity]
    return ", ".join(intensity_predictions)