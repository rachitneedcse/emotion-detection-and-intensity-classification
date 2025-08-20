import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

# j-hartmann original labels
jh_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Mapping to your 5 emotions
label_mapping = {
    "anger": "Anger",
    "disgust": "Disapproval",
    "fear": "Confusion",
    "sadness": "Disappointment",
    "surprise": "Annoyance",
    "joy": None,      # ignored
    "neutral": None   # ignored
}

target_labels = ["Anger", "Annoyance", "Confusion", "Disappointment", "Disapproval"]

def get_emotion_intensity(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
    
    mapped_logits = []
    for jh_label in jh_labels:
        mapped = label_mapping[jh_label]
        if mapped:
            mapped_logits.append(logits[jh_labels.index(jh_label)].item())
    
    # Apply softmax over mapped logits
    probs = F.softmax(torch.tensor(mapped_logits), dim=0)

    mapped_probs = dict(zip(target_labels, probs.tolist()))
    # Sort by intensity
    emotion_intensity = sorted(mapped_probs.items(), key=lambda x: x[1], reverse=True)

    return emotion_intensity
