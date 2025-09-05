import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np


@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = BertTokenizer.from_pretrained(r'C:\Users\rachi\ml\ai_project\best_model_bert_intensity')
    model = BertForSequenceClassification.from_pretrained(
        r'C:\Users\rachi\ml\ai_project\best_model_bert_intensity',
        use_safetensors=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Emotion labels
emotion_labels = ["anger", "fear", "joy", "sadness", "surprise"]


st.title("Emotion Intensity Prediction")
st.write("Enter text below to predict emotion intensities:")

user_input = st.text_area("Your Text:", height=100)

if st.button("Predict") and user_input.strip():
    
    inputs = tokenizer(
        user_input,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

   
    st.subheader("Predicted Intensities:")
    for i, emotion in enumerate(emotion_labels):
        st.write(f"**{emotion.capitalize()}:** {probabilities[i]:.2f}")
        st.progress(min(1.0, float(probabilities[i])))

