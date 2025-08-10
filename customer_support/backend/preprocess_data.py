# preprocess_dataset.py
import pandas as pd
import re
import string
import pickle
import os

# Paths
DATA_DIR = r"data"
RAW_CSV = DATA_DIR + r"\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
CLEAN_DF_PATH = DATA_DIR + r"\customer_support_cleaned_df.pkl"

os.makedirs(DATA_DIR, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load and clean dataset
df = pd.read_csv(RAW_CSV)
df = df[['instruction', 'response']].rename(columns={'instruction': 'Question', 'response': 'Answer'})
df['Question'] = df['Question'].apply(clean_text)
df['Answer'] = df['Answer'].apply(clean_text)
df = df[df['Question'] != ""]
df = df[df['Answer'] != ""]


with open(CLEAN_DF_PATH, "wb") as f:
    pickle.dump(df, f)

print(f"✅ Cleaned dataset saved to {CLEAN_DF_PATH} with {len(df)} rows")
