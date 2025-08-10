
import os
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_DIR = r"data"
CLEAN_DF_PATH = os.path.join(DATA_DIR, "customer_support_cleaned_df.pkl")
INDEX_PATH = os.path.join(DATA_DIR, "customer_support.index")
PICKLE_PATH = os.path.join(DATA_DIR, "customer_support_df.pkl")


if not os.path.exists(CLEAN_DF_PATH):
    raise FileNotFoundError(f"Preprocessed file not found: {CLEAN_DF_PATH}")

df = pd.read_pickle(CLEAN_DF_PATH)
print(f"✅ Loaded cleaned dataset with {len(df)} rows")


# Load embedding model

model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded")




print("🔹 Creating embeddings...")
embeddings = model.encode(df["Question"].tolist(), convert_to_numpy=True, show_progress_bar=True)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (works with normalized vectors for cosine)
index.add(embeddings)
print(f"✅ FAISS index created with {index.ntotal} vectors")


os.makedirs(DATA_DIR, exist_ok=True)
faiss.write_index(index, INDEX_PATH)

with open(PICKLE_PATH, "wb") as f:
    pickle.dump(df, f)

print(f"📁 Files saved: \n- {INDEX_PATH} \n- {PICKLE_PATH}")
print("🎯 Build index complete!")
