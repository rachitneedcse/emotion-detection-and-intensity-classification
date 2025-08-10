
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "customer_support.index")
PICKLE_PATH = os.path.join(DATA_DIR, "customer_support_df.pkl")


if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)


if not os.path.exists(PICKLE_PATH):
    raise FileNotFoundError(f"DataFrame pickle not found: {PICKLE_PATH}")
with open(PICKLE_PATH, "rb") as f:
    df = pickle.load(f)


model = SentenceTransformer("all-MiniLM-L6-v2")


def get_rag_response(query, k=3, score_threshold=0.5):
    """
    Retrieve top-k answers from FAISS index with confidence scores.

    Args:
        query (str): User question
        k (int): Number of top results to return
        score_threshold (float): Minimum cosine similarity score (0-1)

    
    """
    # Encode & normalize query for cosine similarity
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    scores, indices = index.search(query_embedding, k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < score_threshold:
            continue
        return df.iloc[idx]['Answer']

    return "Sorry, I couldn't find a good match."
