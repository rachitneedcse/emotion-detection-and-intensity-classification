
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import together
together.api_key = "tgp_v1_hcAYeb5IquESKVQOsx6_wbAn0jkRNJTWHnNipFUTIlI"
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


model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4")

def combine_and_reword_answers(answers):
    answers = [res['answer'] for res in answers]

    prompt = f"""
You are a helpful and friendly customer support assistant.

Here are several answers retrieved from the knowledge base:
{answers}

Please combine and reword these answers into a single, clear, and concise response suitable for a customer.

If the topic requires instructions, provide step-by-step guidance in a simple and easy-to-follow manner.

Do NOT include any internal reasoning, commentary, or repeated information.

Avoid repeating any characters, lines, or phrases.

Keep the tone polite, professional, and approachable.

Provide only the final customer-facing answer.
"""



    # Generate reworded answer
    response = together.Complete.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        prompt=prompt,
        max_tokens=400,
        temperature=0.3
    )

    return response["choices"][0]["text"].strip()



def retrieve_top_answers(query, top_k=3, score_threshold=0.5):
    
    # Encode & normalize query for cosine similarity
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    scores, indices = index.search(query_embedding, top_k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < score_threshold:
            continue
        results.append({
            "answer": df.iloc[idx]['Answer'],
            "score": float(score)
        })

    if not results:
        return "Sorry, I couldn't find a good match."

    return results


def rag_pipeline(query):
    top_answers = retrieve_top_answers(query, top_k=3)
    merged_answer = combine_and_reword_answers(top_answers)
    return merged_answer
