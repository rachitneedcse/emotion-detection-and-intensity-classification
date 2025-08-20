
from fastapi import FastAPI
from pydantic import BaseModel
from emotion_module import get_emotion_intensity
from rag_module import rag_pipeline
from db_module import get_agent_email

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    answer = rag_pipeline(request.query)
    
    emotion_intensity_list = get_emotion_intensity(request.query)
    
    top_emotion, top_score = emotion_intensity_list[0]

    # Step 3: Intervention check
    intervention = None
    threshold = 0.85
    if top_score >= threshold:
        agent_email = get_agent_email()
        if agent_email:
            intervention = f"Human intervention triggered. A support agent will contact you at {agent_email}."
        else:
            intervention = "Human intervention triggered, but no agent email found."

    return {
        "bot_response": answer,
        "emotions": emotion_intensity_list,
        "intervention": intervention
    }
