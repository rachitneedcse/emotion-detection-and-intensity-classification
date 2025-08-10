
from fastapi import FastAPI
from pydantic import BaseModel
from emotion_module import get_emotion_intensity
from rag_module import get_rag_response

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    answer = get_rag_response(request.query)
    emotion_intensity_str = get_emotion_intensity(request.query)
    return {
        "bot_response": answer,
        "emotion_intensity": emotion_intensity_str
    }
