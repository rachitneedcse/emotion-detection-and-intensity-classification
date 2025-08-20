import streamlit as st
import requests

st.title("Customer Support Chatbot")


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:")

if st.button("Send") and user_input.strip():
    response = requests.post("http://127.0.0.1:8000/chat", json={"query": user_input})
    data = response.json()
    
    bot_reply = data.get("bot_response", "Sorry, no response.")
    emotion_intensity = data.get("emotions", "No emotion data.")
    intervention_message = data.get("intervention", None)
    st.session_state.chat_history.append({
        "user": user_input,
        "bot": bot_reply,
        "emotion": emotion_intensity,
        "intervention": intervention_message
    })

# Display chat history
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown(f"*Emotion intensities:* {chat['emotion']}")
    st.markdown(f"*Intervention:* {chat['intervention']}" if chat['intervention'] else "")
    st.markdown("---")