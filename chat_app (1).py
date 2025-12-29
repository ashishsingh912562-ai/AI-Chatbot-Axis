import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("API key not found. Please set GOOGLE_API_KEY in .env file.")
    st.stop()

# Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Page config
st.set_page_config(page_title="Gemini Chat", page_icon="🤖")
st.title("🤖 Gemini Multi-Turn Chat")

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Gemini response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(user_prompt)
            import time

            assistant_reply = response.text
            
            placeholder = st.empty()
            typed_text = ""
            
            for char in assistant_reply:
                typed_text += char
                placeholder.markdown(typed_text)
                time.sleep(0.0002)  # typing speed (adjust)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

# Reset button
if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.messages = []
    st.experimental_rerun()





