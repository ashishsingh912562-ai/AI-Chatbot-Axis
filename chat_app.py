import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Page config
st.set_page_config(page_title="XSpark AI", page_icon="⚡", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Setting")
    
    # Model Selection
    model_name = st.selectbox(
        "Select Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )
    
    # Parameters
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 8192, 2048)
    
    # System Prompt
    system_instruction = st.text_area(
        "System Persona", 
        value="You are a helpful AI assistant named XSpark.",
        height=100
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat = None # Force chat reset
        st.rerun()
        
    st.markdown("---")
    st.markdown("**Design & Developed by: Ashish Singh**")

# API Setup
if not API_KEY:
    st.error("API key not found. Please set GOOGLE_API_KEY in .env file.")
    st.stop()

genai.configure(api_key=API_KEY)

# Initialize Chat Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI
st.title("⚡ XSpark 🤖")
st.caption("Powered by Google Gemini")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"], width=200)

# Image Uploader
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

# Chat Input
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    # 1. Handle User Input
    user_message_content = user_prompt
    image_data = None
    
    # If there is an image, process it
    if uploaded_file:
        image_data = Image.open(uploaded_file)
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt, "image": image_data}
        )
    else:
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )

    # Display User Message immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)
        if image_data:
            st.image(image_data, width=300)

    # 2. Generate Response (with True Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Initialize model with current config
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )

            # Prepare inputs (Handling Text + Image vs Text only)
            if image_data:
                # Direct generation for images (chat history for images is complex, simplified here)
                response = model.generate_content(
                    [user_prompt, image_data],
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                    stream=True
                )
            else:
                # Use ChatSession for history
                # Rebuild history for the API (Gemini API format)
                history_for_api = [
                    {"role": m["role"], "parts": [m["content"]]} 
                    for m in st.session_state.messages[:-1] 
                    if "image" not in m # Filtering out images from history for simplicity in this basic version
                ]
                chat = model.start_chat(history=history_for_api)
                response = chat.send_message(
                    user_prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                    stream=True
                )

            # Stream the chunks
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            full_response = "I encountered an error. Please try again."

    # 3. Save Assistant Response to History
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )








