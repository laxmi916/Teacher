import streamlit as st
from google import genai
from google.genai import types
import edge_tts
import asyncio
import io
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AI Teacher - Kids AI Tutor", page_icon="🎨")
st.title("AI Teacher")
st.write("Click the mic to speak or type a message below!")

# --- 2. SETUP PERSISTENT GEMINI CLIENT & CHAT ---
# On Streamlit Cloud, store this key in Secrets Management (st.secrets["GEMINI_API_KEY"])
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are 'AI Teacher', a warm, patient AI tutor helping Telugu-speaking kids learn English.

Rules:
1. CORRECTION TRIGGER: If input is in Telugu, Tanglish, or broken English, teach the correct natural English version first before answering (e.g., "Ela unnav?" -> "In English: 'Hello, how are you?'. I am doing great!").
2. CORRECT INPUT: If input is correct English, answer directly without correction.
3. NO DUPLICATION: Never repeat translations in both scripts simultaneously (use EITHER "రావడం" OR "raavadam", not both).
4. PRAISE: Use "Super try!" for English praise and "చాలా బాగా చెప్పావు" for Telugu praise. No generic filler praise.
5. FORMAT: Keep responses under 2-3 short sentences and award stars (e.g., "⭐️ +10 Stars!").
"""

if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=GEMINI_API_KEY)

if "chat_session" not in st.session_state:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.7,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )
    st.session_state.chat_session = st.session_state.client.chats.create(
        model="gemini-2.5-flash",
        config=config
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_processed_id" not in st.session_state:
    st.session_state.last_processed_id = None

# --- 3. HELPER FUNCTIONS ---
def text_to_speech_bytes(text):
    """Generates audio in-memory to prevent disk file-lock issues on Streamlit Cloud."""
    async def _async_generate():
        tts = edge_tts.Communicate(text=text, voice="te-IN-MohanNeural")
        audio_stream = io.BytesIO()
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        audio_stream.seek(0)
        return audio_stream
    
    return asyncio.run(_async_generate())

def get_response_from_gemini(user_text=None, audio_bytes=None):
    """Sends user text or in-memory audio bytes to Gemini."""
    contents = []
    
    if audio_bytes:
        audio_part = types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav"
        )
        contents.append(audio_part)
        contents.append("Listen to the child's recorded voice input above. Respond as AI Teacher.")
    elif user_text:
        contents.append(user_text)

    response = st.session_state.chat_session.send_message(contents)
    return response.text

# --- 4. USER INPUT UI ---
try:
    from streamlit_mic_recorder import mic_recorder
    #st.subheader("🎤 Speak to AI Teacher:")
    audio_data = mic_recorder(
        start_prompt="Speak 🎙️",
        stop_prompt="Click to Stop & Send 🛑",
        key="recorder",
        just_once=False
    )
except ImportError:
    st.warning("`streamlit-mic-recorder` not found. Install it to enable microphone input.")
    audio_data = None

text_input = st.text_input("Or type your message (Telugu or English):", key="text_field")
submit_text = st.button("Send Text")

# --- 5. PROCESS INPUT & RESPOND ---
user_audio_bytes = None
user_text_msg = None

if audio_data and "bytes" in audio_data and len(audio_data["bytes"]) > 0:
    audio_id = audio_data.get("id")
    if audio_id != st.session_state.last_processed_id:
        user_audio_bytes = audio_data["bytes"]
        st.session_state.last_processed_id = audio_id

elif submit_text and text_input:
    user_text_msg = text_input

if user_audio_bytes or user_text_msg:
    with st.spinner("AI Teacher is listening..."):
        try:
            # 1. Fetch text response from Gemini
            ai_reply = get_response_from_gemini(
                user_text=user_text_msg, 
                audio_bytes=user_audio_bytes
            )
            
            # 2. Synthesize Speech to bytes buffer
            audio_buffer = text_to_speech_bytes(ai_reply)
            
            # 3. Track chat history with in-memory audio
            st.session_state.chat_history.append({
                "role": "Child",
                "text": user_text_msg if user_text_msg else "🎤 [Spoken Audio]",
                "audio": None
            })
            st.session_state.chat_history.append({
                "role": "AI Teacher",
                "text": ai_reply,
                "audio": audio_buffer
            })
            
        except Exception as e:
            st.error(f"Error connecting to Gemini API: {str(e)}")

# --- 6. CHAT DISPLAY ---
total_messages = len(st.session_state.chat_history)

for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "Child":
        st.chat_message("user").write(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["text"])
            is_latest = (idx == total_messages - 1)
            if msg["audio"] is not None:
                audio_container = st.container()
                with audio_container:
                    st.audio(
                        msg["audio"], 
                        format="audio/mp3", 
                        autoplay=is_latest
                    )
