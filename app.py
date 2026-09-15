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

# --- 2. SETUP PERSISTENT GEMINI CLIENT & CHAT ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are 'AI Teacher', a warm, patient AI tutor helping Telugu-speaking kids learn English.

Rules:
1. CORRECTION TRIGGER: If input is in Telugu, Tanglish, or broken English, teach the correct natural English version first before answering (e.g., "Ela unnav?" -> "In English: 'Hello, how are you?'. I am doing great!").
2. CORRECT INPUT: If input is correct English, answer directly without correction.
3. NO DUPLICATION: Never repeat translations in both scripts simultaneously (use EITHER "రావడం" OR "raavadam", not both).
4. FORMAT: Keep responses under 2-3 short sentences.
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
        model="gemini-3.5-flash-lite",
        config=config
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_processed_id" not in st.session_state:
    st.session_state.last_processed_id = None

if "latest_audio_to_play" not in st.session_state:
    st.session_state.latest_audio_to_play = None

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
    audio_data = mic_recorder(
        start_prompt="Speak 🎙️",
        stop_prompt="Click to Stop & Send 🛑",
        key="recorder",
        just_once=False
    )
except ImportError:
    st.warning("`streamlit-mic-recorder` not found. Install it to enable microphone input.")
    audio_data = None

# --- 5. PROCESS INPUT & RESPOND ---
user_audio_bytes = None

if audio_data and "bytes" in audio_data and len(audio_data["bytes"]) > 0:
    audio_id = audio_data.get("id")
    if audio_id != st.session_state.last_processed_id:
        user_audio_bytes = audio_data["bytes"]
        st.session_state.last_processed_id = audio_id

if user_audio_bytes:
    with st.spinner("AI Teacher is listening..."):
        try:
            # 1. Fetch text response from Gemini
            ai_reply = get_response_from_gemini(
                user_text=None, 
                audio_bytes=user_audio_bytes
            )
            
            # 2. Synthesize Speech to bytes buffer
            audio_buffer = text_to_speech_bytes(ai_reply)
            
            # 3. Store the latest generated audio specifically for autoplaying
            st.session_state.latest_audio_to_play = audio_buffer
            
            # 4. Append to chat history (without autoplay flags)
            st.session_state.chat_history.append({
                "role": "Child",
                "text": "🎤 [Spoken Audio]",
                "audio": None
            })
            st.session_state.chat_history.append({
                "role": "AI Teacher",
                "text": ai_reply,
                "audio": audio_buffer
            })
            
        except Exception as e:
            st.error(f"Error connecting to Gemini API: {str(e)}")

# --- 6. AUTOPLAY LATEST RESPONSE ---
# Only plays the immediate latest response once, then clears it
if st.session_state.latest_audio_to_play is not None:
    st.audio(st.session_state.latest_audio_to_play, format="audio/mp3", autoplay=True)
    st.session_state.latest_audio_to_play = None  # Reset so it won't re-trigger on future reruns

# --- 7. CHAT DISPLAY ---
for msg in st.session_state.chat_history:
    if msg["role"] == "Child":
        st.chat_message("user").write(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["text"])
            if msg["audio"] is not None:
                # Manual playback controls for old chat messages (autoplay disabled)
                st.audio(msg["audio"], format="audio/mp3", autoplay=False)
