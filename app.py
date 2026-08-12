"""
Mini Cascaded Voice Agent -- Streamlit UI

Pipeline: Mic input -> VAD trim -> ASR (faster-whisper) -> LLM (Gemini)
          -> TTS (edge-tts) -> Audio playback

Run with:
    streamlit run app.py
"""

import streamlit as st
from pipeline import (
    trim_silence_vad,
    load_asr_model,
    transcribe,
    get_llm_response,
    synthesize_speech,
)

st.set_page_config(page_title="Mini Voice Agent", page_icon="🎙️", layout="centered")

st.title("🎙️ Mini Cascaded Voice Agent")
st.caption("VAD → ASR (local Whisper) → LLM (Gemini) → TTS (edge-tts)")

# --- Sidebar: config -------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password",
                                 help="Get a free key at https://aistudio.google.com/app/apikey")
    asr_size = st.selectbox("Whisper model size", ["tiny", "base", "small"], index=1)
    voice = st.selectbox(
        "TTS voice",
        ["en-US-AriaNeural", "en-US-GuyNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
        index=2,
    )
    st.divider()
    st.caption("Everything except the LLM call runs 100% locally, "
               "free, no API cost.")

# --- Session state -----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "turns" not in st.session_state:
    st.session_state.turns = []

# --- Load ASR model (cached so it only loads once) --------------------
@st.cache_resource
def get_asr_model(size):
    return load_asr_model(size)

# --- Main interaction ---------------------------------------------------
st.subheader("Speak")
audio_value = st.audio_input("Record your message")

if audio_value is not None:
    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar first.")
    else:
        audio_bytes = audio_value.getvalue()

        with st.status("Running pipeline...", expanded=True) as status:
            # Stage 1: VAD trim
            st.write("**Stage 1 — VAD:** trimming silence...")
            trimmed_audio, vad_ms = trim_silence_vad(audio_bytes)
            st.write(f"done in `{vad_ms:.0f} ms`")

            # Stage 2: ASR
            st.write("**Stage 2 — ASR:** transcribing (local Whisper)...")
            asr_model = get_asr_model(asr_size)
            transcript, asr_ms = transcribe(asr_model, trimmed_audio)
            st.write(f"→ *\"{transcript}\"*  —  `{asr_ms:.0f} ms`")

            if not transcript:
                status.update(label="No speech detected", state="error")
                st.stop()

            # Stage 3: LLM
            st.write("**Stage 3 — LLM:** generating response (Gemini)...")
            reply, llm_ms, new_history = get_llm_response(
                api_key, transcript, st.session_state.chat_history
            )
            st.session_state.chat_history = new_history
            st.write(f"→ *\"{reply}\"*  —  `{llm_ms:.0f} ms`")

            # Stage 4: TTS
            st.write("**Stage 4 — TTS:** synthesizing speech (edge-tts)...")
            reply_audio, tts_ms = synthesize_speech(reply, voice)
            st.write(f"done in `{tts_ms:.0f} ms`")

            total_ms = vad_ms + asr_ms + llm_ms + tts_ms
            status.update(label=f"Done — total pipeline latency: {total_ms:.0f} ms",
                          state="complete")

        st.subheader("Response")
        st.audio(reply_audio, format="audio/mp3", autoplay=True)

        st.session_state.turns.append({
            "transcript": transcript,
            "reply": reply,
            "latency": {"vad": vad_ms, "asr": asr_ms, "llm": llm_ms, "tts": tts_ms},
        })

# --- Conversation log + latency chart -----------------------------------
if st.session_state.turns:
    st.divider()
    st.subheader("Conversation log")
    for i, turn in enumerate(reversed(st.session_state.turns), 1):
        with st.expander(f"Turn {len(st.session_state.turns) - i + 1}: "
                          f"{turn['transcript'][:50]}"):
            st.markdown(f"**You:** {turn['transcript']}")
            st.markdown(f"**Agent:** {turn['reply']}")
            lat = turn["latency"]
            st.markdown(
                f"`VAD {lat['vad']:.0f}ms` · `ASR {lat['asr']:.0f}ms` · "
                f"`LLM {lat['llm']:.0f}ms` · `TTS {lat['tts']:.0f}ms` · "
                f"**total {sum(lat.values()):.0f}ms**"
            )

    st.divider()
    latest = st.session_state.turns[-1]["latency"]
    st.subheader("Latest turn — stage latency breakdown")
    st.bar_chart(latest)