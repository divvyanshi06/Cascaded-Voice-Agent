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

st.markdown(
    """
    <style>
        .hero-card {
            background: rgba(59, 130, 246, 0.10);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 16px;
            padding: 2rem;
        }
        .main-title {
            background: linear-gradient(135deg, #3B82F6 0%, #93C5FD 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .pipeline-caption {
            margin: 0 0 0.9rem 0;
            font-size: 0.9rem;
            color: #dfe7ff;
            letter-spacing: 0.02em;
            opacity: 0.9;
        }
        .pipeline-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.4rem;
        }
        .pipeline-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            border: 1px solid rgba(59, 130, 246, 0.28);
            background: rgba(59, 130, 246, 0.10);
            color: #93C5FD;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.38rem 0.7rem;
        }
        .pipeline-badge.arrow {
            background: rgba(59, 130, 246, 0.10);
            border-color: rgba(59, 130, 246, 0.28);
            color: #93C5FD;
        }
        .stage-card {
            background: rgba(59, 130, 246, 0.10);
            border: 1px solid rgba(59, 130, 246, 0.28);
            border-radius: 12px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 0.6rem;
        }
        .stage-card .label {
            display: block;
            margin-bottom: 0.35rem;
            color: #8ec5ff;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .stage-card .content {
            color: #ecf3ff;
            font-size: 0.96rem;
            line-height: 1.5;
        }
        .stage-card .content em {
            color: #ffd166;
            font-style: italic;
        }
        .stage-card .ms {
            display: inline-block;
            margin-top: 0.45rem;
            color: #7dd3fc;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }
        .metric-card {
            flex: 1;
            background: rgba(59, 130, 246, 0.10);
            border: 1px solid rgba(59, 130, 246, 0.28);
            border-radius: 12px;
            padding: 0.8rem;
            text-align: center !important;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .metric-card .label {
            display: block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #9cc8ff;
            margin-bottom: 0.4rem;
        }
        .metric-card .value {
            display: block;
            font-size: 1.35rem;
            font-weight: 800;
            color: #f8fbff;
        }
        .stExpander {
            background: rgba(59, 130, 246, 0.10) !important;
            border: 1px solid rgba(59, 130, 246, 0.28) !important;
            border-radius: 12px !important;
        }
        section[data-testid="stSidebar"] {
            background: #0F0F12;
            border-right: 1px solid rgba(59, 130, 246, 0.2);
        }
        .speak-card {
            background: rgba(24, 24, 27, 0.96);
            border: 1px solid #27272A;
            border-radius: 16px;
            padding: 0.9rem 1rem 0.7rem 1rem;
            margin: 0 0 0.75rem 0;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.12);
        }
        .speak-header {
            font-size: 1.25rem;
            font-weight: 800;
            color: #f8fbff;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        .speak-instruction {
            color: #bfcfe8;
            font-size: 0.82rem;
            opacity: 0.9;
        }
        .audio-wrap {
            background: rgba(24, 24, 27, 0.7);
            border: 1px solid #27272A;
            border-radius: 12px;
            padding: 0.7rem;
            margin-top: 0.5rem;
        }
        @media (max-width: 700px) {
            .metric-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '''
    <div class="hero-card">
        <div class="main-title">🎙️ Mini Cascaded Voice Agent</div>
        <div class="pipeline-caption">Voice pipeline for local speech capture and response</div>
        <div class="pipeline-badges">
            <span class="pipeline-badge">VAD</span>
            <span class="pipeline-badge arrow">→</span>
            <span class="pipeline-badge">ASR</span>
            <span class="pipeline-badge arrow">→</span>
            <span class="pipeline-badge">LLM</span>
            <span class="pipeline-badge arrow">→</span>
            <span class="pipeline-badge">TTS</span>
        </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# --- Sidebar: config -------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
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
st.markdown(
    '''
    <div class="speak-card">
        <div class="speak-header">Speak</div>
        <div class="speak-instruction">Click the mic, speak, then stop — your reply plays automatically</div>
    </div>
    ''',
    unsafe_allow_html=True,
)
st.subheader("Speak")
st.markdown(
    '''
    <div class="audio-wrap">
    ''',
    unsafe_allow_html=True,
)
audio_value = st.audio_input("Record your message")
st.markdown(
    '''
    </div>
    ''',
    unsafe_allow_html=True,
)

if audio_value is not None:
    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar first.")
    else:
        audio_bytes = audio_value.getvalue()

        with st.status("Running pipeline...", expanded=True) as status:
            # Stage 1: VAD trim
            trimmed_audio, vad_ms = trim_silence_vad(audio_bytes)
            st.markdown(
                f'''
                <div class="stage-card">
                    <span class="label">Stage 1 — VAD</span>
                    <div class="content">trimming silence...</div>
                    <span class="ms">{vad_ms:.0f} ms</span>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            # Stage 2: ASR
            asr_model = get_asr_model(asr_size)
            transcript, asr_ms = transcribe(asr_model, trimmed_audio)
            st.markdown(
                f'''
                <div class="stage-card">
                    <span class="label">Stage 2 — ASR</span>
                    <div class="content">→ <em>"{transcript}"</em></div>
                    <span class="ms">{asr_ms:.0f} ms</span>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            if not transcript:
                status.update(label="No speech detected", state="error")
                st.stop()

            # Stage 3: LLM
            reply, llm_ms, new_history = get_llm_response(
                api_key, transcript, st.session_state.chat_history
            )
            st.session_state.chat_history = new_history
            st.markdown(
                f'''
                <div class="stage-card">
                    <span class="label">Stage 3 — LLM</span>
                    <div class="content">→ <em>"{reply}"</em></div>
                    <span class="ms">{llm_ms:.0f} ms</span>
                </div>
                ''',
                unsafe_allow_html=True,
            )

            # Stage 4: TTS
            reply_audio, tts_ms = synthesize_speech(reply, voice)
            st.markdown(
                f'''
                <div class="stage-card">
                    <span class="label">Stage 4 — TTS</span>
                    <div class="content">synthesizing speech (edge-tts)...</div>
                    <span class="ms">{tts_ms:.0f} ms</span>
                </div>
                ''',
                unsafe_allow_html=True,
            )

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
            st.markdown(f"<div><strong>You:</strong> {turn['transcript']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div><strong>Agent:</strong> {turn['reply']}</div>", unsafe_allow_html=True)
            lat = turn["latency"]
            st.markdown(
                f"<div><span style='color:#8ec5ff;'>VAD {lat['vad']:.0f}ms</span> · "
                f"<span style='color:#8ec5ff;'>ASR {lat['asr']:.0f}ms</span> · "
                f"<span style='color:#8ec5ff;'>LLM {lat['llm']:.0f}ms</span> · "
                f"<span style='color:#8ec5ff;'>TTS {lat['tts']:.0f}ms</span> · "
                f"<strong>total {sum(lat.values()):.0f}ms</strong></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    latest = st.session_state.turns[-1]["latency"]
    st.subheader("Latest turn — stage latency breakdown")
    st.markdown(
        """
        <div class="metric-row">
            <div class="metric-card">
                <span class="label">VAD</span>
                <span class="value">{vad_ms:.0f} ms</span>
            </div>
            <div class="metric-card">
                <span class="label">ASR</span>
                <span class="value">{asr_ms:.0f} ms</span>
            </div>
            <div class="metric-card">
                <span class="label">LLM</span>
                <span class="value">{llm_ms:.0f} ms</span>
            </div>
            <div class="metric-card">
                <span class="label">TTS</span>
                <span class="value">{tts_ms:.0f} ms</span>
            </div>
        </div>
        """.format(
            vad_ms=latest["vad"],
            asr_ms=latest["asr"],
            llm_ms=latest["llm"],
            tts_ms=latest["tts"],
        ),
        unsafe_allow_html=True,
    )