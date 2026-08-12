"""
Mini Cascaded Voice Agent — pipeline.py

This file implements the 4 stages of a cascaded voice agent:
  1. VAD    -> trim silence from the recorded audio
  2. ASR    -> speech to text (local, faster-whisper)
  3. LLM    -> text to text (Gemini API, free tier)
  4. TTS    -> text to speech (edge-tts, local & free)

Each function times itself and returns (result, latency_ms) so you can
show the latency budget in the UI -- this is the metric that matters
most in production voice agents.
"""

import io
import time
import wave
import asyncio
import struct

import numpy as np
import webrtcvad
import soundfile as sf
import edge_tts
from faster_whisper import WhisperModel
import google.generativeai as genai


# ---------------------------------------------------------------------------
# Stage 1: VAD (Voice Activity Detection)
# ---------------------------------------------------------------------------
# webrtcvad works on 16-bit mono PCM audio, in 10/20/30ms frames, at
# 8000/16000/32000/48000 Hz. It classifies each frame as speech or silence.
# We use this to trim leading/trailing silence before running ASR --
# in a real streaming system, VAD is what tells the orchestrator
# "the user has started/stopped talking," which is the trigger for
# starting ASR and later for turn-taking / interruption handling.

def trim_silence_vad(audio_bytes: bytes, sample_rate: int = 16000,
                      frame_ms: int = 30, aggressiveness: int = 2):
    """
    Trims leading and trailing silence from a WAV byte stream using VAD.
    Falls back to returning the original audio untouched if anything
    about the input format doesn't match what webrtcvad expects.
    """
    t0 = time.time()
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        if audio.ndim > 1:
            audio = audio[:, 0]  # force mono

        if sr != sample_rate:
            # webrtcvad requires 8k/16k/32k/48k -- resample naively if needed
            duration = len(audio) / sr
            target_len = int(duration * sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio), target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.int16)
            sr = sample_rate

        vad = webrtcvad.Vad(aggressiveness)
        frame_len = int(sr * frame_ms / 1000)
        frames = [
            audio[i:i + frame_len]
            for i in range(0, len(audio) - frame_len, frame_len)
        ]

        speech_flags = []
        for f in frames:
            pcm = struct.pack("<%dh" % len(f), *f)
            speech_flags.append(vad.is_speech(pcm, sr))

        if not any(speech_flags):
            # nothing detected as speech -- return original, don't destroy data
            return audio_bytes, (time.time() - t0) * 1000

        first = speech_flags.index(True)
        last = len(speech_flags) - 1 - speech_flags[::-1].index(True)

        start_sample = first * frame_len
        end_sample = min((last + 2) * frame_len, len(audio))  # pad 1 frame
        trimmed = audio[start_sample:end_sample]

        buf = io.BytesIO()
        sf.write(buf, trimmed, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), (time.time() - t0) * 1000

    except Exception:
        # VAD is a latency/quality optimization, not a hard requirement --
        # if it fails for any reason, fall back to the raw audio.
        return audio_bytes, (time.time() - t0) * 1000


# ---------------------------------------------------------------------------
# Stage 2: ASR (Speech to Text) -- local, via faster-whisper
# ---------------------------------------------------------------------------
# Model is loaded once and cached by the caller (see app.py @st.cache_resource)
# "base" model is a good CPU-speed/accuracy tradeoff for a demo.

def load_asr_model(model_size: str = "base"):
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(model: WhisperModel, audio_bytes: bytes):
    t0 = time.time()
    buf = io.BytesIO(audio_bytes)
    segments, _info = model.transcribe(buf, beam_size=1, language="en")
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip(), (time.time() - t0) * 1000


# ---------------------------------------------------------------------------
# Stage 3: LLM -- Gemini API (free tier)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. Keep answers short "
    "(2-3 sentences max) since they will be spoken aloud, not read."
)


def get_llm_response(api_key: str, user_text: str, history=None):
    t0 = time.time()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )
    chat = model.start_chat(history=history or [])
    response = chat.send_message(user_text)
    return response.text.strip(), (time.time() - t0) * 1000, chat.history


# ---------------------------------------------------------------------------
# Stage 4: TTS (Text to Speech) -- edge-tts, local & free
# ---------------------------------------------------------------------------

async def _synthesize_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


def synthesize_speech(text: str, voice: str = "en-US-AriaNeural"):
    t0 = time.time()
    audio_bytes = asyncio.run(_synthesize_async(text, voice))
    return audio_bytes, (time.time() - t0) * 1000