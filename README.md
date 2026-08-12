# Mini Cascaded Voice Agent

🔗 **Live demo:** https://cascaded-voice-agent.streamlit.app
A working demo of the cascaded voice AI architecture:

```
Mic → VAD (webrtcvad) → ASR (faster-whisper, local) → LLM (Gemini API)
    → TTS (edge-tts, local) → Speaker
```

Everything runs locally and free, except the LLM call which uses the
Gemini API's free tier.

## Setup

1. **Get a free Gemini API key**
   Go to https://aistudio.google.com/app/apikey → create key → copy it.
   You can either paste it into the app's sidebar each time you run it,
   or save it once so it's picked up automatically:
   create `.streamlit/secrets.toml` in the project root with:
```toml
   GEMINI_API_KEY = "your-key-here"
```
   (this file is git-ignored, so it stays local and never gets committed)

2. **Create a virtual environment and install dependencies**
   ```bash
   cd Cascaded-Voice-Agent
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Mac/Linux
   pip install -r requirements.txt
   ```

   Notes:
   - `faster-whisper` will download the Whisper model weights on first run
     (a few hundred MB depending on model size — pick "tiny" if your laptop
     is slow).
   - `webrtcvad` sometimes needs Microsoft C++ Build Tools on Windows to
     install. If it fails, install "Build Tools for Visual Studio" first.

3. **Run it**
   ```bash
   streamlit run app.py
   ```
   This opens a browser tab at `localhost:8501`.

4. **Use it**
   - If you haven't set up `secrets.toml`, paste your Gemini API key in the sidebar.
   - Click the record widget, speak, stop recording.
   - Watch each pipeline stage run live with its latency.
   - The agent's spoken reply plays automatically.

## What to actually look at (for your report)

- **Per-stage latency** is shown after every turn — this is the real
  metric that matters in production voice agents. Compare which stage
  is your bottleneck (usually ASR or LLM on CPU-only hardware).
- **VAD trimming**: the raw recorded clip usually has silence at the
  start/end — check the VAD stage output vs. the original clip length
  to see it working.
- **What's missing vs. a real production system**: this demo processes
  one full utterance at a time (record → stop → process). Real voice
  agents stream continuously and handle barge-in/interruption mid-turn.
  That's the gap between this demo and a real telephony voice agent —
  worth a paragraph in your report explaining why that's architecturally
  harder (see the turn-taking discussion).

## Known limitations (be upfront about these, don't hide them)

- No real-time streaming — audio is recorded fully, then processed.
  A production system pipelines all 4 stages concurrently.
- No interruption/barge-in handling — this is a request/response loop,
  not full-duplex.
- CPU-only Whisper on a laptop will have noticeably higher ASR latency
  than a production GPU-backed system. That's expected and worth noting,
  not a bug.
  <img width="1914" height="912" alt="image" src="https://github.com/user-attachments/assets/cc15a696-135c-4c98-be67-b1eeee173cfc" />

