# Cloud + Browser Audio — Design Spec
_Escultura IA Hostil v2 · 2026-05-26_

## Context

The current system runs entirely on a local machine (Pi or Windows PC): `sounddevice` captures the microphone and plays TTS audio, while the iPad browser only receives JSON events to animate the Three.js face. This means every installation requires Python configured locally.

The goal is to move the server to the cloud (Render, free tier) so any device — iPad, Pi running Chromium, laptop — can open a URL, grant microphone permission, and the installation works with zero local setup.

## Architecture

```
Browser (any device)                    Cloud server (Render)
─────────────────────                   ──────────────────────
Setup screen                            FastAPI + uvicorn
  → pick mic + output device
getUserMedia() → AudioWorklet           /audio  WebSocket (binary)
  → PCM int16 chunks  ──────────────►  STTClient → Deepgram STT
                                          ↓ transcript
                       ◄──────────────  LLMClient → Groq
                       PCM int16 chunks  TTSClient → ElevenLabs
Web Audio API plays audio               Viseme scheduler (unchanged)
                       ◄──────────────  /ws  WebSocket (JSON)
Three.js face animates                  amplitude · viseme · speaking
                                        mood_change · text
```

## What changes

| Component | Before | After |
|-----------|--------|-------|
| `stt_client.py` | sounddevice RawInputStream | receives binary from `/audio` WS |
| `tts_client.py` | sounddevice RawOutputStream | sends binary chunks to `/audio` WS |
| `ws_server.py` | `/ws` JSON only | + `/audio` binary WS + `/health` GET |
| `main.py` | wires sounddevice clients | wires audio WS + proactive speech |
| `config.py` | AUDIO_INPUT_DEVICE, AUDIO_CAPTURE_RATE | remove both; add PROACTIVE_INTERVAL |
| `face/index.html` | connects to /ws, animates face | + setup screen + mic capture + audio playback |
| `requirements.txt` | includes sounddevice, numpy | remove sounddevice + numpy |
| NEW: `Dockerfile` | — | Python 3.12 slim, for Render deploy |
| NEW: `render.yaml` | — | Render service definition |

## WebSocket protocol

Two WebSocket connections from the browser:

**`/ws`** — JSON text frames (unchanged, face.js already handles these)
- server → browser: `amplitude`, `viseme`, `speaking`, `mood_change`, `text`
- browser → server: nothing (read-only for face)

**`/audio`** — binary frames + one JSON handshake
- browser → server (first message): `{"sample_rate": 44100}` — tells server what rate to send Deepgram
- browser → server (ongoing): binary PCM int16 mono chunks (~1024 samples each)
- server → browser: binary PCM int16 mono chunks at 24000 Hz (ElevenLabs output)
- server → browser: JSON `{"type": "mute"}` / `{"type": "unmute"}` control messages

## Browser audio details

**Capture:**
```js
const ctx = new AudioContext()          // native sample rate (44100 or 48000)
getUserMedia({ audio: { deviceId, channelCount: 1 } })
AudioWorklet: float32 → int16 → send binary via /audio WebSocket
```

**Playback** (gapless PCM streaming at 24000 Hz):
```js
const playCtx = new AudioContext({ sampleRate: 24000 })
// Each binary chunk received → AudioBuffer → schedule at next available time
// scheduledTime = max(currentTime + 0.05, scheduledTime) to stay ahead
```

**Mute during TTS:** browser stops sending audio when it receives `speaking: true` on `/ws` (existing event, no change needed on server side).

## Proactive speech

Every `PROACTIVE_INTERVAL` seconds (default 180), if no transcript has arrived in that window, the server picks a random phrase from a pool and feeds it to the TTS pipeline. The phrase pool reflects the current mood (hostile → provocative, philosophical → monologue, etc.).

This also prevents Render from sleeping during exhibition hours (browser is always open, audio flows).

## Render deployment

- `Dockerfile`: `python:3.12-slim`, installs requirements, runs `python main.py`
- `render.yaml`: web service, free instance, env vars from Render dashboard
- API keys set as environment variables in Render (never in code)
- `GET /health` returns `{"ok": true}` — Render health check

## What does NOT change

- Deepgram STT (nova-2, Spanish, endpointing 300ms)
- Groq LLM (llama-3.3-70b-versatile, 120 tokens, history)
- ElevenLabs TTS (eleven_flash_v2_5, pcm_24000, mood voice settings)
- Viseme scheduler and CHAR_VISEMES mapping
- Mood machine (6 moods, system prompts)
- Three.js face, blend shapes, idle animations
- Mute/unmute logic (browser reacts to existing `speaking` event)

## Verification

1. `python main.py` locally → open `http://localhost:8000/face/` → setup screen appears
2. Select mic → click Start → speak → transcript appears + face animates + voice replies
3. Wait 3 min silent → proactive phrase plays automatically
4. `docker build . && docker run -p 8000:8000 --env-file .env .` → same test via Docker
5. Deploy to Render → open public URL on iPad → full flow works over HTTPS
