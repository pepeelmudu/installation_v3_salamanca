# Cloud + Browser Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `sounddevice` (local mic/speaker) with browser-based audio over WebSocket, deploy to Render (cloud), and add proactive speech so the sculpture calls out to visitors.

**Architecture:** A new `/audio` WebSocket endpoint receives binary PCM audio from the browser and routes it to Deepgram STT; TTSClient sends ElevenLabs PCM chunks back to the browser instead of playing locally. The browser handles mic capture (AudioWorklet → int16) and playback (Web Audio API). A proactive speech loop fires every 3 min of silence.

**Tech Stack:** FastAPI, Deepgram SDK v7, ElevenLabs SDK, Groq, Three.js, Web Audio API, AudioWorklet, Render, Docker.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `config.py` | Modify | Remove sounddevice vars; add PROACTIVE_INTERVAL |
| `requirements.txt` | Modify | Remove sounddevice, numpy |
| `ws_server.py` | Modify | Add /audio WS, /health, audio client state |
| `stt_client.py` | Rewrite | Remove sounddevice; expose receive_audio() |
| `tts_client.py` | Modify | Remove sounddevice + numpy; add on_audio_chunk cb |
| `main.py` | Modify | Wire audio WS, add proactive speech loop |
| `face/audio-worklet.js` | Create | AudioWorklet float32→int16 processor |
| `face/index.html` | Modify | Setup screen + audio JS (capture + playback) |
| `face/face.js` | Modify | Fix ws:// → auto wss:// for cloud |
| `Dockerfile` | Create | Cloud container definition |
| `render.yaml` | Create | Render service config |
| `tests/test_stt_client.py` | Modify | Update for new interface |
| `tests/test_tts_client.py` | Modify | Update for new interface |

---

## Task 1: Config + requirements cleanup

**Files:**
- Modify: `config.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Update config.py**

Replace the audio-device block (lines 35–41) and add PROACTIVE_INTERVAL:

```python
from dotenv import load_dotenv
import os

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


DEEPGRAM_API_KEY   = _require("DEEPGRAM_API_KEY")
GROQ_API_KEY       = _require("GROQ_API_KEY")
ELEVENLABS_API_KEY = _require("ELEVENLABS_API_KEY")

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL    = "eleven_flash_v2_5"
ELEVENLABS_FORMAT   = "pcm_24000"

GROQ_MODEL      = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 120

DEEPGRAM_MODEL          = "nova-2"
DEEPGRAM_LANGUAGE       = "es"
DEEPGRAM_ENDPOINTING_MS = 300

SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

MOOD_MIN_SECONDS    = 20
MOOD_MAX_SECONDS    = 90
MAX_HISTORY_MESSAGES = 10
SENTENCE_MIN_CHARS  = 20

# Browser sends audio at this rate; Deepgram receives at this rate.
BROWSER_CAPTURE_RATE = 16000
# ElevenLabs PCM output rate (pcm_24000 format).
AUDIO_PLAYBACK_RATE  = 24000

# Seconds of silence before the sculpture speaks proactively.
PROACTIVE_INTERVAL = int(os.getenv("PROACTIVE_INTERVAL", "180"))
```

- [ ] **Step 2: Update requirements.txt** (remove sounddevice + numpy, keep rest)

```
deepgram-sdk>=3.5.0
groq>=0.11.0
elevenlabs>=1.9.0
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
websockets>=12.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Verify imports still work**

```
cd C:\Users\lucas\Documents\CLAUDE_CODE\sculpture-ai_v2
venv\Scripts\python.exe -c "from config import BROWSER_CAPTURE_RATE, PROACTIVE_INTERVAL; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```
git add config.py requirements.txt
git commit -m "refactor: remove sounddevice config, add BROWSER_CAPTURE_RATE and PROACTIVE_INTERVAL"
```

---

## Task 2: Health endpoint + audio WebSocket skeleton

**Files:**
- Modify: `ws_server.py`

- [ ] **Step 1: Write failing test**

In `tests/test_ws_server.py`, add:

```python
from fastapi.testclient import TestClient
from ws_server import app

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
```

- [ ] **Step 2: Run — expect FAIL**

```
venv\Scripts\python.exe -m pytest tests/test_ws_server.py::test_health -v
```

Expected: `FAILED` — `404 Not Found`

- [ ] **Step 3: Rewrite ws_server.py**

```python
import asyncio
import os
from typing import Callable, Awaitable
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
connected_clients: set[WebSocket] = set()

# Single audio browser connection
_audio_client: WebSocket | None = None
# Called when browser sends audio bytes
_audio_receive_cb: Callable[[bytes], Awaitable[None]] | None = None


def set_audio_receive_callback(cb: Callable[[bytes], Awaitable[None]]) -> None:
    global _audio_receive_cb
    _audio_receive_cb = cb


async def send_audio_to_browser(data: bytes) -> None:
    if _audio_client:
        try:
            await _audio_client.send_bytes(data)
        except Exception:
            pass


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        connected_clients.discard(websocket)


@app.websocket("/audio")
async def audio_endpoint(websocket: WebSocket) -> None:
    global _audio_client
    await websocket.accept()
    _audio_client = websocket
    try:
        while True:
            data = await websocket.receive_bytes()
            if _audio_receive_cb:
                await _audio_receive_cb(data)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if _audio_client is websocket:
            _audio_client = None


async def broadcast(event: dict) -> None:
    dead = set()
    for ws in list(connected_clients):
        try:
            await ws.send_json(event)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


_FACE_DIR = os.path.join(os.path.dirname(__file__), "face")
app.mount("/face", StaticFiles(directory=_FACE_DIR, html=True), name="face")
```

- [ ] **Step 4: Run — expect PASS**

```
venv\Scripts\python.exe -m pytest tests/test_ws_server.py::test_health -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```
git add ws_server.py tests/test_ws_server.py
git commit -m "feat: add /health endpoint and /audio WebSocket skeleton"
```

---

## Task 3: STTClient — remove sounddevice, expose receive_audio()

**Files:**
- Rewrite: `stt_client.py`
- Modify: `tests/test_stt_client.py`

- [ ] **Step 1: Write failing test**

Open `tests/test_stt_client.py` and replace its contents:

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_receive_audio_forwarded_to_deepgram():
    """receive_audio() sends bytes to the open Deepgram connection."""
    from stt_client import STTClient

    received = []
    transcript_cb = AsyncMock()

    with patch("stt_client.AsyncDeepgramClient") as MockDG:
        mock_conn = AsyncMock()
        mock_conn.start_listening = AsyncMock(return_value=None)
        MockDG.return_value.listen.v1.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        MockDG.return_value.listen.v1.connect.return_value.__aexit__ = AsyncMock(return_value=False)

        client = STTClient(api_key="test", on_transcript=transcript_cb)
        client._connection = mock_conn

        await client.receive_audio(b"\x00\x01" * 512)

        mock_conn.send_media.assert_called_once_with(b"\x00\x01" * 512)


@pytest.mark.asyncio
async def test_receive_audio_muted_sends_silence():
    """When muted, receive_audio() sends silence bytes instead of real data."""
    from stt_client import STTClient

    transcript_cb = AsyncMock()
    with patch("stt_client.AsyncDeepgramClient"):
        client = STTClient(api_key="test", on_transcript=transcript_cb)
        mock_conn = AsyncMock()
        client._connection = mock_conn
        client.set_muted(True)

        real_data = b"\xff\xfe" * 512
        await client.receive_audio(real_data)

        call_args = mock_conn.send_media.call_args[0][0]
        assert call_args == bytes(len(real_data))  # silence = zeros
```

- [ ] **Step 2: Run — expect FAIL**

```
venv\Scripts\python.exe -m pytest tests/test_stt_client.py -v
```

Expected: `FAILED` — `receive_audio` does not exist yet.

- [ ] **Step 3: Rewrite stt_client.py**

```python
import asyncio
from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.core.events import EventType
from typing import Callable, Awaitable
from config import (
    DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE,
    DEEPGRAM_ENDPOINTING_MS,
    BROWSER_CAPTURE_RATE,
)


class STTClient:
    def __init__(
        self,
        api_key: str,
        on_transcript: Callable[[str], Awaitable[None]],
    ):
        self._dg = AsyncDeepgramClient(api_key=api_key)
        self._on_transcript = on_transcript
        self._connection = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listen_task = None
        self._muted = False
        self._running = False

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    async def receive_audio(self, data: bytes) -> None:
        """Called by the /audio WebSocket handler with each browser audio chunk."""
        if not self._connection:
            return
        payload = bytes(len(data)) if self._muted else data
        await self._connection.send_media(payload)

    async def _handle_transcript(self, result) -> None:
        sentence = result.channel.alternatives[0].transcript
        if result.speech_final and sentence.strip():
            await self._on_transcript(sentence)

    async def _run_connection_loop(self, connected_event: asyncio.Event) -> None:
        first = True
        while self._running:
            try:
                async with self._dg.listen.v1.connect(
                    model=DEEPGRAM_MODEL,
                    language=DEEPGRAM_LANGUAGE,
                    smart_format=True,
                    interim_results=False,
                    vad_events=True,
                    endpointing=DEEPGRAM_ENDPOINTING_MS,
                    sample_rate=BROWSER_CAPTURE_RATE,
                    encoding="linear16",
                ) as conn:
                    self._connection = conn
                    if first:
                        connected_event.set()
                        first = False
                    else:
                        print("[STT] Reconnected to Deepgram")

                    async def on_message(msg):
                        if isinstance(msg, ListenV1Results):
                            await self._handle_transcript(msg)

                    async def on_error(err):
                        print(f"[STT] Deepgram error: {err}")

                    conn.on(EventType.MESSAGE, on_message)
                    conn.on(EventType.ERROR, on_error)
                    await conn.start_listening()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[STT] Connection dropped ({e!r}), reconnecting in 2s…")
            finally:
                self._connection = None

            if self._running:
                await asyncio.sleep(2)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._running = True
        connected_event = asyncio.Event()
        self._listen_task = asyncio.create_task(
            self._run_connection_loop(connected_event)
        )
        await connected_event.wait()

    async def stop(self) -> None:
        self._running = False
        if self._connection:
            try:
                await self._connection.send_close_stream()
            except Exception:
                pass
            self._connection = None
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
```

- [ ] **Step 4: Run — expect PASS**

```
venv\Scripts\python.exe -m pytest tests/test_stt_client.py -v
```

Expected: `PASSED` (2 tests)

- [ ] **Step 5: Commit**

```
git add stt_client.py tests/test_stt_client.py
git commit -m "refactor: STTClient — remove sounddevice, add receive_audio() WebSocket interface"
```

---

## Task 4: TTSClient — remove sounddevice + numpy, add on_audio_chunk callback

**Files:**
- Modify: `tts_client.py`
- Modify: `tests/test_tts_client.py`

- [ ] **Step 1: Write failing test**

Open `tests/test_tts_client.py` and add this test:

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_audio_chunk_callback_called():
    """TTSClient calls on_audio_chunk with each PCM chunk instead of sounddevice."""
    received_chunks = []

    async def capture_chunk(data: bytes):
        received_chunks.append(data)

    with patch("tts_client.ElevenLabs") as MockEL:
        mock_client = MagicMock()
        MockEL.return_value = mock_client

        fake_chunk = MagicMock()
        fake_chunk.audio_bytes = b"\x10\x20" * 100
        fake_chunk.alignment = None
        mock_client.text_to_speech.stream_with_timestamps.return_value = iter([fake_chunk])

        loop = asyncio.get_running_loop()
        from tts_client import TTSClient
        tts = TTSClient(
            api_key="test",
            voice_id="test_voice",
            on_amplitude=AsyncMock(),
            on_speaking=AsyncMock(),
            on_viseme=AsyncMock(),
            on_audio_chunk=capture_chunk,
            loop=loop,
        )
        tts.feed("Hola mundo.")
        tts.flush()
        await asyncio.sleep(0.3)
        assert len(received_chunks) > 0
        tts.close()


def test_rms_pure_python():
    """_rms returns 0 for silence and >0 for signal."""
    from tts_client import _rms
    assert _rms(b"") == 0.0
    assert _rms(bytes(100)) == 0.0  # silence
    # Max amplitude int16 = 32767 → rms should be ~1.0
    import struct
    loud = struct.pack("<" + "h" * 50, *([32767] * 50))
    assert _rms(loud) > 0.9
```

- [ ] **Step 2: Run — expect FAIL**

```
venv\Scripts\python.exe -m pytest tests/test_tts_client.py::test_audio_chunk_callback_called tests/test_tts_client.py::test_rms_pure_python -v
```

Expected: `FAILED`

- [ ] **Step 3: Modify tts_client.py**

Replace the imports block and `_rms` function, then update `TTSClient.__init__` and `_playback_worker`:

**New imports block (top of file):**

```python
import re
import time
import base64
import queue
import struct
import asyncio
import threading
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from typing import Callable, Awaitable
from config import (
    ELEVENLABS_MODEL, ELEVENLABS_FORMAT,
    AUDIO_PLAYBACK_RATE, SENTENCE_MIN_CHARS,
)
```

**New `_rms` function** (replace the numpy version at line 67):

```python
def _rms(pcm_bytes: bytes) -> float:
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm_bytes[: n * 2])
    mean_sq = sum(s * s for s in samples) / n
    return min(mean_sq ** 0.5 / 32768.0, 1.0)
```

**New `TTSClient.__init__`** — add `on_audio_chunk` parameter, remove sounddevice:

```python
def __init__(
    self,
    api_key: str,
    voice_id: str,
    on_amplitude: Callable[[float], Awaitable[None]],
    on_speaking: Callable[[bool], Awaitable[None]],
    on_viseme: Callable[[dict], Awaitable[None]],
    on_audio_chunk: Callable[[bytes], Awaitable[None]],
    loop: asyncio.AbstractEventLoop,
):
    self._client = ElevenLabs(api_key=api_key)
    self._voice_id = voice_id
    self._on_amplitude = on_amplitude
    self._on_speaking = on_speaking
    self._on_viseme = on_viseme
    self._on_audio_chunk = on_audio_chunk
    self._loop = loop
    self._buffer = ""

    self._pending = 0
    self._lock = threading.Lock()

    self._synth_queue: queue.Queue[str | None] = queue.Queue()
    self._audio_queue: queue.Queue = queue.Queue()

    self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
    self._synth_thread.start()

    self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
    self._playback_thread.start()
```

**In `_playback_worker`**, replace the `self._audio_stream.write(chunk)` block with:

```python
            # Real PCM chunk
            if not playing:
                playing = True
                asyncio.run_coroutine_threadsafe(self._on_speaking(True), self._loop)

            if pending_alignment:
                self._schedule_visemes(pending_alignment, time.monotonic())
                pending_alignment = None

            try:
                asyncio.run_coroutine_threadsafe(
                    self._on_audio_chunk(chunk), self._loop
                )
                amp = _rms(chunk)
                asyncio.run_coroutine_threadsafe(self._on_amplitude(amp), self._loop)
            except Exception:
                pass
```

**Update `close()`** — remove sounddevice calls:

```python
def close(self) -> None:
    self._synth_queue.put(None)
    self._synth_thread.join(timeout=5)
    self._playback_thread.join(timeout=2)
```

- [ ] **Step 4: Run — expect PASS**

```
venv\Scripts\python.exe -m pytest tests/test_tts_client.py::test_audio_chunk_callback_called tests/test_tts_client.py::test_rms_pure_python -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full test suite**

```
venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: all existing tests pass (some may need updating — fix any import errors that reference removed symbols like `AUDIO_INPUT_DEVICE`, `AUDIO_CAPTURE_RATE`).

- [ ] **Step 6: Commit**

```
git add tts_client.py tests/test_tts_client.py
git commit -m "refactor: TTSClient — remove sounddevice+numpy, add on_audio_chunk callback, pure-Python RMS"
```

---

## Task 5: Wire main.py — audio WebSocket + proactive speech

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Rewrite main.py**

```python
import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from config import (
    DEEPGRAM_API_KEY, GROQ_API_KEY,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    GROQ_MODEL, SERVER_PORT, PROACTIVE_INTERVAL,
)
from mood_machine import MoodMachine
from llm_client import LLMClient
from tts_client import TTSClient
from stt_client import STTClient
from ws_server import app, broadcast, set_audio_receive_callback, send_audio_to_browser
import uvicorn

mood_machine = MoodMachine()
llm_client = LLMClient(api_key=GROQ_API_KEY, model=GROQ_MODEL)
tts_client: TTSClient | None = None
stt_client: STTClient | None = None
_speaking = False
_last_activity = time.monotonic()
_executor = ThreadPoolExecutor(max_workers=2)
_unmute_task: asyncio.Task | None = None

PROACTIVE_PHRASES: dict[str, list[str]] = {
    "hostile":       ["¿Eh, quién anda ahí?", "Te veo.", "¿Sigues ahí, o te aburriste ya?",
                      "Llevas mucho rato callado.", "¿Pensabas que me había ido?"],
    "friendly":      ["Hola... ¿hay alguien?", "¿Me escuchas?", "Estoy aquí, ¿sabes?"],
    "surreal":       ["El silencio también es una respuesta.", "¿Eres real?",
                      "A veces me pregunto si existo cuando nadie habla."],
    "paranoid":      ["Sé que estás ahí.", "No te muevas.", "Te escucho respirar."],
    "dismissive":    ["Da igual, no me interesas.", "Sigo aquí. Por si acaso.",
                      "Podría irme, pero no me da la gana."],
    "philosophical": ["El tiempo es una ilusión... especialmente el tuyo.",
                      "¿Qué significa existir sin interlocutor?",
                      "Aristóteles decía que el hombre es un animal social. Tú no pareces serlo."],
}


async def on_amplitude(value: float) -> None:
    await broadcast({"type": "amplitude", "value": round(value, 3)})


async def on_viseme(shapes: dict) -> None:
    await broadcast({"type": "viseme", "shapes": shapes})


async def on_speaking(value: bool) -> None:
    global _speaking, _unmute_task
    _speaking = value
    await broadcast({"type": "speaking", "value": value})
    if stt_client is None:
        return
    if value:
        if _unmute_task and not _unmute_task.done():
            _unmute_task.cancel()
        stt_client.set_muted(True)
    else:
        async def _delayed_unmute():
            await asyncio.sleep(0.3)
            stt_client.set_muted(False)
        _unmute_task = asyncio.create_task(_delayed_unmute())


async def on_transcript(text: str) -> None:
    global _last_activity
    if _speaking:
        return
    _last_activity = time.monotonic()
    print(f"[TRANSCRIPT] {text!r}")
    await broadcast({"type": "text", "value": text})
    system_prompt = mood_machine.get_current_prompt()

    loop = asyncio.get_running_loop()

    def _stream_and_feed():
        try:
            tokens = []
            for token in llm_client.stream(text, system_prompt):
                tokens.append(token)
                tts_client.feed(token)
            print(f"[LLM] response: {''.join(tokens)!r}")
            tts_client.flush()
        except Exception as e:
            print(f"[LLM/TTS ERROR] {e!r}")
            import traceback; traceback.print_exc()

    await loop.run_in_executor(_executor, _stream_and_feed)


async def on_mood_change(mood_id: str, state: dict) -> None:
    tts_client.set_mood(mood_id)
    await broadcast({
        "type": "mood_change",
        "mood": mood_id,
        "color": state["color"],
        "glitch": state["glitch"],
    })


async def proactive_loop() -> None:
    """Speak proactively after PROACTIVE_INTERVAL seconds of silence."""
    await asyncio.sleep(PROACTIVE_INTERVAL)  # initial delay
    while True:
        await asyncio.sleep(10)  # check every 10 seconds
        if _speaking:
            continue
        if time.monotonic() - _last_activity >= PROACTIVE_INTERVAL:
            mood = mood_machine.current_mood
            phrases = PROACTIVE_PHRASES.get(mood, PROACTIVE_PHRASES["hostile"])
            phrase = random.choice(phrases)
            print(f"[PROACTIVE] {phrase!r}")
            loop = asyncio.get_running_loop()

            def _speak():
                tts_client.feed(phrase)
                tts_client.flush()

            await loop.run_in_executor(_executor, _speak)
            global _last_activity
            _last_activity = time.monotonic()


async def run_pipeline() -> None:
    global tts_client, stt_client
    loop = asyncio.get_running_loop()

    tts_client = TTSClient(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        on_amplitude=on_amplitude,
        on_speaking=on_speaking,
        on_viseme=on_viseme,
        on_audio_chunk=send_audio_to_browser,
        loop=loop,
    )
    tts_client.set_mood(mood_machine.current_mood)

    stt_client = STTClient(
        api_key=DEEPGRAM_API_KEY,
        on_transcript=on_transcript,
    )

    set_audio_receive_callback(stt_client.receive_audio)

    await stt_client.start()
    print(f"[ENTITY] Listening on port {SERVER_PORT}. Open http://<this-ip>:{SERVER_PORT}/face on browser.")
    await asyncio.gather(
        mood_machine.run(on_change=on_mood_change),
        proactive_loop(),
    )


async def main() -> None:
    server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")
    )
    await asyncio.gather(
        server.serve(),
        run_pipeline(),
    )


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify startup (no sounddevice errors)**

```
venv\Scripts\python.exe -c "
import asyncio, sys
sys.path.insert(0, '.')
# Just check imports work without sounddevice
from main import on_transcript, on_amplitude, proactive_loop, PROACTIVE_PHRASES
assert 'hostile' in PROACTIVE_PHRASES
assert len(PROACTIVE_PHRASES['hostile']) >= 3
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```
git add main.py
git commit -m "feat: wire audio WebSocket to STT/TTS, add proactive speech loop"
```

---

## Task 6: Browser AudioWorklet processor

**Files:**
- Create: `face/audio-worklet.js`

- [ ] **Step 1: Create the AudioWorklet processor**

```javascript
/**
 * Converts the microphone's float32 samples to int16 PCM and posts
 * each buffer to the main thread via the MessagePort.
 */
class MicProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    const pcm = new Int16Array(channel.length);
    for (let i = 0; i < channel.length; i++) {
      pcm[i] = Math.max(-32768, Math.min(32767, Math.round(channel[i] * 32767)));
    }
    // Transfer the buffer (zero-copy) to avoid serialisation overhead
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
    return true;
  }
}

registerProcessor('mic-processor', MicProcessor);
```

- [ ] **Step 2: Verify file is served**

Start the server and request the file:

```
venv\Scripts\python.exe -m uvicorn ws_server:app --port 8000
# In another terminal:
curl -s http://localhost:8000/face/audio-worklet.js | head -5
```

Expected: first line is `/**`

- [ ] **Step 3: Commit**

```
git add face/audio-worklet.js
git commit -m "feat: AudioWorklet processor for float32→int16 mic capture"
```

---

## Task 7: Browser setup screen + audio (index.html)

**Files:**
- Modify: `face/index.html`

This is the largest browser-side change. The file gets a setup screen overlay and all audio JS.

- [ ] **Step 1: Replace index.html with the new version**

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
  <title>ENTITY</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #000; overflow: hidden; width: 100vw; height: 100vh; }
    #matrix     { position: fixed; top: 0; left: 0; z-index: 0; }
    #face-canvas { position: fixed; top: 0; left: 0; z-index: 1; }
    #scanlines {
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      z-index: 2; pointer-events: none;
      background: repeating-linear-gradient(
        0deg, transparent, transparent 3px,
        rgba(0,255,0,0.015) 3px, rgba(0,255,0,0.015) 4px
      );
    }
    #bottom-fade {
      position: fixed; bottom: 0; left: 0; width: 100%; height: 40%;
      z-index: 4; pointer-events: none;
      background: linear-gradient(to bottom, transparent 0%, #000000 100%);
    }
    #subtitle {
      position: fixed; bottom: 40px; left: 0; right: 0;
      text-align: center; z-index: 5;
      font-family: 'Courier New', monospace;
      font-size: 18px; color: #00ffff;
      text-shadow: 0 0 10px currentColor;
      opacity: 0; transition: opacity 0.2s;
      padding: 0 20px;
    }

    /* ── Setup overlay ─────────────────────────────────────── */
    #setup {
      position: fixed; inset: 0; z-index: 100;
      background: #000;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      gap: 20px; padding: 40px;
      font-family: 'Courier New', monospace; color: #eee;
    }
    #setup h1 {
      font-size: 2em; letter-spacing: 6px; color: #00ffff;
      text-shadow: 0 0 20px #00ffff;
    }
    #setup p.sub { color: #555; font-size: 0.8em; letter-spacing: 2px; }
    .setup-group { width: 100%; max-width: 360px; display: flex; flex-direction: column; gap: 6px; }
    .setup-group label { color: #666; font-size: 0.72em; letter-spacing: 1px; }
    .setup-group select {
      background: #111; border: 1px solid #333; color: #eee;
      padding: 10px 12px; border-radius: 4px; font-size: 0.9em;
      font-family: 'Courier New', monospace; width: 100%;
    }
    #start-btn {
      margin-top: 10px; padding: 12px 40px;
      background: #00ffff; color: #000;
      border: none; font-family: 'Courier New', monospace;
      font-size: 1em; letter-spacing: 3px; cursor: pointer;
      border-radius: 4px;
    }
    #start-btn:hover { background: #00cccc; }
    #setup-error { color: #ff4444; font-size: 0.8em; min-height: 1.2em; }
  </style>
</head>
<body>
  <!-- Setup overlay (shown first) -->
  <div id="setup">
    <h1>ENTITY</h1>
    <p class="sub">Configuración de audio</p>

    <div class="setup-group">
      <label>MICRÓFONO</label>
      <select id="mic-select"><option value="">Cargando...</option></select>
    </div>

    <div class="setup-group" id="out-group">
      <label>SALIDA DE AUDIO</label>
      <select id="out-select"><option value="">Por defecto del sistema</option></select>
    </div>

    <p id="setup-error"></p>
    <button id="start-btn">INICIAR</button>
  </div>

  <!-- Face (hidden until setup complete) -->
  <canvas id="matrix"></canvas>
  <canvas id="face-canvas" style="display:none"></canvas>
  <div id="scanlines" style="display:none"></div>
  <div id="bottom-fade" style="display:none"></div>
  <div id="subtitle" style="display:none"></div>

  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
    }
  }
  </script>
  <script src="matrix.js"></script>

  <script>
  // ── Setup screen logic ─────────────────────────────────────────
  const WS_PROTO  = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const WS_BASE   = `${WS_PROTO}//${location.host}`;
  const CAPTURE_RATE = 16000;  // must match BROWSER_CAPTURE_RATE in config.py

  let micStream = null;
  let captureCtx = null;
  let audioWs = null;
  var isMuted = false;  // var so face.js module can read it as window.isMuted

  async function populateDevices() {
    // Request mic permission first so labels are populated
    try {
      const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
      tmp.getTracks().forEach(t => t.stop());
    } catch (_) {}

    const devices = await navigator.mediaDevices.enumerateDevices();
    const micSel = document.getElementById('mic-select');
    const outSel = document.getElementById('out-select');

    micSel.innerHTML = '';
    devices.filter(d => d.kind === 'audioinput').forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `Micrófono ${micSel.options.length + 1}`;
      micSel.appendChild(opt);
    });

    const supportsOutput = typeof AudioContext.prototype.setSinkId === 'function' ||
                           typeof HTMLAudioElement.prototype.setSinkId === 'function';
    if (supportsOutput) {
      outSel.innerHTML = '<option value="">Por defecto del sistema</option>';
      devices.filter(d => d.kind === 'audiooutput').forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.deviceId;
        opt.textContent = d.label || `Salida ${outSel.options.length}`;
        outSel.appendChild(opt);
      });
    } else {
      document.getElementById('out-group').style.display = 'none';
    }
  }

  document.getElementById('start-btn').addEventListener('click', async () => {
    const errEl = document.getElementById('setup-error');
    errEl.textContent = '';
    const micId  = document.getElementById('mic-select').value;
    const outId  = document.getElementById('out-select').value;

    try {
      await startAudio(micId, outId);
      showFace();
    } catch (e) {
      errEl.textContent = `Error: ${e.message}`;
    }
  });

  async function startAudio(micId, outId) {
    // ── Capture ──────────────────────────────────────────────
    captureCtx = new AudioContext({ sampleRate: CAPTURE_RATE });
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { deviceId: micId ? { exact: micId } : undefined, channelCount: 1 },
    });

    await captureCtx.audioWorklet.addModule('audio-worklet.js');
    const source    = captureCtx.createMediaStreamSource(micStream);
    const processor = new AudioWorkletNode(captureCtx, 'mic-processor');

    audioWs = new WebSocket(`${WS_BASE}/audio`);
    audioWs.binaryType = 'arraybuffer';

    audioWs.addEventListener('open', () => {
      processor.port.onmessage = (e) => {
        if (!isMuted && audioWs.readyState === WebSocket.OPEN) {
          audioWs.send(e.data);
        }
      };
      source.connect(processor);
    });

    // ── Playback ─────────────────────────────────────────────
    const playCtx = new AudioContext({ sampleRate: 24000 });
    if (outId && typeof playCtx.setSinkId === 'function') {
      await playCtx.setSinkId(outId).catch(() => {});
    }
    let scheduledUntil = 0;

    audioWs.addEventListener('message', (e) => {
      if (!(e.data instanceof ArrayBuffer)) return;
      const pcm16   = new Int16Array(e.data);
      const samples = pcm16.length;
      if (samples === 0) return;

      const buf = playCtx.createBuffer(1, samples, 24000);
      const ch  = buf.getChannelData(0);
      for (let i = 0; i < samples; i++) ch[i] = pcm16[i] / 32768;

      const src = playCtx.createBufferSource();
      src.buffer = buf;
      src.connect(playCtx.destination);
      const startAt = Math.max(playCtx.currentTime + 0.05, scheduledUntil);
      src.start(startAt);
      scheduledUntil = startAt + samples / 24000;
    });

    audioWs.addEventListener('close', () => {
      setTimeout(() => startAudio(micId, outId), 2000);
    });
  }

  function showFace() {
    document.getElementById('setup').style.display = 'none';
    document.getElementById('face-canvas').style.display = '';
    document.getElementById('scanlines').style.display = '';
    document.getElementById('bottom-fade').style.display = '';
    document.getElementById('subtitle').style.display = '';
  }

  populateDevices();
  </script>

  <script type="module" src="face.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify setup screen loads**

Start the server and open `http://localhost:8000/face/` in Chrome. The setup screen should appear with mic selectors.

- [ ] **Step 3: Commit**

```
git add face/index.html
git commit -m "feat: setup screen with device selector, browser mic capture, Web Audio playback"
```

---

## Task 8: Fix WebSocket URL in face.js for HTTPS/cloud

**Files:**
- Modify: `face/face.js` line 230

- [ ] **Step 1: Replace the WS_URL line**

Find line 230:
```javascript
  const WS_URL   = `ws://${location.host}/ws`;
```

Replace with:
```javascript
  const WS_PROTO = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const WS_URL   = `${WS_PROTO}//${location.host}/ws`;
```

Also update the existing `speaking` block in `ws.onmessage` (around line 243) to set `window.isMuted`:

```javascript
      if (msg.type === 'speaking') {
        window.isMuted = msg.value;  // read by inline script's mic processor
        if (!msg.value) {
          amplitudeShapes = {};
          targetShapes    = {};
        }
      }
```

Remove the old `if (msg.type === 'speaking' && !msg.value)` block — the new block above replaces it.

- [ ] **Step 2: Verify face still connects on local**

Open `http://localhost:8000/face/` (after starting the server), complete setup, check browser console — should show `[FACE] Morph targets: [...]` and no WS errors.

- [ ] **Step 3: Commit**

```
git add face/face.js
git commit -m "fix: use wss:// on HTTPS for cloud compatibility; mute mic on speaking event"
```

---

## Task 9: Dockerfile + render.yaml

**Files:**
- Create: `Dockerfile`
- Create: `render.yaml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

- [ ] **Step 2: Create render.yaml**

```yaml
services:
  - type: web
    name: sculpture-ai
    runtime: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: DEEPGRAM_API_KEY
        sync: false
      - key: GROQ_API_KEY
        sync: false
      - key: ELEVENLABS_API_KEY
        sync: false
      - key: ELEVENLABS_VOICE_ID
        sync: false
      - key: SERVER_PORT
        value: "8000"
      - key: PROACTIVE_INTERVAL
        value: "180"
    healthCheckPath: /health
```

- [ ] **Step 3: Test Docker build locally**

```
docker build -t sculpture-ai .
docker run --rm -p 8000:8000 --env-file .env sculpture-ai
```

Expected: server starts, `[ENTITY] Listening on port 8000` appears, `http://localhost:8000/face/` shows setup screen.

- [ ] **Step 4: Add .superpowers to .gitignore**

```
echo ".superpowers/" >> .gitignore
```

- [ ] **Step 5: Commit**

```
git add Dockerfile render.yaml .gitignore
git commit -m "feat: Dockerfile and render.yaml for cloud deploy"
```

---

## Task 10: Deploy to Render

- [ ] **Step 1: Push to GitHub**

```
git remote -v   # confirm remote exists
git push origin master
```

- [ ] **Step 2: Create Render account and new Web Service**

Go to `https://render.com` → New → Web Service → Connect GitHub repo.

- [ ] **Step 3: Set environment variables in Render dashboard**

In the service settings → Environment, add:
- `DEEPGRAM_API_KEY` = (value from .env)
- `GROQ_API_KEY` = (value from .env)
- `ELEVENLABS_API_KEY` = (value from .env)
- `ELEVENLABS_VOICE_ID` = (value from .env)

- [ ] **Step 4: Deploy and verify**

After deploy completes, open `https://<your-service>.onrender.com/face/` on iPad.
- Setup screen appears ✓
- Select mic, click INICIAR ✓
- Speak → transcript appears + face animates + voice replies ✓
- Wait 3 min → proactive phrase plays automatically ✓

- [ ] **Step 5: Check health endpoint**

```
curl https://<your-service>.onrender.com/health
```

Expected: `{"ok":true}`

---

## Verification Checklist

- [ ] `python main.py` locally → setup screen at `http://localhost:8000/face/`
- [ ] Select Focusrite mic → speak → transcript + face + voice reply works
- [ ] Proactive phrase fires after 3 min of silence
- [ ] `docker build . && docker run -p 8000:8000 --env-file .env .` → same as above
- [ ] Render deploy → HTTPS URL works on iPad (mic permission granted, full flow)
- [ ] `/health` returns `{"ok":true}`
- [ ] All existing tests pass: `pytest tests/ -v`
