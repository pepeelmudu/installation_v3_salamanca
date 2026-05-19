import re
import time
import base64
import queue
import asyncio
import threading
import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from typing import Callable, Awaitable
from config import (
    ELEVENLABS_MODEL, ELEVENLABS_FORMAT,
    AUDIO_PLAYBACK_RATE, SENTENCE_MIN_CHARS,
)

SENTENCE_END_RE = re.compile(r'[.!?…]+\s*$')
_SENTENCE_END = object()  # sentinel: end of one sentence's PCM

SENTENCE_MIN_CHARS = SENTENCE_MIN_CHARS

# ARKit blend shapes per Spanish character
CHAR_VISEMES: dict[str, dict[str, float]] = {
    'a': {'jawOpen': 0.8,  'mouthShrugLower': 0.3},
    'á': {'jawOpen': 0.8,  'mouthShrugLower': 0.3},
    'e': {'jawOpen': 0.4,  'mouthSmileLeft': 0.3,  'mouthSmileRight': 0.3},
    'é': {'jawOpen': 0.4,  'mouthSmileLeft': 0.3,  'mouthSmileRight': 0.3},
    'i': {'jawOpen': 0.15, 'mouthSmileLeft': 0.6,  'mouthSmileRight': 0.6},
    'í': {'jawOpen': 0.15, 'mouthSmileLeft': 0.6,  'mouthSmileRight': 0.6},
    'y': {'jawOpen': 0.15, 'mouthSmileLeft': 0.5,  'mouthSmileRight': 0.5},
    'o': {'jawOpen': 0.5,  'mouthFunnel': 0.6},
    'ó': {'jawOpen': 0.5,  'mouthFunnel': 0.6},
    'u': {'jawOpen': 0.25, 'mouthPucker': 0.7},
    'ú': {'jawOpen': 0.25, 'mouthPucker': 0.7},
    'ü': {'jawOpen': 0.25, 'mouthPucker': 0.7},
    'p': {'mouthClose': 0.9,  'mouthPressLeft': 0.5, 'mouthPressRight': 0.5},
    'b': {'mouthClose': 0.9,  'mouthPressLeft': 0.5, 'mouthPressRight': 0.5},
    'm': {'mouthClose': 1.0},
    'f': {'mouthLowerDownLeft': 0.5, 'mouthLowerDownRight': 0.5},
    'v': {'mouthLowerDownLeft': 0.5, 'mouthLowerDownRight': 0.5},
    's': {'jawOpen': 0.2, 'mouthStretchLeft': 0.2, 'mouthStretchRight': 0.2},
    'z': {'jawOpen': 0.2, 'mouthStretchLeft': 0.2, 'mouthStretchRight': 0.2},
    'c': {'jawOpen': 0.2, 'mouthStretchLeft': 0.15,'mouthStretchRight': 0.15},
    'r': {'jawOpen': 0.2, 'mouthShrugLower': 0.3},
    'l': {'jawOpen': 0.2},
    'n': {'jawOpen': 0.15},
    'ñ': {'jawOpen': 0.15, 'mouthFunnel': 0.2},
    'd': {'jawOpen': 0.2, 'mouthShrugUpper': 0.2},
    't': {'jawOpen': 0.2, 'mouthShrugUpper': 0.2},
    'g': {'jawOpen': 0.3},
    'k': {'jawOpen': 0.3},
    'q': {'jawOpen': 0.3},
    'j': {'jawOpen': 0.3},
    'h': {'jawOpen': 0.1},
    'x': {'jawOpen': 0.25, 'mouthFunnel': 0.2},
}


class _AlignmentData:
    __slots__ = ('chars', 'start_times')

    def __init__(self, chars: list[str], start_times: list[float]) -> None:
        self.chars = chars
        self.start_times = start_times


def _rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < 2:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(min(np.sqrt(np.mean(samples ** 2)) / 32768.0, 1.0))


def _is_sentence_end(text: str, min_chars: int) -> bool:
    return len(text) >= min_chars and bool(SENTENCE_END_RE.search(text.rstrip()))


class TTSClient:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        on_amplitude: Callable[[float], Awaitable[None]],
        on_speaking: Callable[[bool], Awaitable[None]],
        on_viseme: Callable[[dict], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
    ):
        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._on_amplitude = on_amplitude
        self._on_speaking = on_speaking
        self._on_viseme = on_viseme
        self._loop = loop
        self._buffer = ""

        self._pending = 0
        self._lock = threading.Lock()

        self._synth_queue: queue.Queue[str | None] = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        self._audio_stream = sd.RawOutputStream(
            samplerate=AUDIO_PLAYBACK_RATE, channels=1, dtype="int16"
        )
        self._audio_stream.start()

        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._synth_thread.start()

        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    # ── workers ────────────────────────────────────────────────────

    def _synth_worker(self) -> None:
        while True:
            text = self._synth_queue.get()
            if text is None:
                self._audio_queue.put(None)
                break
            try:
                self._synth_sentence(text)
            except Exception as e:
                print(f"[TTS ERROR] {e!r}")
            finally:
                self._audio_queue.put(_SENTENCE_END)

    def _synth_sentence(self, text: str) -> None:
        # Check if timestamps endpoint exists before calling it
        has_timestamps = hasattr(self._client.text_to_speech, 'stream_with_timestamps')

        if not has_timestamps:
            self._stream_plain(text)
            return

        alignment_chars: list[str] = []
        alignment_times: list[float] = []
        audio_chunks: list[bytes] = []

        try:
            gen = self._client.text_to_speech.stream_with_timestamps(
                self._voice_id,
                text=text,
                model_id=ELEVENLABS_MODEL,
                output_format=ELEVENLABS_FORMAT,
                voice_settings=self._mood_voice_settings(),
            )
            for chunk in gen:
                audio = self._extract_audio(chunk)
                if audio:
                    audio_chunks.append(audio)
                alignment = getattr(chunk, 'alignment', None)
                if alignment:
                    chars = list(getattr(alignment, 'characters', []) or [])
                    times = list(getattr(alignment, 'character_start_times_seconds', []) or [])
                    alignment_chars.extend(chars)
                    alignment_times.extend(times)

        except Exception as e:
            print(f"[TTS] stream_with_timestamps failed ({e!r}), falling back")
            self._stream_plain(text)
            return

        if not audio_chunks:
            print("[TTS] stream_with_timestamps returned no audio, falling back")
            self._stream_plain(text)
            return

        if alignment_chars:
            self._audio_queue.put(_AlignmentData(alignment_chars, alignment_times))
        for chunk in audio_chunks:
            self._audio_queue.put(chunk)

    @staticmethod
    def _extract_audio(chunk) -> bytes | None:
        """Extract PCM bytes from a chunk regardless of SDK version."""
        if isinstance(chunk, (bytes, bytearray)) and chunk:
            return bytes(chunk)
        for attr in ('audio_bytes', 'audio'):
            val = getattr(chunk, attr, None)
            if isinstance(val, (bytes, bytearray)) and val:
                return bytes(val)
        # Some SDK versions return base64-encoded audio
        for attr in ('audio_base_64', 'audio_base64', 'audio'):
            val = getattr(chunk, attr, None)
            if isinstance(val, str) and val:
                try:
                    return base64.b64decode(val)
                except Exception:
                    pass
        return None

    def _stream_plain(self, text: str) -> None:
        """Fallback: stream audio without timestamps."""
        for chunk in self._client.text_to_speech.stream(
            self._voice_id,
            text=text,
            model_id=ELEVENLABS_MODEL,
            output_format=ELEVENLABS_FORMAT,
            voice_settings=self._mood_voice_settings(),
        ):
            if chunk:
                self._audio_queue.put(chunk)

    def _playback_worker(self) -> None:
        playing = False
        pending_alignment: _AlignmentData | None = None

        while True:
            chunk = self._audio_queue.get()

            if chunk is None:  # shutdown
                if playing:
                    asyncio.run_coroutine_threadsafe(self._on_speaking(False), self._loop)
                break

            if isinstance(chunk, _AlignmentData):
                pending_alignment = chunk
                continue

            if chunk is _SENTENCE_END:
                with self._lock:
                    self._pending -= 1
                    remaining = self._pending
                if remaining == 0 and playing:
                    playing = False
                    asyncio.run_coroutine_threadsafe(self._on_speaking(False), self._loop)
                    asyncio.run_coroutine_threadsafe(self._on_viseme({}), self._loop)
                pending_alignment = None
                continue

            # Real PCM chunk
            if not playing:
                playing = True
                asyncio.run_coroutine_threadsafe(self._on_speaking(True), self._loop)

            # Schedule visemes for each sentence (fires on first chunk of each sentence)
            if pending_alignment:
                self._schedule_visemes(pending_alignment, time.monotonic())
                pending_alignment = None

            try:
                self._audio_stream.write(chunk)
                amp = _rms(chunk)
                asyncio.run_coroutine_threadsafe(self._on_amplitude(amp), self._loop)
            except Exception:
                pass

    def _schedule_visemes(self, alignment: _AlignmentData, t0: float) -> None:
        now = time.monotonic()
        raw = [
            (t0 + st - now, st, CHAR_VISEMES[c.lower()])
            for c, st in zip(alignment.chars, alignment.start_times)
            if c.lower() in CHAR_VISEMES
        ]
        if not raw:
            return

        events: list[tuple[float, dict]] = []
        for i, (delay, st, shapes) in enumerate(raw):
            events.append((max(0.0, delay), shapes))
            if i + 1 < len(raw):
                gap = raw[i + 1][1] - st
                if gap > 0.10:  # word gap — close mouth mid-gap
                    events.append((max(0.0, delay + min(gap * 0.5, 0.08)), {}))
            else:
                events.append((max(0.0, delay + 0.08), {}))  # close after last viseme

        async def _do_schedule() -> None:
            loop = asyncio.get_running_loop()
            for delay, shapes in events:
                loop.call_later(
                    delay,
                    lambda s=shapes: asyncio.run_coroutine_threadsafe(
                        self._on_viseme(s), self._loop
                    ),
                )

        asyncio.run_coroutine_threadsafe(_do_schedule(), self._loop)

    # ── public API ─────────────────────────────────────────────────

    def feed(self, token: str) -> None:
        self._buffer += token
        if _is_sentence_end(self._buffer, SENTENCE_MIN_CHARS):
            with self._lock:
                self._pending += 1
            self._synth_queue.put(self._buffer.strip())
            self._buffer = ""

    def flush(self) -> None:
        if self._buffer.strip():
            with self._lock:
                self._pending += 1
            self._synth_queue.put(self._buffer.strip())
            self._buffer = ""

    def set_mood(self, mood_id: str) -> None:
        self._current_mood = mood_id

    def _mood_voice_settings(self) -> VoiceSettings:
        mood = getattr(self, "_current_mood", "friendly")
        presets = {
            "friendly":      VoiceSettings(stability=0.7, similarity_boost=0.8, style=0.2),
            "hostile":       VoiceSettings(stability=0.2, similarity_boost=0.6, style=0.8),
            "surreal":       VoiceSettings(stability=0.1, similarity_boost=0.5, style=0.9),
            "paranoid":      VoiceSettings(stability=0.3, similarity_boost=0.7, style=0.7),
            "dismissive":    VoiceSettings(stability=0.5, similarity_boost=0.7, style=0.3),
            "philosophical": VoiceSettings(stability=0.8, similarity_boost=0.9, style=0.1),
        }
        return presets.get(mood, presets["friendly"])

    def close(self) -> None:
        self._synth_queue.put(None)
        self._synth_thread.join(timeout=5)
        self._playback_thread.join(timeout=2)
        self._audio_stream.stop()
        self._audio_stream.close()
