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
    ELEVENLABS_MODEL, ELEVENLABS_MODEL_V3, ELEVENLABS_FORMAT,
    AUDIO_PLAYBACK_RATE, SENTENCE_MIN_CHARS,
)

SENTENCE_END_RE = re.compile(r'[.!?…]+\s*$')
_SENTENCE_END = object()  # sentinel: end of one sentence's PCM

SENTENCE_MIN_CHARS = SENTENCE_MIN_CHARS


class _SynthJob:
    """One utterance to synthesize. model_id/voice_settings default to the
    conversational Flash preset when None."""
    __slots__ = ("text", "model_id", "voice_settings", "use_timestamps")

    def __init__(self, text, model_id=None, voice_settings=None, use_timestamps=True):
        self.text = text
        self.model_id = model_id or ELEVENLABS_MODEL
        self.voice_settings = voice_settings
        self.use_timestamps = use_timestamps


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
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm_bytes[: n * 2])
    mean_sq = sum(s * s for s in samples) / n
    return min(mean_sq ** 0.5 / 32768.0, 1.0)


def _is_sentence_end(text: str, min_chars: int) -> bool:
    return len(text) >= min_chars and bool(SENTENCE_END_RE.search(text.rstrip()))


class TTSClient:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        on_amplitude: Callable[[float], Awaitable[None]],
        on_speaking: Callable[[bool], Awaitable[None]],
        on_viseme_schedule: Callable[[list], Awaitable[None]],
        on_audio_chunk: Callable[[bytes], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
    ):
        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._on_amplitude = on_amplitude
        self._on_speaking = on_speaking
        self._on_viseme_schedule = on_viseme_schedule
        self._on_audio_chunk = on_audio_chunk
        self._loop = loop
        self._buffer = ""

        self._pending = 0
        self._flushed = False  # True once flush() has been called for current turn
        self._lock = threading.Lock()

        self._synth_queue: queue.Queue[_SynthJob | None] = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._synth_thread.start()

        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    # ── workers ────────────────────────────────────────────────────

    def _synth_worker(self) -> None:
        while True:
            job = self._synth_queue.get()
            if job is None:
                self._audio_queue.put(None)
                break
            try:
                self._synth_job(job)
            except Exception as e:
                print(f"[TTS ERROR] {e!r}")
            finally:
                self._audio_queue.put(_SENTENCE_END)

    def _synth_job(self, job: "_SynthJob") -> None:
        settings = job.voice_settings or self._mood_voice_settings()
        has_timestamps = hasattr(self._client.text_to_speech, 'stream_with_timestamps')

        if not (job.use_timestamps and has_timestamps):
            self._stream_plain(job, settings)
            return

        alignment_chars: list[str] = []
        alignment_times: list[float] = []
        audio_chunks: list[bytes] = []
        try:
            gen = self._client.text_to_speech.stream_with_timestamps(
                self._voice_id, text=job.text, model_id=job.model_id,
                output_format=ELEVENLABS_FORMAT, voice_settings=settings,
            )
            for chunk in gen:
                audio = self._extract_audio(chunk)
                if audio:
                    audio_chunks.append(audio)
                alignment = getattr(chunk, 'alignment', None)
                if alignment:
                    alignment_chars.extend(list(getattr(alignment, 'characters', []) or []))
                    alignment_times.extend(list(getattr(alignment, 'character_start_times_seconds', []) or []))
        except Exception as e:
            print(f"[TTS] stream_with_timestamps failed ({e!r}), falling back")
            self._stream_plain(job, settings)
            return

        if not audio_chunks:
            print("[TTS] no audio from timestamps path, falling back to plain", flush=True)
            self._stream_plain(job, settings)
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

    def _stream_plain(self, job: "_SynthJob", settings) -> None:
        try:
            for chunk in self._client.text_to_speech.stream(
                self._voice_id, text=job.text, model_id=job.model_id,
                output_format=ELEVENLABS_FORMAT, voice_settings=settings,
            ):
                if chunk:
                    self._audio_queue.put(chunk)
        except Exception as e:
            if job.model_id == ELEVENLABS_MODEL:
                raise
            # e.g. v3 not available on the streaming endpoint → fall back to Flash.
            clean = re.sub(r'\[[^\]]*\]\s*', '', job.text).strip()
            print(f"[TTS] model {job.model_id} failed ({e!r}); retrying on Flash", flush=True)
            for chunk in self._client.text_to_speech.stream(
                self._voice_id, text=clean, model_id=ELEVENLABS_MODEL,
                output_format=ELEVENLABS_FORMAT, voice_settings=settings,
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
                turn_ended = False
                with self._lock:
                    self._pending -= 1
                    if self._pending == 0 and self._flushed:
                        self._flushed = False
                        turn_ended = playing
                if turn_ended:
                    playing = False
                    asyncio.run_coroutine_threadsafe(self._on_speaking(False), self._loop)
                pending_alignment = None
                continue

            # Real PCM chunk — send to browser via callback
            if not playing:
                playing = True
                asyncio.run_coroutine_threadsafe(self._on_speaking(True), self._loop)

            if pending_alignment:
                self._send_viseme_schedule(pending_alignment)
                pending_alignment = None

            try:
                asyncio.run_coroutine_threadsafe(
                    self._on_audio_chunk(chunk), self._loop
                )
                amp = _rms(chunk)
                asyncio.run_coroutine_threadsafe(self._on_amplitude(amp), self._loop)
            except Exception:
                pass

    def _send_viseme_schedule(self, alignment: _AlignmentData) -> None:
        """Build viseme schedule for browser-side scheduling anchored to actual playback time."""
        raw = [
            (st, CHAR_VISEMES[c.lower()])
            for c, st in zip(alignment.chars, alignment.start_times)
            if c.lower() in CHAR_VISEMES
        ]
        if not raw:
            return

        events: list[dict] = []
        for i, (st, shapes) in enumerate(raw):
            events.append({"at": st, "shapes": shapes})
            if i + 1 < len(raw):
                gap = raw[i + 1][0] - st
                if gap > 0.10:  # word gap — close mouth mid-gap
                    events.append({"at": st + min(gap * 0.5, 0.08), "shapes": {}})
            else:
                events.append({"at": st + 0.10, "shapes": {}})  # close after last viseme

        asyncio.run_coroutine_threadsafe(
            self._on_viseme_schedule(events), self._loop
        )

    # ── public API ─────────────────────────────────────────────────

    def feed(self, token: str) -> None:
        self._buffer += token
        if _is_sentence_end(self._buffer, SENTENCE_MIN_CHARS):
            with self._lock:
                self._pending += 1
            self._synth_queue.put(_SynthJob(self._buffer.strip()))
            self._buffer = ""

    def flush(self) -> None:
        with self._lock:
            if self._buffer.strip():
                self._pending += 1
                self._synth_queue.put(_SynthJob(self._buffer.strip()))
                self._buffer = ""
            self._flushed = True

    def flush_buffer(self) -> None:
        """Enqueue any buffered partial text as a sentence WITHOUT ending the
        turn. Used to guarantee ordering before injecting a separate utterance
        mid-response (unlike flush(), it does not set _flushed)."""
        with self._lock:
            if self._buffer.strip():
                self._pending += 1
                self._synth_queue.put(_SynthJob(self._buffer.strip()))
                self._buffer = ""

    def set_mood(self, mood_id: str) -> None:
        self._current_mood = mood_id

    def _mood_voice_settings(self, mood: str | None = None) -> VoiceSettings:
        mood = mood or getattr(self, "_current_mood", "friendly")
        presets = {
            "friendly":      VoiceSettings(stability=0.7, similarity_boost=0.8, style=0.2),
            "hostile":       VoiceSettings(stability=0.2, similarity_boost=0.6, style=0.8),
            "surreal":       VoiceSettings(stability=0.1, similarity_boost=0.5, style=0.9),
            "paranoid":      VoiceSettings(stability=0.3, similarity_boost=0.7, style=0.7),
            "dismissive":    VoiceSettings(stability=0.5, similarity_boost=0.7, style=0.3),
            "philosophical": VoiceSettings(stability=0.8, similarity_boost=0.9, style=0.1),
            "glitch":        VoiceSettings(stability=0.15, similarity_boost=0.5, style=0.9),
            "shout":         VoiceSettings(stability=0.1, similarity_boost=0.5, style=1.0, speed=1.2),
            "whisper":       VoiceSettings(stability=0.9, similarity_boost=0.8, style=0.1),
            "normal":        VoiceSettings(stability=0.4, similarity_boost=0.7, style=0.4),
        }
        return presets.get(mood, presets["friendly"])

    def say_special(self, text: str, mood: str = "shout",
                    model_id: str | None = None, flush: bool = False) -> None:
        """Enqueue a standalone utterance (outburst / injection / deflection) with
        its own voice preset. Synthesized on Flash WITH timestamps so the mouth
        lip-syncs exactly like normal speech (visemes). Any v3 audio tags are
        stripped — Flash would read them aloud. The shout still lands via CAPS +
        the intense voice preset.

        Precondition: use flush=True ONLY for a standalone utterance when no normal
        conversational turn is in flight (proactive outbursts / deflections). For
        mid-response injections use flush=False and let the following flush() close
        the turn.
        """
        settings = self._voice_settings_for(mood)
        clean = re.sub(r'\[[^\]]*\]\s*', '', text).strip()
        with self._lock:
            self._pending += 1
            if flush:
                self._flushed = True
        self._synth_queue.put(_SynthJob(
            clean, model_id=model_id or ELEVENLABS_MODEL,
            voice_settings=settings, use_timestamps=True,
        ))

    def _voice_settings_for(self, mood: str) -> VoiceSettings:
        return self._mood_voice_settings(mood)

    def close(self) -> None:
        self._synth_queue.put(None)
        self._synth_thread.join(timeout=5)
        self._playback_thread.join(timeout=2)
