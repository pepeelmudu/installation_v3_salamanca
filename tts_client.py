import re
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


def _rms(pcm_bytes: bytes) -> float:
    if len(pcm_bytes) < 2:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(min(np.sqrt(np.mean(samples ** 2)) / 32768.0, 1.0))


def _is_sentence_end(text: str, min_chars: int) -> bool:
    return len(text) >= min_chars and bool(SENTENCE_END_RE.search(text.rstrip()))


SENTENCE_MIN_CHARS = SENTENCE_MIN_CHARS


class TTSClient:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        on_amplitude: Callable[[float], Awaitable[None]],
        on_speaking: Callable[[bool], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
    ):
        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._on_amplitude = on_amplitude
        self._on_speaking = on_speaking
        self._loop = loop
        self._buffer = ""
        self._audio_stream = sd.RawOutputStream(
            samplerate=AUDIO_PLAYBACK_RATE, channels=1, dtype="int16"
        )
        self._audio_stream.start()

    def feed(self, token: str) -> None:
        self._buffer += token
        if _is_sentence_end(self._buffer, SENTENCE_MIN_CHARS):
            text = self._buffer.strip()
            self._buffer = ""
            threading.Thread(target=self._synthesize_and_play, args=(text,), daemon=True).start()

    def flush(self) -> None:
        if self._buffer.strip():
            text = self._buffer.strip()
            self._buffer = ""
            threading.Thread(target=self._synthesize_and_play, args=(text,), daemon=True).start()

    def _synthesize_and_play(self, text: str) -> None:
        asyncio.run_coroutine_threadsafe(self._on_speaking(True), self._loop)
        try:
            voice_settings = self._mood_voice_settings()
            audio_gen = self._client.text_to_speech.convert_as_stream(
                voice_id=self._voice_id,
                text=text,
                model_id=ELEVENLABS_MODEL,
                output_format=ELEVENLABS_FORMAT,
                voice_settings=voice_settings,
            )
            for chunk in audio_gen:
                if chunk:
                    self._audio_stream.write(chunk)
                    amp = _rms(chunk)
                    asyncio.run_coroutine_threadsafe(self._on_amplitude(amp), self._loop)
        finally:
            asyncio.run_coroutine_threadsafe(self._on_speaking(False), self._loop)

    def set_mood(self, mood_id: str) -> None:
        self._current_mood = mood_id

    def _mood_voice_settings(self) -> VoiceSettings:
        mood = getattr(self, "_current_mood", "friendly")
        presets = {
            "friendly":     VoiceSettings(stability=0.7, similarity_boost=0.8, style=0.2),
            "hostile":      VoiceSettings(stability=0.2, similarity_boost=0.6, style=0.8),
            "surreal":      VoiceSettings(stability=0.1, similarity_boost=0.5, style=0.9),
            "paranoid":     VoiceSettings(stability=0.3, similarity_boost=0.7, style=0.7),
            "dismissive":   VoiceSettings(stability=0.5, similarity_boost=0.7, style=0.3),
            "philosophical":VoiceSettings(stability=0.8, similarity_boost=0.9, style=0.1),
        }
        return presets.get(mood, presets["friendly"])

    def close(self) -> None:
        self._audio_stream.stop()
        self._audio_stream.close()
