import asyncio
import sounddevice as sd
from deepgram import DeepgramClient, AsyncDeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.core.events import EventType
from typing import Callable, Awaitable
from config import (
    DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE,
    DEEPGRAM_ENDPOINTING_MS,
    AUDIO_CAPTURE_RATE, AUDIO_CHUNK_SIZE, AUDIO_INPUT_DEVICE,
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
        self._audio_stream = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._listen_task = None

    async def _handle_transcript(self, result) -> None:
        sentence = result.channel.alternatives[0].transcript
        if result.speech_final and sentence.strip():
            await self._on_transcript(sentence)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        connected_event = asyncio.Event()

        async def _run_connection():
            async with self._dg.listen.v1.connect(
                model=DEEPGRAM_MODEL,
                language=DEEPGRAM_LANGUAGE,
                smart_format=True,
                interim_results=False,
                vad_events=True,
                endpointing=DEEPGRAM_ENDPOINTING_MS,
                sample_rate=AUDIO_CAPTURE_RATE,
                encoding="linear16",
            ) as conn:
                self._connection = conn
                connected_event.set()  # Signal that the WebSocket connection is ready

                async def on_message(msg):
                    if isinstance(msg, ListenV1Results):
                        await self._handle_transcript(msg)

                async def on_error(err):
                    print(f"[STT] Deepgram error: {err}")

                conn.on(EventType.MESSAGE, on_message)
                conn.on(EventType.ERROR, on_error)

                await conn.start_listening()

        self._listen_task = asyncio.create_task(_run_connection())

        # Wait until the WebSocket handshake completes before starting audio
        await connected_event.wait()

        def audio_callback(indata, frames, time, status):
            if self._connection and self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._connection.send_media(bytes(indata)),
                    self._loop,
                )

        self._audio_stream = sd.RawInputStream(
            device=AUDIO_INPUT_DEVICE,
            samplerate=AUDIO_CAPTURE_RATE,
            channels=1,
            dtype="int16",
            blocksize=AUDIO_CHUNK_SIZE,
            callback=audio_callback,
        )
        self._audio_stream.start()

    async def stop(self) -> None:
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream = None
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
