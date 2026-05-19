import asyncio
import sounddevice as sd
from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.core.events import EventType
from typing import Callable, Awaitable
from config import (
    DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE,
    DEEPGRAM_ENDPOINTING_MS,
    AUDIO_CAPTURE_RATE, AUDIO_CHUNK_SIZE, AUDIO_INPUT_DEVICE,
)

_SILENCE = bytes(AUDIO_CHUNK_SIZE * 2)  # int16 mono zeros — keeps Deepgram alive


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
        self._muted = False
        self._running = False

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

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
                    sample_rate=AUDIO_CAPTURE_RATE,
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

        def audio_callback(indata, frames, time, status):
            if not self._connection or not self._loop:
                return
            data = _SILENCE if self._muted else bytes(indata)
            try:
                asyncio.run_coroutine_threadsafe(
                    self._connection.send_media(data),
                    self._loop,
                )
            except Exception:
                pass

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
        self._running = False
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
