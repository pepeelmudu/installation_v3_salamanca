import asyncio
from deepgram import DeepgramClient
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
        self._dg = DeepgramClient(api_key=api_key)
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
                options = {
                    "model": DEEPGRAM_MODEL,
                    "language": DEEPGRAM_LANGUAGE,
                    "smart_format": True,
                    "interim_results": False,
                    "vad_events": True,
                    "endpointing": DEEPGRAM_ENDPOINTING_MS,
                    "sample_rate": BROWSER_CAPTURE_RATE,
                    "encoding": "linear16",
                }
                async with self._dg.listen.asyncwebsocket.v("1").connect(options) as conn:
                    self._connection = conn
                    if first:
                        print("[STT] Connected to Deepgram OK", flush=True)
                        connected_event.set()
                        first = False
                    else:
                        print("[STT] Reconnected to Deepgram", flush=True)

                    async def on_message(self_inner, result, **kwargs):
                        try:
                            sentence = result.channel.alternatives[0].transcript
                            print(f"[STT] result speech_final={result.speech_final} transcript={sentence!r}", flush=True)
                            if result.speech_final and sentence.strip():
                                await on_transcript_inner(sentence)
                        except Exception as ex:
                            print(f"[STT] on_message error: {ex}", flush=True)

                    async def on_error(self_inner, error, **kwargs):
                        print(f"[STT] Deepgram error: {error}", flush=True)

                    on_transcript_inner = self._on_transcript
                    conn.on("Results", on_message)
                    conn.on("Error", on_error)
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
                await self._connection.finish()
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
