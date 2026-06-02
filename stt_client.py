import asyncio
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
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
        self._muted = False
        self._running = False
        self._listen_task = None
        self._language = DEEPGRAM_LANGUAGE

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def set_language(self, language: str) -> None:
        """Change recognition language and force a reconnect so the new
        language takes effect. The reconnect loop already handles re-dialing."""
        if language == self._language:
            return
        self._language = language
        conn = self._connection
        if conn:
            try:
                asyncio.get_running_loop().create_task(conn.finish())
            except Exception:
                pass

    async def receive_audio(self, data: bytes) -> None:
        if not self._connection:
            return
        payload = bytes(len(data)) if self._muted else data
        await self._connection.send(payload)

    async def _run_connection_loop(self, connected_event: asyncio.Event) -> None:
        first = True
        while self._running:
            conn = None
            closed_event = asyncio.Event()
            try:
                options = LiveOptions(
                    model=DEEPGRAM_MODEL,
                    language=self._language,
                    smart_format=True,
                    interim_results=False,
                    vad_events=True,
                    endpointing=DEEPGRAM_ENDPOINTING_MS,
                    sample_rate=BROWSER_CAPTURE_RATE,
                    encoding="linear16",
                )

                conn = self._dg.listen.asyncwebsocket.v("1")
                on_transcript_cb = self._on_transcript

                async def on_message(self_inner, result, **kwargs):
                    try:
                        sentence = result.channel.alternatives[0].transcript
                        print(f"[STT] speech_final={result.speech_final} transcript={sentence!r}", flush=True)
                        if result.speech_final and sentence.strip():
                            await on_transcript_cb(sentence)
                    except Exception as ex:
                        print(f"[STT] on_message error: {ex}", flush=True)

                async def on_error(self_inner, error, **kwargs):
                    print(f"[STT] Deepgram error: {error}", flush=True)

                async def on_close(self_inner, **kwargs):
                    closed_event.set()

                conn.on(LiveTranscriptionEvents.Transcript, on_message)
                conn.on(LiveTranscriptionEvents.Error, on_error)
                conn.on(LiveTranscriptionEvents.Close, on_close)

                started = await conn.start(options)
                if not started:
                    raise Exception("Deepgram start() returned False")

                self._connection = conn
                if first:
                    print("[STT] Connected to Deepgram OK", flush=True)
                    connected_event.set()
                    first = False
                else:
                    print("[STT] Reconnected to Deepgram", flush=True)

                await closed_event.wait()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[STT] Connection dropped ({e!r}), reconnecting in 2s…", flush=True)
            finally:
                self._connection = None
                if conn:
                    try:
                        await conn.finish()
                    except Exception:
                        pass

            if self._running:
                await asyncio.sleep(2)

    async def start(self) -> None:
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
