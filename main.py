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
    global _last_activity
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
