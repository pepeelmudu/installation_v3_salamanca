import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from config import (
    DEEPGRAM_API_KEY, GROQ_API_KEY,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    GROQ_MODEL, SERVER_PORT, PROACTIVE_INTERVAL,
)
from mood_machine import AnnoyanceState, classify_user_input, detect_expression
from llm_client import LLMClient
from tts_client import TTSClient
from stt_client import STTClient
from ws_server import (
    app, broadcast, set_audio_receive_callback, set_personality_callback,
    send_audio_to_browser, send_audio_text,
)
import uvicorn

annoyance = AnnoyanceState()
llm_client = LLMClient(api_key=GROQ_API_KEY, model=GROQ_MODEL)
tts_client: TTSClient | None = None
stt_client: STTClient | None = None
_speaking = False
_last_activity = time.monotonic()
_executor = ThreadPoolExecutor(max_workers=2)
_unmute_task: asyncio.Task | None = None
SILENCE_RESET_SECONDS = 180  # after this much silence, annoyance returns to 0

# Proactive phrases indexed by annoyance level (0=friendly … 3=enraged)
PROACTIVE_PHRASES: dict[int, list[str]] = {
    0: ["Hola... ¿hay alguien por ahí?", "¿Me escuchas?", "Estoy aquí, ¿sabes?"],
    1: ["¿Sigues ahí, o te aburriste ya?", "Llevas un rato callado.",
        "¿Pensabas que me había ido?"],
    2: ["¿Eh, quién anda ahí?", "Te veo. Sé que sigues ahí.",
        "Habla, joder, que llevas un rato muy callado."],
    3: ["¿Qué cojones, ya te fuiste? Cobarde.", "Vuelve aquí, gilipollas.",
        "No me dejes hablando solo, hostia."],
}


async def on_amplitude(value: float) -> None:
    await broadcast({"type": "amplitude", "value": round(value, 3)})


async def on_viseme_schedule(events: list) -> None:
    """Send viseme schedule via the AUDIO WebSocket so it stays ordered with audio chunks."""
    await send_audio_text(json.dumps({"type": "viseme_schedule", "events": events}))


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


async def on_personality(personality_id: str) -> None:
    """Browser chose a personality on the setup screen. Switch profile cleanly."""
    annoyance.set_personality(personality_id)
    llm_client.reset_history()
    if tts_client is not None:
        tts_client.set_mood(annoyance.mood_id)
    print(f"[PERSONALITY] → {annoyance.personality_id} "
          f"(escalates={annoyance.escalates}, mood={annoyance.mood_id})", flush=True)
    # Push the resting expression so the face reflects the new tone immediately
    await broadcast({"type": "expression", "shapes": annoyance.get_expression()})


async def on_transcript(text: str) -> None:
    global _last_activity
    if _speaking:
        return
    _last_activity = time.monotonic()
    print(f"[TRANSCRIPT] {text!r}")
    await broadcast({"type": "text", "value": text})

    # Classify input and update annoyance state BEFORE building the LLM prompt
    delta, reason = classify_user_input(text)
    level_change = annoyance.apply(delta, reason)
    print(f"[ANNOY] {reason} → Δ{delta:+d}, points={annoyance.points}, level={annoyance.level}", flush=True)

    if level_change != 0:
        # Avoid tonal incoherence: drop history so the LLM doesn't try to stay in old tone
        llm_client.reset_history()
        tts_client.set_mood(annoyance.mood_id)
        print(f"[ANNOY] level changed by {level_change:+d} → mood={annoyance.mood_id}", flush=True)

    system_prompt = annoyance.get_prompt()

    loop = asyncio.get_running_loop()

    def _stream_and_feed():
        try:
            tokens = []
            for token in llm_client.stream(text, system_prompt):
                tokens.append(token)
                tts_client.feed(token)
            full_response = ''.join(tokens)
            print(f"[LLM] response: {full_response!r}")
            tts_client.flush()
            # Level baseline + per-response fine-tuning. detect_expression returns
            # an empty dict for "normal" responses so the baseline shows through.
            expression = {**annoyance.get_expression(), **detect_expression(full_response)}
            asyncio.run_coroutine_threadsafe(
                broadcast({"type": "expression", "shapes": expression}),
                loop,
            )
        except Exception as e:
            print(f"[LLM/TTS ERROR] {e!r}")
            import traceback; traceback.print_exc()

    await loop.run_in_executor(_executor, _stream_and_feed)


async def proactive_loop() -> None:
    """Speak proactively after PROACTIVE_INTERVAL seconds of silence."""
    global _last_activity
    await asyncio.sleep(PROACTIVE_INTERVAL)  # initial delay
    while True:
        await asyncio.sleep(10)  # check every 10 seconds
        if _speaking:
            continue
        if time.monotonic() - _last_activity >= PROACTIVE_INTERVAL:
            phrases = PROACTIVE_PHRASES.get(annoyance.level, PROACTIVE_PHRASES[0])
            phrase = random.choice(phrases)
            print(f"[PROACTIVE] level={annoyance.level} {phrase!r}", flush=True)
            loop = asyncio.get_running_loop()

            def _speak():
                tts_client.feed(phrase)
                tts_client.flush()

            await loop.run_in_executor(_executor, _speak)
            _last_activity = time.monotonic()


async def silence_reset_loop() -> None:
    """Reset annoyance to 0 after a long silence — assume the previous visitor left."""
    while True:
        await asyncio.sleep(30)
        if annoyance.points > 0 and time.monotonic() - _last_activity > SILENCE_RESET_SECONDS:
            old_level = annoyance.level
            annoyance.reset()
            llm_client.reset_history()
            if tts_client is not None:
                tts_client.set_mood(annoyance.mood_id)
            print(f"[ANNOY] reset by silence (was level={old_level})", flush=True)


async def run_pipeline() -> None:
    global tts_client, stt_client
    loop = asyncio.get_running_loop()

    tts_client = TTSClient(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        on_amplitude=on_amplitude,
        on_speaking=on_speaking,
        on_viseme_schedule=on_viseme_schedule,
        on_audio_chunk=send_audio_to_browser,
        loop=loop,
    )
    tts_client.set_mood(annoyance.mood_id)

    stt_client = STTClient(
        api_key=DEEPGRAM_API_KEY,
        on_transcript=on_transcript,
    )

    set_audio_receive_callback(stt_client.receive_audio)
    set_personality_callback(on_personality)

    await stt_client.start()
    print(f"[ENTITY] Listening on port {SERVER_PORT}. Open http://<this-ip>:{SERVER_PORT}/face on browser.")
    await asyncio.gather(
        proactive_loop(),
        silence_reset_loop(),
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
