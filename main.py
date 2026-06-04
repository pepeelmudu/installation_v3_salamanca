import asyncio
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from config import (
    DEEPGRAM_API_KEY, GROQ_API_KEY,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    GROQ_MODEL, GROQ_FALLBACK_MODELS, SERVER_PORT, PROACTIVE_INTERVAL,
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
import expo_glitch as glitch
from expo_glitch import GlitchBuffer, robotify, split_sentences

# Windows consoles default to cp1252, which makes print() crash on non-latin
# chars (→, Δ, emojis from glitch lines). On a crash inside on_personality this
# kills the /audio WebSocket handler and the mic never reaches STT. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

annoyance = AnnoyanceState()
glitch_buffer = GlitchBuffer(min_size=2)
llm_client = LLMClient(api_key=GROQ_API_KEY, model=GROQ_MODEL, fallback_models=GROQ_FALLBACK_MODELS)
tts_client: TTSClient | None = None
stt_client: STTClient | None = None
_speaking = False
_last_activity = time.monotonic()
_executor = ThreadPoolExecutor(max_workers=3)  # 3: reply path + glitch refill shouldn't starve each other
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
        # Watchdog: never leave the mic muted longer than this, even if a
        # speaking=False event is ever missed (would otherwise wedge the mic).
        async def _force_unmute():
            await asyncio.sleep(25)
            stt_client.set_muted(False)
            print("[STT] watchdog force-unmute", flush=True)
        _unmute_task = asyncio.create_task(_force_unmute())
    else:
        if _unmute_task and not _unmute_task.done():
            _unmute_task.cancel()
        async def _delayed_unmute():
            await asyncio.sleep(0.1)
            stt_client.set_muted(False)
        _unmute_task = asyncio.create_task(_delayed_unmute())


async def on_personality(personality_id: str) -> None:
    """Browser chose a personality on the setup screen. Switch profile cleanly."""
    annoyance.set_personality(personality_id)
    if stt_client is not None:
        stt_client.set_language("es")  # all personalities listen in Spanish
    llm_client.reset_history()
    if tts_client is not None:
        tts_client.set_mood(annoyance.mood_id)
    print(f"[PERSONALITY] → {annoyance.personality_id} "
          f"(escalates={annoyance.escalates}, mood={annoyance.mood_id})", flush=True)
    # Push the resting expression so the face reflects the new tone immediately
    await broadcast({"type": "expression", "shapes": annoyance.get_expression()})


async def _handle_expo_turn(text: str) -> None:
    loop = asyncio.get_running_loop()

    # ~1/3 of the time: ignore the question, deflect instead.
    if random.random() < glitch.DEFLECT_PROB:
        line = glitch_buffer.pop("deflection")
        if line:
            text_out, mood = glitch.style_for_category(line, "deflection")
            print(f"[GLITCH] deflection: {text_out!r}", flush=True)
            await broadcast({"type": "caption", "value": line})
            await loop.run_in_executor(
                _executor, lambda t=text_out, m=mood: tts_client.say_special(t, mood=m, flush=True))
            return
        # buffer empty → fall through to a normal reply

    # ~1 in 5 replies: a flicker of humanity — warm, romantic, NO glitch/insults.
    romantic = random.random() < glitch.ROMANTIC_PROB
    system_prompt = glitch.ROMANTIC_PROMPT if romantic else annoyance.get_prompt()
    temperature = 1.0 if romantic else 1.3
    inject = (not romantic) and (random.random() < glitch.INJECT_PROB)
    injection = glitch_buffer.pop("injection") if inject else None

    def _stream_and_feed():
        try:
            full = "".join(llm_client.stream(text, system_prompt, temperature=temperature, max_tokens=40))
            print(f"[LLM] expo response ({'romantic' if romantic else 'chaotic'}): {full!r}")
            asyncio.run_coroutine_threadsafe(
                broadcast({"type": "caption", "value": full}), loop)
            sentences = split_sentences(full)
            for i, s in enumerate(sentences):
                piece = s if romantic else robotify(s, prob=glitch.ROBOT_PROB)
                tts_client.feed(piece + " ")
                if i == 0 and injection:
                    tts_client.flush_buffer()   # ensure sentence 0 is queued before the blurt
                    inj_text, inj_mood = glitch.style_for_category(injection, "injection")
                    print(f"[GLITCH] injection: {inj_text!r}", flush=True)
                    tts_client.say_special(inj_text, mood=inj_mood, flush=False)
            tts_client.flush()
            asyncio.run_coroutine_threadsafe(
                broadcast({"type": "expression", "shapes": annoyance.get_expression()}),
                loop,
            )
        except Exception as e:
            print(f"[EXPO/LLM ERROR] {e!r}")
            import traceback; traceback.print_exc()

    await loop.run_in_executor(_executor, _stream_and_feed)


async def on_transcript(text: str) -> None:
    global _last_activity
    if _speaking:
        return
    _last_activity = time.monotonic()
    print(f"[TRANSCRIPT] {text!r}")
    await broadcast({"type": "text", "value": text})

    if annoyance.personality_id == "expo":
        await _handle_expo_turn(text)
        return

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
            asyncio.run_coroutine_threadsafe(
                broadcast({"type": "caption", "value": full_response}), loop)
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
    """Neutral: nag after PROACTIVE_INTERVAL of silence. Expo: shout every
    EXPO_PROACTIVE_INTERVAL of quiet, unless currently speaking."""
    global _last_activity
    await asyncio.sleep(10)
    while True:
        await asyncio.sleep(5)
        if _speaking or tts_client is None:
            continue
        now = time.monotonic()

        if annoyance.personality_id == "expo":
            if now - _last_activity >= glitch.EXPO_PROACTIVE_INTERVAL:
                line = glitch_buffer.pop("outburst")
                if line:
                    text_out, mood = glitch.style_for_category(line, "outburst")
                    print(f"[GLITCH] outburst: {text_out!r}", flush=True)
                    await broadcast({"type": "caption", "value": line})
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        _executor, lambda t=text_out, m=mood: tts_client.say_special(t, mood=m, flush=True))
                _last_activity = now   # snooze until next interval even if buffer was empty
            continue

        # ── neutral (unchanged) ──
        if now - _last_activity >= PROACTIVE_INTERVAL:
            phrases = PROACTIVE_PHRASES.get(annoyance.level, PROACTIVE_PHRASES[0])
            phrase = random.choice(phrases)
            print(f"[PROACTIVE] level={annoyance.level} {phrase!r}", flush=True)
            loop = asyncio.get_running_loop()

            def _speak():
                tts_client.feed(phrase)
                tts_client.flush()

            await loop.run_in_executor(_executor, _speak)
            _last_activity = time.monotonic()


async def glitch_refill_loop() -> None:
    """Keep the expo phrase buffer topped up with fresh LLM lines."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(12)
        if annoyance.personality_id != "expo":
            continue
        for category in glitch_buffer.low_categories():
            prompt = glitch.GLITCH_PROMPTS[category]

            def _gen(p=prompt):
                return llm_client.generate_oneshot(glitch.GLITCH_SYSTEM, p)

            try:
                line = await loop.run_in_executor(_executor, _gen)
            except Exception as e:
                print(f"[GLITCH] gen failed: {e!r}", flush=True)
                line = ""
            if line:
                glitch_buffer.add(category, robotify(line, prob=glitch.ROBOT_PROB))


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
        glitch_refill_loop(),
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
