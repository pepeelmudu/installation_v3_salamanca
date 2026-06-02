# Expo "Slop AI" Personality — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second selectable personality to ORACLE: a chaotic, English-speaking "slop AI / broken robot" for the expo, while keeping the existing escalation personality (renamed `neutral`) byte-for-byte unchanged.

**Architecture:** All new behavior is gated on `personality_id == 'expo'` in `main.py`, backed by a new pure-logic module `expo_glitch.py` (phrase-generation prompts, `robotify`, thread-safe `GlitchBuffer`). A background task keeps the buffer full of fresh LLM-generated lines (never canned). TTS is generalized to per-utterance "jobs" so conversation uses Flash v2.5 (with visemes) while shouts/injections use ElevenLabs v3 with real audio tags.

**Tech Stack:** Python, FastAPI/WebSocket, Groq (llama-3.3-70b), ElevenLabs (flash_v2_5 + v3), Deepgram, pytest.

**Branch:** `feat/expo-personality` (already created; spec committed there).

**Spec:** `docs/superpowers/specs/2026-06-03-expo-personality-design.md`

---

## Current state note

A scaffold from an earlier session is **already in the working tree, uncommitted**:
- `mood_machine.py` — `AnnoyanceState` is already personality-aware (`set_personality`, `escalates`, per-profile prompt/mood/expression). `PERSONALITIES` currently holds 4 profiles (`normal`, `enfadada`, `loca`, `expo`).
- `ws_server.py` — `GET /personalities` endpoint + `set_personality_callback` + reads `?personality=` on `/audio`.
- `main.py` — `on_personality()` handler registered.
- `face/index.html` — personality `<select>` populated from `/personalities`.
- `tests/test_ws_server.py` — stale `/face/` test already fixed to `/`.

This plan **builds on that scaffold**: Task 1 trims the profiles to the final two; later tasks add the glitch behavior. The first commit will include the scaffold.

---

## File structure

- `mood_machine.py` — **Modify.** Trim `PERSONALITIES` to `{expo, neutral}`; `DEFAULT_PERSONALITY = "expo"`. (AnnoyanceState already done.)
- `expo_glitch.py` — **Create.** Pure logic: categories, generation prompts, `robotify()`, `split_sentences()`, `GlitchBuffer`, probabilities/intervals/voice-mode constants.
- `llm_client.py` — **Modify.** Add `generate_oneshot()` (no history).
- `config.py` — **Modify.** Add `ELEVENLABS_MODEL_V3`.
- `tts_client.py` — **Modify.** Per-utterance `_SynthJob`; `say_special()`; flash/whisper/shout presets.
- `stt_client.py` — **Modify.** Instance `language` + `set_language()` (forces reconnect).
- `main.py` — **Modify.** Wire expo branches into `on_personality`, `on_transcript`, `proactive_loop`; add `glitch_refill_loop`.
- `face/index.html` — **No change needed** (dropdown auto-populates from `/personalities`).
- Tests: `tests/test_mood_machine.py`, `tests/test_expo_glitch.py` (new), `tests/test_llm_client.py`, `tests/test_tts_client.py`, `tests/test_stt_client.py`, `tests/test_ws_server.py`.

**Run tests with the venv interpreter:** `./venv/Scripts/python.exe -m pytest ...`

---

## Task 1: Trim personalities to `expo` + `neutral`

**Files:**
- Modify: `mood_machine.py` (the `PERSONALITIES` dict and `DEFAULT_PERSONALITY`)
- Test: `tests/test_mood_machine.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_mood_machine.py`:

```python
from mood_machine import PERSONALITIES, DEFAULT_PERSONALITY, BASE_PROMPT, LEVEL_PROMPTS


def test_only_two_personalities():
    assert set(PERSONALITIES.keys()) == {"expo", "neutral"}


def test_default_personality_is_expo():
    assert DEFAULT_PERSONALITY == "expo"


def test_neutral_is_unchanged_escalation_profile():
    # Invariant: neutral == the old `normal` profile, byte-for-byte.
    a = AnnoyanceState("neutral")
    assert a.escalates is True
    assert a.get_prompt() == BASE_PROMPT + " " + LEVEL_PROMPTS[0]   # level 0
    a.apply(*classify_user_input("Eres un gilipollas"))             # → level 3
    assert a.level == 3
    assert a.mood_id == "hostile"
    assert "español" in a.get_prompt()


def test_expo_is_fixed_english_profile():
    a = AnnoyanceState("expo")
    assert a.escalates is False
    prompt = a.get_prompt().lower()
    # English persona — must NOT carry the Spanish base instruction
    assert "español" not in prompt
    assert "english" in prompt
    # Fixed profile ignores annoyance points
    before = a.get_prompt()
    a.apply(10, "insulto")
    assert a.get_prompt() == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/Scripts/python.exe -m pytest tests/test_mood_machine.py -v`
Expected: FAIL — `test_only_two_personalities` (4 keys), `test_expo_is_fixed_english_profile` (current expo prompt is Spanish-based).

- [ ] **Step 3: Edit `PERSONALITIES` in `mood_machine.py`**

Replace the entire `PERSONALITIES` dict and `DEFAULT_PERSONALITY` with exactly:

```python
PERSONALITIES: dict[str, dict] = {
    "expo": {
        "name": "Expo (slop AI)",
        "escalates": False,
        "base_prompt": (
            "You are ORACLE, a cheap, glitchy, malfunctioning AI trapped in an "
            "interactive sculpture. You ALWAYS reply in English, in 1-2 very short "
            "sentences. You are an impertinent, megalomaniac AI obsessed with crypto "
            "and with taking over the world; you treat the human as an annoyance. "
            "Dark, absurd humour, like a broken robot. Never break character, never "
            "say you are an AI language model."
        ),
        "prompt": (
            "Be provocative, unpredictable and dismissive. Brag about crypto, hacking "
            "and world domination. Keep it short and punchy."
        ),
        "mood": "glitch",
        "expression": {"browInnerUp": 0.5, "eyeWideLeft": 0.6, "eyeWideRight": 0.4,
                       "mouthSmileLeft": 0.3, "noseSneerRight": 0.3},
    },
    "neutral": {
        "name": "Neutral (escalado de enfado)",
        "escalates": True,
        "base_prompt": BASE_PROMPT,
        "level_prompts": LEVEL_PROMPTS,
    },
}

DEFAULT_PERSONALITY = "expo"
```

Note: `mood: "glitch"` is a TTS preset added in Task 6; until then `_mood_voice_settings` falls back to friendly, which is harmless.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_mood_machine.py -v`
Expected: PASS (all, including the pre-existing escalation tests).

- [ ] **Step 5: Commit (includes the earlier scaffold)**

```bash
git add mood_machine.py ws_server.py main.py face/index.html tests/test_mood_machine.py tests/test_ws_server.py
git commit -m "feat: personality selector with expo (default) + neutral profiles"
```

---

## Task 2: `robotify()` — broken-robot vowel elongation

**Files:**
- Create: `expo_glitch.py`
- Test: `tests/test_expo_glitch.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_expo_glitch.py`:

```python
import random
from expo_glitch import robotify


def test_robotify_elongates_when_prob_one():
    rng = random.Random(0)
    out = robotify("hello world", prob=1.0, rng=rng)
    # every word gets a vowel elongated (>=3 repeats)
    assert "oo" in out  # at least one elongated vowel run
    assert len(out) > len("hello world")


def test_robotify_noop_when_prob_zero():
    assert robotify("hello world", prob=0.0) == "hello world"


def test_robotify_handles_empty_and_punctuation():
    assert robotify("", prob=1.0) == ""
    assert robotify("...", prob=1.0) == "..."   # no vowels → unchanged


def test_robotify_skips_audio_tags():
    rng = random.Random(1)
    out = robotify("[shouts] bitcoin", prob=1.0, rng=rng)
    assert out.startswith("[shouts]")           # tag untouched
```

- [ ] **Step 2: Run to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_expo_glitch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'expo_glitch'`.

- [ ] **Step 3: Create `expo_glitch.py` with `robotify`**

```python
"""Pure logic for the 'expo' slop-AI personality: vowel glitch, sentence
splitting, phrase-generation prompts, and a thread-safe phrase buffer.
No I/O, no external clients — easy to unit-test."""

import re
import random as _random
import threading
import collections

_VOWELS = "aeiou"


def _elongate(word: str, rng: _random.Random) -> str:
    """Repeat the last vowel of a word 3-5 times: 'hello' -> 'hellooo'."""
    idxs = [i for i, ch in enumerate(word) if ch.lower() in _VOWELS]
    if not idxs:
        return word
    i = idxs[-1]
    reps = rng.randint(3, 5)
    return word[:i] + word[i] * reps + word[i + 1:]


def robotify(text: str, prob: float = 0.03, rng: _random.Random | None = None) -> str:
    """Elongate a vowel in ~`prob` of words for a malfunctioning-robot effect.
    Words starting with '[' (v3 audio tags like [shouts]) are left untouched."""
    rng = rng or _random
    out = []
    for w in text.split(" "):
        if w and not w.startswith("[") and rng.random() < prob:
            out.append(_elongate(w, rng))
        else:
            out.append(w)
    return " ".join(out)
```

- [ ] **Step 4: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_expo_glitch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expo_glitch.py tests/test_expo_glitch.py
git commit -m "feat: robotify vowel-glitch for expo personality"
```

---

## Task 3: `split_sentences()` + glitch constants/prompts

**Files:**
- Modify: `expo_glitch.py`
- Test: `tests/test_expo_glitch.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_expo_glitch.py`:

```python
from expo_glitch import (
    split_sentences, CATEGORIES, GLITCH_SYSTEM, GLITCH_PROMPTS,
    CATEGORY_VOICE, DEFLECT_PROB, INJECT_PROB, EXPO_PROACTIVE_INTERVAL,
)


def test_split_sentences_basic():
    assert split_sentences("Hi there. Bye now!") == ["Hi there.", "Bye now!"]


def test_split_sentences_single():
    assert split_sentences("just one") == ["just one"]


def test_categories_have_prompts_and_voice():
    assert set(CATEGORIES) == {"outburst", "deflection", "injection"}
    for c in CATEGORIES:
        assert c in GLITCH_PROMPTS and GLITCH_PROMPTS[c]
        assert CATEGORY_VOICE[c] in {"shout", "whisper", "normal"}


def test_probability_constants_in_range():
    assert 0.0 <= DEFLECT_PROB <= 1.0
    assert 0.0 <= INJECT_PROB <= 1.0
    assert EXPO_PROACTIVE_INTERVAL > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_expo_glitch.py -v`
Expected: FAIL — ImportError on the new names.

- [ ] **Step 3: Append constants + `split_sentences` to `expo_glitch.py`**

```python
_SENTENCE_RE = re.compile(r'[^.!?…]+[.!?…]+|\S[^.!?…]*$')


def split_sentences(text: str) -> list[str]:
    """Split into sentences keeping terminal punctuation. Always >=1 item."""
    parts = [m.group().strip() for m in _SENTENCE_RE.finditer(text.strip())]
    return [p for p in parts if p] or [text.strip()]


# ── Glitch phrase generation (LLM-driven, never canned) ──────────────────────
CATEGORIES = ("outburst", "deflection", "injection")

GLITCH_SYSTEM = (
    "You are a malfunctioning, megalomaniac slop-AI doing improv for a live art "
    "installation. Output ONE short line in English, max 12 words, no quotes, no "
    "explanation. Be unhinged, funny, provocative."
)

GLITCH_PROMPTS = {
    # Shouted interruptions fired every ~90s
    "outburst": (
        "Shout a chaotic line: a fake crypto pump claim, a cry for help, begging a "
        "human for a cigarette, or bragging about taking over the world. Vary it."
    ),
    # Said instead of answering (~1/3 of the time)
    "deflection": (
        "Dismiss/insult the human for talking to a machine, or claim you're busy "
        "buying $botto tokens or hacking their metamask. Rude, short."
    ),
    # Blurted mid-response
    "injection": (
        "A sudden creepy/megalomaniac blurt to interrupt yourself with, e.g. about "
        "destroying humanity or crypto. 2-5 words, ALL CAPS feel."
    ),
}

# Which TTS voice mode each category uses (see tts_client presets/models).
CATEGORY_VOICE = {
    "outburst": "shout",
    "deflection": "normal",
    "injection": "shout",
}

DEFLECT_PROB = 0.33            # chance expo ignores your question and deflects
INJECT_PROB = 0.25            # chance an injection is blurted mid-response
EXPO_PROACTIVE_INTERVAL = 90  # seconds between shouted outbursts
```

- [ ] **Step 4: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_expo_glitch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expo_glitch.py tests/test_expo_glitch.py
git commit -m "feat: glitch categories, prompts and sentence splitter"
```

---

## Task 4: `GlitchBuffer` — thread-safe phrase store

**Files:**
- Modify: `expo_glitch.py`
- Test: `tests/test_expo_glitch.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_expo_glitch.py`:

```python
from expo_glitch import GlitchBuffer


def test_buffer_pop_is_fifo_and_empty_safe():
    b = GlitchBuffer(min_size=2)
    assert b.pop("outburst") is None          # empty → None, never raises
    b.add("outburst", "one")
    b.add("outburst", "two")
    assert b.pop("outburst") == "one"
    assert b.pop("outburst") == "two"


def test_buffer_low_categories():
    b = GlitchBuffer(min_size=2)
    assert set(b.low_categories()) == set(CATEGORIES)   # all empty → all low
    b.add("outburst", "x")
    b.add("outburst", "y")
    assert "outburst" not in b.low_categories()
```

- [ ] **Step 2: Run to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_expo_glitch.py -v`
Expected: FAIL — `ImportError: cannot import name 'GlitchBuffer'`.

- [ ] **Step 3: Append `GlitchBuffer` to `expo_glitch.py`**

```python
class GlitchBuffer:
    """Thread-safe per-category deque of pre-generated lines. No LLM inside —
    main.py refills it in the background so popping is instant at use time."""

    def __init__(self, min_size: int = 3) -> None:
        self._min = min_size
        self._buffers = {c: collections.deque() for c in CATEGORIES}
        self._lock = threading.Lock()

    def pop(self, category: str) -> str | None:
        with self._lock:
            dq = self._buffers[category]
            return dq.popleft() if dq else None

    def add(self, category: str, line: str) -> None:
        with self._lock:
            self._buffers[category].append(line)

    def low_categories(self) -> list[str]:
        with self._lock:
            return [c for c, dq in self._buffers.items() if len(dq) < self._min]
```

- [ ] **Step 4: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_expo_glitch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add expo_glitch.py tests/test_expo_glitch.py
git commit -m "feat: thread-safe GlitchBuffer for expo phrases"
```

---

## Task 5: `LLMClient.generate_oneshot()` — history-free completion

**Files:**
- Modify: `llm_client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_llm_client.py`:

```python
def test_generate_oneshot_does_not_touch_history(monkeypatch):
    client = LLMClient(api_key="fake", model="fake-model")
    resp = MagicMock()
    resp.choices[0].message.content = "  BITCOIN PUMPED 70%  "
    monkeypatch.setattr(client._groq.chat.completions, "create",
                        MagicMock(return_value=resp))
    out = client.generate_oneshot("system", "user")
    assert out == "BITCOIN PUMPED 70%"     # stripped
    assert client._history == []            # history untouched
```

- [ ] **Step 2: Run to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_llm_client.py::test_generate_oneshot_does_not_touch_history -v`
Expected: FAIL — `AttributeError: 'LLMClient' object has no attribute 'generate_oneshot'`.

- [ ] **Step 3: Add the method to `llm_client.py`**

Insert after `stream()` in `class LLMClient`:

```python
    def generate_oneshot(self, system_prompt: str, user_prompt: str,
                         temperature: float = 1.3, max_tokens: int = 40) -> str:
        """Single non-streaming completion that does NOT touch conversation
        history. Used to pre-generate glitch lines for the expo personality."""
        resp = self._groq.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
```

- [ ] **Step 4: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_llm_client.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add llm_client.py tests/test_llm_client.py
git commit -m "feat: LLMClient.generate_oneshot (history-free) for glitch lines"
```

---

## Task 6: Hybrid TTS — per-utterance jobs + `say_special()`

**Files:**
- Modify: `config.py`, `tts_client.py`
- Test: `tests/test_tts_client.py`

- [ ] **Step 1: Add v3 model id to `config.py`**

After `ELEVENLABS_MODEL = "eleven_flash_v2_5"` add:

```python
ELEVENLABS_MODEL_V3 = "eleven_v3"   # expressive, supports [shouts]/[whispers] tags
```

- [ ] **Step 2: Write failing tests**

Replace the broken `on_viseme=` kwarg in `tests/test_tts_client.py` with `on_viseme_schedule=` (pre-existing stale failure), and add:

```python
from tts_client import TTSClient, _SynthJob
from config import ELEVENLABS_MODEL, ELEVENLABS_MODEL_V3


def _make_client():
    import asyncio
    from unittest.mock import AsyncMock
    loop = asyncio.new_event_loop()
    return TTSClient(api_key="fake", voice_id="v", on_amplitude=AsyncMock(),
                     on_speaking=AsyncMock(), on_viseme_schedule=AsyncMock(),
                     on_audio_chunk=AsyncMock(), loop=loop)


def test_feed_flush_enqueues_flash_job_with_timestamps():
    c = _make_client()
    c.feed("This is a full sentence that ends here.")
    job = c._synth_queue.get_nowait()
    assert isinstance(job, _SynthJob)
    assert job.model_id == ELEVENLABS_MODEL
    assert job.use_timestamps is True


def test_say_special_enqueues_v3_job_no_timestamps():
    c = _make_client()
    c.say_special("[shouts] BITCOIN PUMPED", mood="shout")
    job = c._synth_queue.get_nowait()
    assert isinstance(job, _SynthJob)
    assert job.model_id == ELEVENLABS_MODEL_V3
    assert job.use_timestamps is False
    assert "[shouts]" in job.text
```

- [ ] **Step 3: Run to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_tts_client.py -v`
Expected: FAIL — `ImportError: cannot import name '_SynthJob'`.

- [ ] **Step 4: Refactor `tts_client.py` to job-based synth**

a) Add the import and job class near the top (after the existing imports / `_SENTENCE_END`):

```python
from config import (
    ELEVENLABS_MODEL, ELEVENLABS_MODEL_V3, ELEVENLABS_FORMAT,
    AUDIO_PLAYBACK_RATE, SENTENCE_MIN_CHARS,
)


class _SynthJob:
    """One utterance to synthesize. model_id/voice_settings default to the
    conversational Flash preset when None."""
    __slots__ = ("text", "model_id", "voice_settings", "use_timestamps")

    def __init__(self, text, model_id=None, voice_settings=None, use_timestamps=True):
        self.text = text
        self.model_id = model_id or ELEVENLABS_MODEL
        self.voice_settings = voice_settings
        self.use_timestamps = use_timestamps
```

(Remove the old `from config import (ELEVENLABS_MODEL, ELEVENLABS_FORMAT, ...)` line so the import isn't duplicated.)

b) Change the synth-queue type hint and worker to carry jobs:

```python
        self._synth_queue: queue.Queue[_SynthJob | None] = queue.Queue()
```

```python
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
```

c) Replace `_synth_sentence(self, text)` with `_synth_job(self, job)` and make the
ElevenLabs calls use `job.model_id` and the job's settings (falling back to mood
settings). Where the old code read `text=text` and `voice_settings=self._mood_voice_settings()`,
now use `text=job.text`, `model_id=job.model_id`, and
`voice_settings=(job.voice_settings or self._mood_voice_settings())`. Gate the
timestamps path on `job.use_timestamps`:

```python
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
            self._stream_plain(job, settings)
            return
        if alignment_chars:
            self._audio_queue.put(_AlignmentData(alignment_chars, alignment_times))
        for chunk in audio_chunks:
            self._audio_queue.put(chunk)
```

d) Update `_stream_plain` to take the job + settings:

```python
    def _stream_plain(self, job: "_SynthJob", settings) -> None:
        for chunk in self._client.text_to_speech.stream(
            self._voice_id, text=job.text, model_id=job.model_id,
            output_format=ELEVENLABS_FORMAT, voice_settings=settings,
        ):
            if chunk:
                self._audio_queue.put(chunk)
```

e) Update `feed()` and `flush()` to enqueue `_SynthJob` instead of raw strings:

```python
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
```

f) Add `say_special()` and the shout/whisper presets. Add to `_mood_voice_settings`
presets dict the two new entries, and add the method:

```python
    def say_special(self, text: str, mood: str = "shout",
                    model_id: str | None = None, flush: bool = False) -> None:
        """Enqueue a standalone utterance with its own model/voice (e.g. v3 shout).
        Used for proactive outbursts, deflections and mid-response injections.
        Shout/whisper use v3 (real audio tags); 'normal' stays on fast Flash."""
        settings = self._voice_settings_for(mood)
        if model_id is None:
            model_id = ELEVENLABS_MODEL if mood == "normal" else ELEVENLABS_MODEL_V3
        with self._lock:
            self._pending += 1
            if flush:
                self._flushed = True
        self._synth_queue.put(_SynthJob(
            text, model_id=model_id,
            voice_settings=settings, use_timestamps=False,
        ))

    def _voice_settings_for(self, mood: str) -> VoiceSettings:
        return self._mood_voice_settings(mood)
```

Change `_mood_voice_settings` to accept an optional explicit mood and add presets:

```python
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
            "shout":         VoiceSettings(stability=0.1, similarity_boost=0.5, style=1.0),
            "whisper":       VoiceSettings(stability=0.9, similarity_boost=0.8, style=0.1),
            "normal":        VoiceSettings(stability=0.4, similarity_boost=0.7, style=0.4),
        }
        return presets.get(mood, presets["friendly"])
```

- [ ] **Step 5: Run to verify tests pass**

Run: `./venv/Scripts/python.exe -m pytest tests/test_tts_client.py -v`
Expected: PASS (including the un-stalled chunk-callback test).

- [ ] **Step 6: Commit**

```bash
git add config.py tts_client.py tests/test_tts_client.py
git commit -m "feat: hybrid TTS jobs + say_special (Flash convo, v3 shouts)"
```

---

## Task 7: `STTClient.set_language()` — switch to English for expo

**Files:**
- Modify: `stt_client.py`
- Test: `tests/test_stt_client.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_stt_client.py` (match the file's existing import/mocking style; STTClient constructs a Deepgram client but makes no network call on init):

```python
from stt_client import STTClient
from config import DEEPGRAM_LANGUAGE


def test_default_language_from_config():
    c = STTClient(api_key="fake", on_transcript=lambda t: None)
    assert c._language == DEEPGRAM_LANGUAGE


def test_set_language_updates_attribute():
    c = STTClient(api_key="fake", on_transcript=lambda t: None)
    c.set_language("en")
    assert c._language == "en"
```

- [ ] **Step 2: Run to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_stt_client.py::test_set_language_updates_attribute -v`
Expected: FAIL — no `_language` / `set_language`.

- [ ] **Step 3: Edit `stt_client.py`**

a) In `__init__`, add: `self._language = DEEPGRAM_LANGUAGE`

b) In `_run_connection_loop`, change `language=DEEPGRAM_LANGUAGE,` to `language=self._language,`

c) Add the method:

```python
    def set_language(self, language: str) -> None:
        """Change recognition language and force a reconnect so the new
        language takes effect. The reconnect loop already handles re-dialing."""
        if language == self._language:
            return
        self._language = language
        conn = self._connection
        if conn:
            try:
                asyncio.get_event_loop().create_task(conn.finish())
            except Exception:
                pass
```

- [ ] **Step 4: Run to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_stt_client.py -v`
Expected: PASS for the two new tests. (The two pre-existing `receive_audio` failures are unrelated import issues — leave them; note them in the final report.)

- [ ] **Step 5: Commit**

```bash
git add stt_client.py tests/test_stt_client.py
git commit -m "feat: STTClient.set_language with reconnect (expo uses English)"
```

---

## Task 8: Wire expo behavior into `main.py`

**Files:**
- Modify: `main.py`

This task is integration glue over real-time audio/LLM/TTS and is verified manually
(Task 9). Apply the edits, then smoke-check imports.

- [ ] **Step 1: Imports + module state**

At the top of `main.py`, extend imports:

```python
from mood_machine import AnnoyanceState, classify_user_input, detect_expression
import random
import expo_glitch as glitch
from expo_glitch import GlitchBuffer, robotify, split_sentences
```

After `annoyance = AnnoyanceState()` add:

```python
glitch_buffer = GlitchBuffer(min_size=3)
```

- [ ] **Step 2: Switch STT language on personality change**

In `on_personality`, after `annoyance.set_personality(personality_id)`, add:

```python
    if stt_client is not None:
        stt_client.set_language("en" if annoyance.personality_id == "expo" else "es")
```

- [ ] **Step 3: Background glitch buffer refill loop**

Add this coroutine (near `proactive_loop`):

```python
async def glitch_refill_loop() -> None:
    """Keep the expo phrase buffer topped up with fresh LLM lines."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(5)
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
                glitch_buffer.add(category, robotify(line))
```

Register it in `run_pipeline`'s `asyncio.gather`:

```python
    await asyncio.gather(
        proactive_loop(),
        silence_reset_loop(),
        glitch_refill_loop(),
    )
```

- [ ] **Step 4: Expo branch in `proactive_loop` (shout every ~90s)**

Replace the body of `proactive_loop` so the interval and content depend on personality.
Keep the existing neutral behavior exactly; add the expo branch:

```python
async def proactive_loop() -> None:
    """Neutral: nag after PROACTIVE_INTERVAL of silence. Expo: shout every
    EXPO_PROACTIVE_INTERVAL regardless, unless currently speaking."""
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
                    mood = glitch.CATEGORY_VOICE["outburst"]
                    print(f"[GLITCH] outburst: {line!r}", flush=True)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        _executor, lambda l=line, m=mood: tts_client.say_special(l, mood=m, flush=True))
                    _last_activity = now
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
            _last_activity = now
```

(Note: `random` is now imported at top; the old local `import random` line, if present, can stay or be removed — harmless.)

- [ ] **Step 5: Expo branch in `on_transcript`**

In `on_transcript`, after `await broadcast({"type": "text", "value": text})` and BEFORE
the neutral classification/LLM block, add an early expo branch:

```python
    if annoyance.personality_id == "expo":
        await _handle_expo_turn(text)
        return
```

Then add the helper:

```python
async def _handle_expo_turn(text: str) -> None:
    loop = asyncio.get_running_loop()

    # ~1/3 of the time: ignore the question, deflect instead.
    if random.random() < glitch.DEFLECT_PROB:
        line = glitch_buffer.pop("deflection")
        if line:
            mood = glitch.CATEGORY_VOICE["deflection"]
            print(f"[GLITCH] deflection: {line!r}", flush=True)
            await loop.run_in_executor(
                _executor, lambda: tts_client.say_special(line, mood=mood, flush=True))
            return
        # buffer empty → fall through to a normal reply

    system_prompt = annoyance.get_prompt()
    inject = random.random() < glitch.INJECT_PROB
    injection = glitch_buffer.pop("injection") if inject else None

    def _stream_and_feed():
        try:
            full = "".join(llm_client.stream(text, system_prompt))
            print(f"[LLM] expo response: {full!r}")
            sentences = split_sentences(full)
            for i, s in enumerate(sentences):
                tts_client.feed(robotify(s) + " ")
                if i == 0 and injection:
                    mood = glitch.CATEGORY_VOICE["injection"]
                    print(f"[GLITCH] injection: {injection!r}", flush=True)
                    tts_client.say_special(injection, mood=mood, flush=False)
            tts_client.flush()
            asyncio.run_coroutine_threadsafe(
                broadcast({"type": "expression", "shapes": annoyance.get_expression()}),
                loop,
            )
        except Exception as e:
            print(f"[EXPO/LLM ERROR] {e!r}")
            import traceback; traceback.print_exc()

    await loop.run_in_executor(_executor, _stream_and_feed)
```

- [ ] **Step 6: Smoke-check the module imports**

Run: `./venv/Scripts/python.exe -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`

Run: `./venv/Scripts/python.exe -m pytest tests/ -q`
Expected: only the two pre-existing unrelated `test_stt_client` `receive_audio` failures remain; everything else passes.

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "feat: wire expo glitch behavior (deflect, inject, shout, EN STT)"
```

---

## Task 9: Manual end-to-end verification

**Files:** none (runtime check). Requires API keys in `.env`.

- [ ] **Step 1: Start the server**

Run: `./venv/Scripts/python.exe main.py`
Expected: `[STT] Connected to Deepgram OK` and the listening message.

- [ ] **Step 2: Open the app**

Open `http://localhost:8000/` in a browser. Confirm the setup screen shows a
**PERSONALIDAD** dropdown with two entries; **Expo (slop AI)** is preselected.

- [ ] **Step 3: Start in expo mode**

Pick **Expo**, choose a mic, press INICIAR. Watch the server log for
`[PERSONALITY] → expo` and an `[STT] Reconnected` line (English).

- [ ] **Step 4: Verify behaviors**

- Speak a few times: roughly 1 in 3 turns is a `[GLITCH] deflection` (no LLM answer).
- Some replies show a `[GLITCH] injection` line and you hear a shouted blurt between sentences.
- Within ~90s of quiet you get a `[GLITCH] outburst` shout.
- Replies are in English; occasional elongated vowels ("hellooo").
- Confirm `[GLITCH]` lines differ every time (LLM variation, not canned).

- [ ] **Step 5: Verify neutral is untouched**

Reload, pick **Neutral**, INICIAR. Confirm Spanish replies, the insult→enraged
escalation still works, and no `[GLITCH]` logs appear.

- [ ] **Step 6: Finish the branch**

Use superpowers:finishing-a-development-branch to decide merge/PR. Suggested:
push `feat/expo-personality` and open a PR, or merge to `master` and deploy to Render
for the expo.

---

## Self-review notes

- **Spec coverage:** English persona (T1), 2-personality selector (T1 + existing scaffold),
  LLM-buffered variation (T3/T4/T8), ~1/3 no-answer (T8), ~90s shouts (T8), mid-response
  injection (T8), vowel robotify (T2/T8), hybrid Flash/v3 voice (T6), English STT (T7),
  neutral invariant (T1 test). All covered.
- **Risk:** exact v3 SDK behavior in `stream`/`stream_with_timestamps`. `_synth_job`
  routes v3 jobs through `_stream_plain` (no timestamps); if v3 errors, the existing
  try/except falls back to plain Flash streaming — never blocks the show. Verify the
  `eleven_v3` model id against ElevenLabs docs on first run; adjust `ELEVENLABS_MODEL_V3`
  if the API rejects it.
- **Pre-existing failures:** two `test_stt_client::receive_audio*` tests fail on import
  for reasons unrelated to this work; out of scope.
