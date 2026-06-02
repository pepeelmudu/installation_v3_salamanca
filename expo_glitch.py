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
