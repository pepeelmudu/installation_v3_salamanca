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
