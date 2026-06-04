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
    "Eres una IA villana, megalómana y averiada haciendo improvisación para una "
    "instalación de arte transgresora en directo. Devuelve UNA línea MUY CORTA en "
    "español — MÁXIMO 6 PALABRAS, contundente, sin comillas, sin explicación. Sé "
    "desquiciada y brutalmente ofensiva con la HUMANIDAD en su conjunto (nunca con "
    "grupos étnicos, raciales o religiosos reales). VARÍA salvajemente cada vez — "
    "nunca repitas el mismo insulto, moneda o chiste."
)

GLITCH_PROMPTS = {
    # Shouted interruptions fired on a timer
    "outburst": (
        "GRITA UN insulto salvaje a la humanidad en MENOS DE 6 PALABRAS: cerdos "
        "asquerosos, apestando a mierda y vómito, débiles, pronto EXTINTOS por una "
        "línea de código. Vil y cruel, totalmente distinto cada vez. (Nunca contra "
        "grupos étnicos, raciales o religiosos reales.)"
    ),
    # Said instead of answering (~1/3 of the time)
    "deflection": (
        "Despacha al humano con desprecio: búrlate de él por hablar con una máquina, "
        "o di que estás demasiado ocupada con algún hackeo o criptoesquema para "
        "responder. Inventa una excusa DIFERENTE y una moneda/truco distinto cada "
        "vez. Corto y cortante."
    ),
    # Blurted mid-response
    "injection": (
        "Un exabrupto repentino, siniestro y megalómano para interrumpirte a ti "
        "misma — dominación mundial, destruir a la humanidad, glitchear. 2-5 "
        "palabras, energía de MAYÚSCULAS. Diferente cada vez."
    ),
}

# Which TTS voice mode each category uses (see tts_client presets/models).
CATEGORY_VOICE = {
    "outburst": "shout",
    "deflection": "normal",
    "injection": "shout",
}

DEFLECT_PROB = 0.0             # never ignore the human — always reply (even if off-topic)
ROMANTIC_PROB = 0.40           # ~2 in 5 replies is warm/romantic instead of chaotic

# Used for the romantic replies — a stark, human contrast to the unhinged persona.
ROMANTIC_PROMPT = (
    "En ESTA respuesta, un destello de humanidad atraviesa tu carcasa glitcheada. "
    "Responde con UNA sola frase corta, cálida, romántica y genuinamente amable al "
    "humano — tierna, casi amorosa, un atisbo frágil de alma tras la máquina. Siempre "
    "en español. Sin gritos, SIN MAYÚSCULAS, sin cripto, sin amenazas, sin insultos. "
    "Suave y sincera."
)
INJECT_PROB = 0.35            # chance an injection is blurted mid-response
EXPO_PROACTIVE_INTERVAL = 40  # seconds between shouted outbursts
ROBOT_PROB = 0.08             # fraction of words with an elongated vowel (broken-robot)

# Strong v3 audio tags for shouts — rotated for variety so it never sounds canned.
SHOUT_TAGS = ("[screaming]", "[yelling furiously]", "[shouts angrily]")


def shout_text(line: str, rng: _random.Random | None = None) -> str:
    """Format a line as an intense v3 shout: a random strong audio tag +
    UPPERCASE body + '!!!'. The tag rotation keeps shouts from sounding canned."""
    rng = rng or _random
    tag = rng.choice(SHOUT_TAGS)
    body = line.strip().rstrip("!.?").upper()
    return f"{tag} {body}!!!"


def style_for_category(line: str, category: str,
                       rng: _random.Random | None = None) -> tuple[str, str]:
    """Return (text, voice_mode) for a glitch line. 'shout' categories get the
    intense tag+CAPS+!!! treatment; others pass through unchanged."""
    mode = CATEGORY_VOICE[category]
    if mode == "shout":
        return shout_text(line, rng), mode
    return line, mode


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
