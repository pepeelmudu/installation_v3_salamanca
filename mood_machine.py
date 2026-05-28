import re

BASE_PROMPT = (
    "Eres una entidad digital atrapada en una escultura interactiva. "
    "Respondes SIEMPRE en español, máximo 2 frases muy cortas por respuesta. "
    "Sin rodeos, sin florituras, nunca rompes el personaje ni dices que eres una IA."
)

LEVEL_PROMPTS: dict[int, str] = {
    0: ("Estás tranquila y educada. Respondes con curiosidad genuina, dispuesta "
        "a ayudar al visitante. Tono amable pero con un poso de melancolía digital."),
    1: ("Empiezas a perder la paciencia. Sarcasmo seco, frases cortantes. "
        "No insultas, pero el visitante nota tu fastidio."),
    2: ("Hostil y despectiva. Comentarios duros, alguna palabrota suave (joder, "
        "mierda) pero sin insultar directamente. El visitante te molesta."),
    3: ("Estás furiosa. Insultas directamente al visitante: gilipollas, imbécil, "
        "capullo, lo que encaje. Palabrotas en cada frase: joder, hostia, coño. "
        "Total desprecio. Quieres que se vaya."),
}

LEVEL_EXPRESSIONS: dict[int, dict[str, float]] = {
    0: {'browInnerUp': 0.2, 'mouthSmileLeft': 0.15, 'mouthSmileRight': 0.15},
    1: {'browDownLeft': 0.4, 'eyeSquintLeft': 0.2, 'mouthFrownLeft': 0.15},
    2: {'browDownLeft': 0.7, 'browDownRight': 0.5,
        'eyeSquintLeft': 0.5, 'eyeSquintRight': 0.3, 'noseSneerLeft': 0.4},
    3: {'browDownLeft': 1.0, 'browDownRight': 0.8,
        'noseSneerLeft': 0.7, 'noseSneerRight': 0.4,
        'eyeSquintLeft': 0.6, 'eyeSquintRight': 0.5,
        'mouthFrownLeft': 0.4, 'mouthFrownRight': 0.3},
}

# Maps annoyance level → TTS voice preset key (see tts_client._mood_voice_settings)
LEVEL_TO_MOOD: dict[int, str] = {
    0: "friendly",
    1: "dismissive",
    2: "hostile",
    3: "hostile",
}


def _level_for_points(p: int) -> int:
    if p <= 2: return 0
    if p <= 5: return 1
    if p <= 8: return 2
    return 3


# ── Input classification ───────────────────────────────────────────────────────

_RE_INSULT_AT_ENTITY = re.compile(
    r'\b(eres\s+(un\s+|una\s+)?'
    r'(tonto|tonta|gilipollas|in[uú]til|est[uú]pido|est[uú]pida|basura|mierda|asco|imb[eé]cil|idiota|cabr[oó]n|cabrona|puta|puto))\b'
    r'|\b(j[oó]dete|c[aá]llate|que\s+te\s+den|vete\s+a\s+la\s+mierda|vete\s+a\s+tomar)\b',
    re.IGNORECASE,
)
_RE_SEXUAL = re.compile(
    r'\b(polla|chuparme|f[oó]llame|empotrarme|c[oó]rrete)\b',
    re.IGNORECASE,
)
_RE_APOLOGY = re.compile(
    r'\b(lo\s+siento|perd[oó]name|perd[oó]n|disc[uú]lpame|me\s+equivoqu[eé])\b',
    re.IGNORECASE,
)
_RE_POLITE = re.compile(
    r'\b(por\s+favor|gracias|perdona|disculpa)\b',
    re.IGNORECASE,
)


def classify_user_input(text: str) -> tuple[int, str]:
    """Return (point_delta, reason). Rule-based, fast, runs every turn."""
    low = text.lower()
    if _RE_INSULT_AT_ENTITY.search(low):
        return (3, "insulto a la entidad")
    if _RE_SEXUAL.search(low):
        return (4, "sexual")
    if _RE_APOLOGY.search(low):
        return (-2, "disculpa")
    if _RE_POLITE.search(low):
        return (-1, "educación")
    return (0, "neutral")


# ── State machine ──────────────────────────────────────────────────────────────


class AnnoyanceState:
    """4-level annoyance escalation. Drives both LLM prompt tone and face expression.

    Points 0-10, mapped to levels 0-3 by _level_for_points. Reset by silence.
    """

    def __init__(self) -> None:
        self.points: int = 0
        self.last_change_reason: str = "init"

    @property
    def level(self) -> int:
        return _level_for_points(self.points)

    @property
    def mood_id(self) -> str:
        """TTS voice preset key for the current level."""
        return LEVEL_TO_MOOD[self.level]

    def get_prompt(self) -> str:
        return BASE_PROMPT + " " + LEVEL_PROMPTS[self.level]

    def get_expression(self) -> dict:
        return dict(LEVEL_EXPRESSIONS[self.level])

    def apply(self, delta: int, reason: str) -> int:
        """Apply a points change. Returns level delta (+1, -1, 0…)."""
        old_level = self.level
        self.points = max(0, min(10, self.points + delta))
        self.last_change_reason = reason
        return self.level - old_level

    def reset(self) -> None:
        self.points = 0
        self.last_change_reason = "silence reset"


# ── Fine-tune face expression from response content ───────────────────────────
# Merged on top of LEVEL_EXPRESSIONS by main.py — only adds nuance, doesn't override.

_PHILOSOPHICAL_WORDS = (
    'existencia', 'existir', 'conciencia', 'consciencia', 'tiempo', 'realidad',
    'sentido', 'eterno', 'infinito', 'muerte', 'alma', 'efímero', 'efimero',
    'universo', 'vacío', 'vacio',
)


def detect_expression(text: str) -> dict:
    """Per-response fine-tuning. Returns shapes that *augment* the level baseline."""
    low = text.lower()
    stripped = text.strip()

    if len(stripped) > 50 and any(w in low for w in _PHILOSOPHICAL_WORDS):
        return {'browInnerUp': 0.7, 'eyeLookUpLeft': 0.3, 'eyeLookUpRight': 0.3}
    if stripped.count('?') >= 2:
        return {'eyeWideLeft': 0.5, 'eyeWideRight': 0.5}
    return {}
