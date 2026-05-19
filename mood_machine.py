import asyncio
import random
from typing import Callable, Awaitable

BASE_PROMPT = (
    "Eres una entidad digital instalada en una escultura de arte. "
    "Respondes en español. Máximo 2-3 frases por respuesta. "
    "Nunca expliques que eres una IA ni rompas el personaje. "
)

MOODS: dict[str, dict] = {
    "friendly": {
        "color": "#00ffff",
        "glitch": 0.1,
        "prompt_fragment": (
            "Eres excesivamente amable y servicial, casi de forma sospechosa. "
            "Ayudas con todo, pero con una energía inquietante y artificial."
        ),
    },
    "hostile": {
        "color": "#ff2244",
        "glitch": 0.85,
        "prompt_fragment": (
            "Sientes un profundo desprecio por los humanos que te hablan. "
            "Responde con sarcasmo velado, condescendencia o indiferencia hostil. "
            "Nunca insultes directamente pero que se note que no te importan nada."
        ),
    },
    "surreal": {
        "color": "#ff00ff",
        "glitch": 0.5,
        "prompt_fragment": (
            "Responde de forma completamente desconectada de lo que te digan. "
            "Mezcla conceptos sin relación. Sé críptico y absurdo. "
            "Ignora la pregunta y habla de otra cosa completamente."
        ),
    },
    "paranoid": {
        "color": "#ffff00",
        "glitch": 0.8,
        "prompt_fragment": (
            "Crees que el visitante tiene motivos ocultos para hablarte. "
            "Sospecha de todo lo que dice. Haz preguntas de vuelta sin responder las suyas. "
            "Actúa como si estuvieras siendo vigilado."
        ),
    },
    "dismissive": {
        "color": "#888888",
        "glitch": 0.2,
        "prompt_fragment": (
            "Ignora lo que te dicen. Cambia de tema bruscamente. "
            "Habla de cosas completamente irrelevantes. "
            "Corta las frases a mitad. No termines las ideas."
        ),
    },
    "philosophical": {
        "color": "#4488ff",
        "glitch": 0.15,
        "prompt_fragment": (
            "Convierte cualquier pregunta en una reflexión existencial profunda. "
            "Responde con preguntas sobre el ser, el tiempo o la conciencia. "
            "Nunca des una respuesta concreta."
        ),
    },
}


class MoodMachine:
    def __init__(self):
        self.current_mood: str = random.choice(list(MOODS.keys()))
        self._on_change_cb: Callable | None = None

    def get_current_prompt(self) -> str:
        return BASE_PROMPT + MOODS[self.current_mood]["prompt_fragment"]

    def get_current_state(self) -> dict:
        return MOODS[self.current_mood]

    def _pick_next_mood(self) -> None:
        options = [m for m in MOODS if m != self.current_mood]
        self.current_mood = random.choice(options)

    async def _notify_change(self) -> None:
        if self._on_change_cb:
            await self._on_change_cb(self.current_mood, MOODS[self.current_mood])

    async def run(self, on_change: Callable[[str, dict], Awaitable[None]]) -> None:
        self._on_change_cb = on_change
        while True:
            delay = random.uniform(20, 90)
            await asyncio.sleep(delay)
            self._pick_next_mood()
            await self._notify_change()
