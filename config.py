from dotenv import load_dotenv
import os

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


DEEPGRAM_API_KEY   = _require("DEEPGRAM_API_KEY")
GROQ_API_KEY       = _require("GROQ_API_KEY")
ELEVENLABS_API_KEY = _require("ELEVENLABS_API_KEY")

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL    = "eleven_flash_v2_5"
ELEVENLABS_MODEL_V3 = "eleven_v3"   # expressive, supports [shouts]/[whispers] tags
ELEVENLABS_FORMAT   = "pcm_24000"

GROQ_MODEL      = "llama-3.1-8b-instant"   # primary (fast, cheap, high free daily limit)
# On a 429 (daily free quota exhausted) the LLM client falls through this chain.
# Each model has its OWN separate free daily quota, so this multiplies headroom.
GROQ_FALLBACK_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "qwen/qwen-3-32b",
    "llama-3.3-70b-versatile",
]
GROQ_MAX_TOKENS = 120   # default cap (neutral). Expo overrides with a much shorter cap.

DEEPGRAM_MODEL          = "nova-2"
DEEPGRAM_LANGUAGE       = "es"
DEEPGRAM_ENDPOINTING_MS = 300

SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

MOOD_MIN_SECONDS    = 20
MOOD_MAX_SECONDS    = 90
MAX_HISTORY_MESSAGES = 10
SENTENCE_MIN_CHARS  = 20

# Browser sends audio at this rate; Deepgram receives at this rate.
BROWSER_CAPTURE_RATE = 16000
# ElevenLabs PCM output rate (pcm_24000 format).
AUDIO_PLAYBACK_RATE  = 24000

# Seconds of silence before the sculpture speaks proactively.
PROACTIVE_INTERVAL = int(os.getenv("PROACTIVE_INTERVAL", "180"))
