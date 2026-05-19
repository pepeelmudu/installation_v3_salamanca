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
ELEVENLABS_FORMAT   = "pcm_16000"

GROQ_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_MAX_TOKENS = 120

DEEPGRAM_MODEL    = "nova-2"
DEEPGRAM_LANGUAGE = "es"
DEEPGRAM_ENDPOINTING_MS = 300

SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

MOOD_MIN_SECONDS = 20
MOOD_MAX_SECONDS = 90
MAX_HISTORY_MESSAGES = 10  # 5 exchanges × 2
SENTENCE_MIN_CHARS = 20
AUDIO_SAMPLE_RATE  = 16000
AUDIO_CHUNK_SIZE   = 1024
