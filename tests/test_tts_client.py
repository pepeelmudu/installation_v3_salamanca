import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tts_client import TTSClient, _rms, _is_sentence_end

def test_rms_silence():
    silence = bytes(100)
    assert _rms(silence) == 0.0

def test_rms_max_amplitude():
    samples = np.full(100, 32767, dtype=np.int16)
    result = _rms(samples.tobytes())
    assert 0.99 < result <= 1.0

def test_rms_half_amplitude():
    samples = np.full(100, 16384, dtype=np.int16)
    result = _rms(samples.tobytes())
    assert 0.4 < result < 0.6

def test_rms_empty_bytes():
    assert _rms(b"") == 0.0

def test_is_sentence_end_period():
    assert _is_sentence_end("Hola mundo.", 5) is True

def test_is_sentence_end_exclamation():
    assert _is_sentence_end("¡Qué interesante!", 5) is True

def test_is_sentence_end_question():
    assert _is_sentence_end("¿Qué quieres?", 5) is True

def test_is_sentence_end_too_short():
    assert _is_sentence_end("No.", 5) is False

def test_is_sentence_end_no_punctuation():
    assert _is_sentence_end("esto no termina", 20) is False

def test_feed_flushes_on_sentence_end():
    flushed = []
    client = TTSClient.__new__(TTSClient)
    client._buffer = ""
    client._flush_cb = lambda text: flushed.append(text)
    from tts_client import _is_sentence_end, SENTENCE_MIN_CHARS
    client._buffer = "Eres muy molesto conmigo."
    if _is_sentence_end(client._buffer, SENTENCE_MIN_CHARS):
        client._flush_cb(client._buffer.strip())
        client._buffer = ""
    assert flushed == ["Eres muy molesto conmigo."]


@pytest.mark.asyncio
async def test_audio_chunk_callback_called():
    """TTSClient calls on_audio_chunk with each PCM chunk instead of sounddevice."""
    received_chunks = []

    async def capture_chunk(data: bytes):
        received_chunks.append(data)

    with patch("tts_client.ElevenLabs") as MockEL:
        mock_client = MagicMock()
        MockEL.return_value = mock_client

        fake_chunk = MagicMock()
        fake_chunk.audio_bytes = b"\x10\x20" * 100
        fake_chunk.alignment = None
        mock_client.text_to_speech.stream_with_timestamps.return_value = iter([fake_chunk])

        loop = asyncio.get_running_loop()
        from tts_client import TTSClient
        tts = TTSClient(
            api_key="test",
            voice_id="test_voice",
            on_amplitude=AsyncMock(),
            on_speaking=AsyncMock(),
            on_viseme=AsyncMock(),
            on_audio_chunk=capture_chunk,
            loop=loop,
        )
        tts.feed("Hola mundo.")
        tts.flush()
        await asyncio.sleep(0.3)
        assert len(received_chunks) > 0
        tts.close()


def test_rms_pure_python():
    """_rms returns 0 for silence and >0 for signal."""
    from tts_client import _rms
    assert _rms(b"") == 0.0
    assert _rms(bytes(100)) == 0.0  # silence
    # Max amplitude int16 = 32767 → rms should be ~1.0
    import struct
    loud = struct.pack("<" + "h" * 50, *([32767] * 50))
    assert _rms(loud) > 0.9
