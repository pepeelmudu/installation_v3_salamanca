import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from stt_client import STTClient

@pytest.mark.asyncio
async def test_on_transcript_callback_called_on_speech_final():
    received = []

    async def cb(text):
        received.append(text)

    client = STTClient.__new__(STTClient)
    client._on_transcript = cb

    # Simulate a Deepgram result with speech_final=True
    result = MagicMock()
    result.speech_final = True
    result.channel.alternatives[0].transcript = "hola mundo"

    await client._handle_transcript(result)
    assert received == ["hola mundo"]

@pytest.mark.asyncio
async def test_on_transcript_callback_not_called_if_not_final():
    received = []

    async def cb(text):
        received.append(text)

    client = STTClient.__new__(STTClient)
    client._on_transcript = cb

    result = MagicMock()
    result.speech_final = False
    result.channel.alternatives[0].transcript = "texto parcial"

    await client._handle_transcript(result)
    assert received == []

@pytest.mark.asyncio
async def test_on_transcript_callback_not_called_if_empty():
    received = []

    async def cb(text):
        received.append(text)

    client = STTClient.__new__(STTClient)
    client._on_transcript = cb

    result = MagicMock()
    result.speech_final = True
    result.channel.alternatives[0].transcript = "   "

    await client._handle_transcript(result)
    assert received == []
