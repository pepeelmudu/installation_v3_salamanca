import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_receive_audio_forwarded_to_deepgram():
    """receive_audio() sends bytes to the open Deepgram connection."""
    from stt_client import STTClient

    transcript_cb = AsyncMock()

    with patch("stt_client.AsyncDeepgramClient") as MockDG:
        mock_conn = AsyncMock()
        mock_conn.start_listening = AsyncMock(return_value=None)
        MockDG.return_value.listen.v1.connect.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        MockDG.return_value.listen.v1.connect.return_value.__aexit__ = AsyncMock(return_value=False)

        client = STTClient(api_key="test", on_transcript=transcript_cb)
        client._connection = mock_conn

        await client.receive_audio(b"\x00\x01" * 512)

        mock_conn.send_media.assert_called_once_with(b"\x00\x01" * 512)


@pytest.mark.asyncio
async def test_receive_audio_muted_sends_silence():
    """When muted, receive_audio() sends silence bytes instead of real data."""
    from stt_client import STTClient

    transcript_cb = AsyncMock()
    with patch("stt_client.AsyncDeepgramClient"):
        client = STTClient(api_key="test", on_transcript=transcript_cb)
        mock_conn = AsyncMock()
        client._connection = mock_conn
        client.set_muted(True)

        real_data = b"\xff\xfe" * 512
        await client.receive_audio(real_data)

        call_args = mock_conn.send_media.call_args[0][0]
        assert call_args == bytes(len(real_data))  # silence = zeros
