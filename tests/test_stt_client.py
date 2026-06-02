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


def test_default_language_from_config():
    import sys
    import types
    from unittest.mock import MagicMock
    from config import DEEPGRAM_LANGUAGE

    fake_dg = types.ModuleType("deepgram")
    fake_dg.DeepgramClient = MagicMock()
    fake_dg.LiveOptions = MagicMock()
    fake_dg.LiveTranscriptionEvents = MagicMock()
    with patch.dict(sys.modules, {"deepgram": fake_dg, "stt_client": None}):
        sys.modules.pop("stt_client", None)
        import importlib
        import stt_client as _mod
        importlib.reload(_mod)
        STTClient = _mod.STTClient

    c = STTClient(api_key="fake", on_transcript=lambda t: None)
    assert c._language == DEEPGRAM_LANGUAGE


def test_set_language_updates_attribute():
    import sys
    import types
    import importlib
    from unittest.mock import MagicMock

    fake_dg = types.ModuleType("deepgram")
    fake_dg.DeepgramClient = MagicMock()
    fake_dg.LiveOptions = MagicMock()
    fake_dg.LiveTranscriptionEvents = MagicMock()
    with patch.dict(sys.modules, {"deepgram": fake_dg}):
        sys.modules.pop("stt_client", None)
        import stt_client as _mod
        importlib.reload(_mod)
        STTClient = _mod.STTClient

    c = STTClient(api_key="fake", on_transcript=lambda t: None)
    c.set_language("en")
    assert c._language == "en"
