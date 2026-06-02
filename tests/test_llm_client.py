# tests/test_llm_client.py
import pytest
from unittest.mock import MagicMock, patch
from llm_client import LLMClient

def make_mock_chunk(content):
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk

def test_build_messages_includes_system_and_user():
    client = LLMClient(api_key="fake", model="fake-model")
    msgs = client._build_messages("hola", "eres un robot hostil")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "eres un robot hostil"
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "hola"

def test_build_messages_includes_history():
    client = LLMClient(api_key="fake", model="fake-model")
    client._history = [
        {"role": "user", "content": "pregunta anterior"},
        {"role": "assistant", "content": "respuesta anterior"},
    ]
    msgs = client._build_messages("nueva pregunta", "prompt")
    assert msgs[1]["content"] == "pregunta anterior"
    assert msgs[2]["content"] == "respuesta anterior"
    assert msgs[3]["content"] == "nueva pregunta"

def test_save_exchange_appends_to_history():
    client = LLMClient(api_key="fake", model="fake-model")
    client._save_exchange("user text", "assistant text")
    assert len(client._history) == 2
    assert client._history[0]["role"] == "user"
    assert client._history[1]["role"] == "assistant"

def test_save_exchange_trims_to_max_history():
    from config import MAX_HISTORY_MESSAGES
    client = LLMClient(api_key="fake", model="fake-model")
    for i in range(MAX_HISTORY_MESSAGES + 4):
        client._save_exchange(f"u{i}", f"a{i}")
    assert len(client._history) <= MAX_HISTORY_MESSAGES

def test_stream_yields_tokens(monkeypatch):
    client = LLMClient(api_key="fake", model="fake-model")
    chunks = [make_mock_chunk("Hola"), make_mock_chunk(" mundo"), make_mock_chunk(None)]
    mock_create = MagicMock(return_value=iter(chunks))
    monkeypatch.setattr(client._groq.chat.completions, "create", mock_create)
    tokens = list(client.stream("test", "system prompt"))
    assert tokens == ["Hola", " mundo"]

def test_generate_oneshot_does_not_touch_history(monkeypatch):
    client = LLMClient(api_key="fake", model="fake-model")
    resp = MagicMock()
    resp.choices[0].message.content = "  BITCOIN PUMPED 70%  "
    monkeypatch.setattr(client._groq.chat.completions, "create",
                        MagicMock(return_value=resp))
    out = client.generate_oneshot("system", "user")
    assert out == "BITCOIN PUMPED 70%"     # stripped
    assert client._history == []            # history untouched
