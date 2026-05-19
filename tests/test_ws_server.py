# tests/test_ws_server.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from ws_server import app, broadcast, connected_clients

def test_face_static_redirect():
    client = TestClient(app)
    response = client.get("/face/")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_broadcast_sends_to_connected_clients():
    from unittest.mock import AsyncMock
    mock_ws = AsyncMock()
    connected_clients.add(mock_ws)
    await broadcast({"type": "mood_change", "mood": "hostile"})
    mock_ws.send_json.assert_called_once_with({"type": "mood_change", "mood": "hostile"})
    connected_clients.discard(mock_ws)

@pytest.mark.asyncio
async def test_broadcast_removes_disconnected_client():
    from unittest.mock import AsyncMock
    from websockets.exceptions import ConnectionClosedOK
    mock_ws = AsyncMock()
    mock_ws.send_json.side_effect = Exception("disconnected")
    connected_clients.add(mock_ws)
    await broadcast({"type": "test"})
    assert mock_ws not in connected_clients
