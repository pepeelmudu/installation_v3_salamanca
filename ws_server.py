import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

app = FastAPI()
connected_clients: set[WebSocket] = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        connected_clients.discard(websocket)


async def broadcast(event: dict) -> None:
    dead = set()
    for ws in list(connected_clients):
        try:
            await ws.send_json(event)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


_FACE_DIR = os.path.join(os.path.dirname(__file__), "face")
app.mount("/face", StaticFiles(directory=_FACE_DIR, html=True), name="face")
