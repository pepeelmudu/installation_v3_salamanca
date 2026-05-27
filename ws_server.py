import asyncio
import os
from typing import Callable, Awaitable
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
connected_clients: set[WebSocket] = set()

# Single audio browser connection
_audio_client: WebSocket | None = None
# Called when browser sends audio bytes
_audio_receive_cb: Callable[[bytes], Awaitable[None]] | None = None


def set_audio_receive_callback(cb: Callable[[bytes], Awaitable[None]]) -> None:
    global _audio_receive_cb
    _audio_receive_cb = cb


async def send_audio_to_browser(data: bytes) -> None:
    if _audio_client:
        try:
            await _audio_client.send_bytes(data)
        except Exception:
            pass


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        connected_clients.discard(websocket)


@app.websocket("/audio")
async def audio_endpoint(websocket: WebSocket) -> None:
    global _audio_client
    await websocket.accept()
    _audio_client = websocket
    print("[AUDIO] Browser connected", flush=True)
    chunks = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            chunks += 1
            if chunks == 1:
                import struct
                samples = struct.unpack(f"<{len(data)//2}h", data)
                max_amp = max(abs(s) for s in samples)
                print(f"[AUDIO] First chunk: len={len(data)} max_amplitude={max_amp}", flush=True)
            elif chunks % 200 == 0:
                print(f"[AUDIO] chunks={chunks} bytes={len(data)}", flush=True)
            if _audio_receive_cb:
                await _audio_receive_cb(data)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        print(f"[AUDIO] Browser disconnected after {chunks} chunks", flush=True)
        if _audio_client is websocket:
            _audio_client = None


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
