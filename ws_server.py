import asyncio
import os
from typing import Callable, Awaitable
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from mood_machine import PERSONALITIES, DEFAULT_PERSONALITY

app = FastAPI()
connected_clients: set[WebSocket] = set()

# Single audio browser connection
_audio_client: WebSocket | None = None
# Called when browser sends audio bytes
_audio_receive_cb: Callable[[bytes], Awaitable[None]] | None = None
# Called with the personality id chosen on the setup screen (on /audio connect)
_personality_cb: Callable[[str], Awaitable[None]] | None = None


def set_audio_receive_callback(cb: Callable[[bytes], Awaitable[None]]) -> None:
    global _audio_receive_cb
    _audio_receive_cb = cb


def set_personality_callback(cb: Callable[[str], Awaitable[None]]) -> None:
    global _personality_cb
    _personality_cb = cb


async def send_audio_to_browser(data: bytes) -> None:
    if _audio_client:
        try:
            await _audio_client.send_bytes(data)
        except Exception:
            pass


async def send_audio_text(text: str) -> None:
    """Send a text/JSON message over the audio WebSocket (same channel as audio chunks)."""
    if _audio_client:
        try:
            await _audio_client.send_text(text)
        except Exception:
            pass


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/slop-textures")
async def slop_textures() -> JSONResponse:
    """List the face color textures so the browser can cycle through them."""
    d = os.path.join(os.path.dirname(__file__), "face", "models", "slop_textures", "GLITCH_OK")
    try:
        files = sorted(f for f in os.listdir(d)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    except OSError:
        files = []
    return JSONResponse({"items": [f"models/slop_textures/GLITCH_OK/{f}" for f in files]})


@app.get("/personalities")
async def personalities() -> JSONResponse:
    """List selectable personality profiles for the setup-screen dropdown."""
    return JSONResponse({
        "default": DEFAULT_PERSONALITY,
        "items": [{"id": pid, "name": p["name"]} for pid, p in PERSONALITIES.items()],
    })


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
    personality = websocket.query_params.get("personality", DEFAULT_PERSONALITY)
    print(f"[AUDIO] Browser connected (personality={personality})", flush=True)
    if _personality_cb:
        await _personality_cb(personality)
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
            elif chunks % 100 == 0:
                import struct
                samples = struct.unpack(f"<{len(data)//2}h", data)
                max_amp = max((abs(s) for s in samples), default=0)
                print(f"[AUDIO] chunks={chunks} max_amplitude={max_amp}", flush=True)
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
# Mount at root so `oracle.luker.one/` serves index.html directly.
# Defined AFTER all explicit routes (/health, /ws, /audio) so they take precedence.
app.mount("/", StaticFiles(directory=_FACE_DIR, html=True), name="face")
